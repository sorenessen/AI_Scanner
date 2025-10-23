# app.py (upgraded reference model + MPS + long-text chunking + readable summary)
import os, csv, time, hashlib, pathlib
from typing import List, Dict, Optional
from pydantic import BaseModel
import math

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- Config --------------------
MODEL_NAME = os.getenv("REF_MODEL", "EleutherAI/gpt-neo-1.3B")
MAX_TOKENS_PER_PASS = 768
STRIDE = 128
USE_FP16 = True

# -------------------- Device setup --------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

DTYPE = torch.float16 if (USE_FP16 and (DEVICE.type in {"cuda", "mps"})) else torch.float32

# -------------------- App + CORS --------------------
app = FastAPI(title="AI Text Scanner (MVP, Large Ref Model)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Model --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)
model.eval().to(DEVICE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------- IO Models --------------------
class ScanIn(BaseModel):
    text: str
    tag: Optional[str] = None

# -------------------- Helpers --------------------
def scan_chunk(input_ids: torch.Tensor) -> Dict:
    with torch.no_grad():
        out = model(input_ids=input_ids)
        logits = out.logits
    ids = input_ids[0]
    per_token = []
    topk_bins = {10: 0, 100: 0, 1000: 0, "rest": 0}
    total = max(0, ids.size(0) - 1)
    logprobs = []

    for i in range(1, ids.size(0)):
        next_logits = logits[0, i - 1]
        probs = torch.softmax(next_logits, dim=-1)
        actual_id = ids[i].item()
        p_actual = probs[actual_id].item()
        rank = int((probs > p_actual).sum().item() + 1)

        if rank <= 10:
            topk_bins[10] += 1
        elif rank <= 100:
            topk_bins[100] += 1
        elif rank <= 1000:
            topk_bins[1000] += 1
        else:
            topk_bins["rest"] += 1

        tok_str = tokenizer.convert_ids_to_tokens([actual_id])[0]
        per_token.append({"t": tok_str, "rank": rank, "p": p_actual})
        logprobs.append(math.log(max(p_actual, 1e-12)))

    ppl = math.exp(-sum(logprobs) / max(1, len(logprobs)))
    mean_lp = sum(logprobs) / max(1, len(logprobs))
    burstiness = sum((lp - mean_lp) ** 2 for lp in logprobs) / max(1, len(logprobs))

    if total > 0:
        frac10 = topk_bins[10] / total
        frac100 = topk_bins[100] / total
        overall = min(1.0, 0.75 * frac10 + 0.35 * frac100)
    else:
        overall = 0.0

    return {
        "per_token": per_token,
        "bins": topk_bins,
        "total": total,
        "score": overall,
        "ppl": ppl,
        "burstiness": burstiness,
    }


def chunked_scan(text: str) -> Dict:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    if ids.size(0) <= MAX_TOKENS_PER_PASS:
        input_ids = ids.unsqueeze(0).to(DEVICE)
        return scan_chunk(input_ids)

    all_tokens: List[Dict] = []
    agg_bins = {10: 0, 100: 0, 1000: 0, "rest": 0}
    agg_total = 0
    scores, ppls, bursts = [], [], []

    start = 0
    while start < ids.size(0):
        end = min(start + MAX_TOKENS_PER_PASS, ids.size(0))
        chunk_ids = ids[start:end].unsqueeze(0).to(DEVICE)
        result = scan_chunk(chunk_ids)

        all_tokens.extend(result["per_token"])
        for k in agg_bins:
            agg_bins[k] += result["bins"][k]
        agg_total += result["total"]
        scores.append(result["score"])
        ppls.append(result["ppl"])
        bursts.append(result["burstiness"])

        if end == ids.size(0):
            break
        start = end - STRIDE

    if agg_total > 0:
        frac10 = agg_bins[10] / agg_total
        frac100 = agg_bins[100] / agg_total
        overall = min(1.0, 0.75 * frac10 + 0.35 * frac100)
    else:
        overall = 0.0

    return {
        "per_token": all_tokens,
        "bins": agg_bins,
        "total": agg_total,
        "score": overall,
        "ppl": sum(ppls) / len(ppls) if ppls else 0.0,
        "burstiness": sum(bursts) / len(bursts) if bursts else 0.0,
    }


# -------- CSV logging --------
LOG_PATH = os.getenv("SCAN_LOG_PATH", "scan_logs.csv")
_FIELDNAMES = [
    "ts", "text_sha256", "text_len_chars", "text_len_tokens",
    "model_name", "device", "dtype",
    "overall_score", "top10_frac", "top100_frac", "ppl", "burstiness",
    "ai_likelihood_calibrated", "tag"
]

def _ensure_log_header(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_FIELDNAMES).writeheader()

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def log_scan_row(row: dict):
    """
    Writes each scan to scan_logs.csv and a readable scan_summaries.txt file.
    Ensures consistent, clean CSV formatting with headers and verdicts.
    """
    pathlib.Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    csv_exists = os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0

    verdict = (
        "Likely AI-generated" if row["ai_likelihood_calibrated"] >= 0.6
        else "Likely human-written"
    )

    # Open in append mode
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        # Write header if file is new
        if not csv_exists:
            writer.writerow([
                "# Legend:",
                "Each row = one scan result.",
                "Likelihood % is the probability text was AI-generated.",
                "Verdict is a simple interpretation of the likelihood value."
            ])
            writer.writerow([])
            writer.writerow([
                "Timestamp",
                "Verdict",
                "Likelihood %",
                "Predictability Score",
                "Top-10 %",
                "Top-100 %",
                "Perplexity",
                "Burstiness",
                "Model",
                "Device",
                "Chars",
                "Tokens",
                "Text Hash",
                "Tag"
            ])

        # Write the main data row
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["ts"])),
            verdict,
            f"{int(row['ai_likelihood_calibrated'] * 100)}%",
            f"{row['overall_score']:.2f}",
            f"{row['top10_frac'] * 100:.1f}%",
            f"{row['top100_frac'] * 100:.1f}%",
            f"{row['ppl']:.1f}",
            f"{row['burstiness']:.2f}",
            row["model_name"],
            row["device"],
            row["text_len_chars"],
            row["text_len_tokens"],
            row["text_sha256"][:8],
            row["tag"],
        ])

    # ---------- Write summary to plain text ----------
    summary_path = os.path.join(os.path.dirname(LOG_PATH), "scan_summaries.txt")
    likelihood_pct = int(row["ai_likelihood_calibrated"] * 100)
    verdict_text = (
        "almost certainly AI-generated" if likelihood_pct >= 85 else
        "very likely AI-generated" if likelihood_pct >= 70 else
        "possibly AI-generated" if likelihood_pct >= 50 else
        "likely human-written"
    )

    summary = (
        f"\n--- Scan Summary ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(row['ts']))}) ---\n"
        f"Verdict: {verdict_text.capitalize()}.\n"
        f"Likelihood this was AI-generated: {likelihood_pct}%.\n"
        f"Predictability Score: {row['overall_score']:.2f}\n"
        f"Top-10 Tokens: {row['top10_frac'] * 100:.1f}%  |  Top-100 Tokens: {row['top100_frac'] * 100:.1f}%\n"
        f"Perplexity: {row['ppl']:.1f}  |  Burstiness: {row['burstiness']:.2f}\n"
        f"Model: {row['model_name']}  |  Device: {row['device']}\n"
        f"Text Length: {row['text_len_chars']} chars / {row['text_len_tokens']} tokens\n"
        f"Tag: {row['tag'] or '(none)'}\n"
        "------------------------------------------------------------\n"
    )

    with open(summary_path, "a") as txt:
        txt.write(summary)




# -------------------- Route --------------------
@app.post("/scan")
def scan(inp: ScanIn):
    text = inp.text.strip()
    if not text:
        return {"overall_score": 0.0, "per_token": [], "explanation": "Empty text"}

    out = chunked_scan(text)
    total = max(1, out["total"])
    bins = out["bins"]
    frac10 = bins[10] / total
    frac100 = bins[100] / total

    # --- calibrated probability ---
    fPpl   = max(0, min(1, (25 - (out['ppl'] or 25)) / 20))
    fBurst = max(0, min(1, (8 - (out['burstiness'] or 8)) / 6))
    z = (2.3 * out["score"]) + (0.9 * fPpl) + (0.6 * fBurst) + (0.5 * frac10) - 1.5
    calP = 1 / (1 + math.exp(-4 * z))
    calP = max(0, min(1, calP))

    # --- log row ---
    log_scan_row({
        "ts": int(time.time()),
        "text_sha256": _sha256_text(text),
        "text_len_chars": len(text),
        "text_len_tokens": total,
        "model_name": MODEL_NAME,
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "overall_score": round(out["score"], 6),
        "top10_frac": round(frac10, 6),
        "top100_frac": round(frac100, 6),
        "ppl": round(out["ppl"], 6),
        "burstiness": round(out["burstiness"], 6),
        "ai_likelihood_calibrated": round(calP, 6),
        "tag": (inp.tag or "")
    })

    # -------- readable explanation --------
    percent = round(calP * 100)
    if percent >= 85:
        summary = f"Likelihood this was AI-generated: {percent}% — almost certain."
    elif percent >= 70:
        summary = f"Likelihood this was AI-generated: {percent}% — high likelihood."
    elif percent >= 50:
        summary = f"Likelihood this was AI-generated: {percent}% — moderate likelihood."
    elif percent >= 30:
        summary = f"Likelihood this was AI-generated: {percent}% — low to moderate likelihood."
    else:
        summary = f"Likelihood this was AI-generated: {percent}% — unlikely."

    exp = (
        f"{round(100*frac10)}% of tokens in Top-10; "
        f"{round(100*frac100)}% in Top-100. "
        f"Perplexity≈{out['ppl']:.1f}; Burstiness≈{out['burstiness']:.3f}. "
        "Higher Top-10 fractions and lower perplexity typically correlate with model-like text. "
        + summary
    )

    return {
        "overall_score": out["score"],
        "per_token": out["per_token"],
        "explanation": exp,
        "ppl": out["ppl"],
        "burstiness": out["burstiness"],
        "bins": bins,
        "total": total,
        "model_name": MODEL_NAME,
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "calibrated_prob": calP
    }


# # app.py (upgraded reference model + MPS + long-text chunking)
# import os, csv, time, hashlib, pathlib
# from typing import List, Dict, Optional
# from pydantic import BaseModel
# import math

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # -------------------- Config --------------------
# MODEL_NAME = os.getenv("REF_MODEL", "EleutherAI/gpt-neo-1.3B")
# # Good alternatives:
# #   "EleutherAI/pythia-1.4b"
# #   "gpt2-medium"  (lighter fallback)
# MAX_TOKENS_PER_PASS = 768      # context per forward pass (safe for 1.3B on MPS)
# STRIDE = 128                   # overlap between chunks so boundary tokens get scored
# USE_FP16 = True                # try half precision on MPS or CUDA

# # -------------------- Device setup --------------------
# if torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
# elif torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
# else:
#     DEVICE = torch.device("cpu")

# DTYPE = torch.float16 if (USE_FP16 and (DEVICE.type in {"cuda", "mps"})) else torch.float32

# # -------------------- App + CORS --------------------
# app = FastAPI(title="AI Text Scanner (MVP, Large Ref Model)")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # tighten later
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------- Model --------------------
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)
# model.eval().to(DEVICE)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # -------------------- IO Models --------------------
# class ScanIn(BaseModel):
#     text: str
#     tag: Optional[str] = None   # <- new (can be "ai", "human", or empty)

# # -------------------- Helpers --------------------
# def scan_chunk(input_ids: torch.Tensor) -> Dict:
#     """
#     input_ids: [1, seq_len] on DEVICE
#     returns dict with per-token ranks and probabilities
#     """
#     with torch.no_grad():
#         out = model(input_ids=input_ids)
#         logits = out.logits  # [1, seq_len, vocab]
#     ids = input_ids[0]
#     per_token = []
#     topk_bins = {10: 0, 100: 0, 1000: 0, "rest": 0}
#     total = max(0, ids.size(0) - 1)

#     # Compute logprobs for perplexity/burstiness
#     logprobs = []

#     for i in range(1, ids.size(0)):
#         next_logits = logits[0, i - 1]  # predicts token i
#         probs = torch.softmax(next_logits, dim=-1)

#         actual_id = ids[i].item()
#         p_actual = probs[actual_id].item()

#         # Rank = 1 + number of tokens with prob > p_actual
#         rank = int((probs > p_actual).sum().item() + 1)

#         if rank <= 10:
#             topk_bins[10] += 1
#         elif rank <= 100:
#             topk_bins[100] += 1
#         elif rank <= 1000:
#             topk_bins[1000] += 1
#         else:
#             topk_bins["rest"] += 1

#         tok_str = tokenizer.convert_ids_to_tokens([actual_id])[0]
#         per_token.append({"t": tok_str, "rank": rank, "p": p_actual})
#         # for perplexity metrics
#         logprobs.append(math.log(max(p_actual, 1e-12)))

#     # Perplexity over this chunk (exclude first token)
#     ppl = math.exp(-sum(logprobs) / max(1, len(logprobs)))
#     # Burstiness: variance of log-probs (higher variance = more human-like “unevenness”)
#     mean_lp = sum(logprobs) / max(1, len(logprobs))
#     burstiness = sum((lp - mean_lp) ** 2 for lp in logprobs) / max(1, len(logprobs))

#     # Heuristic overall score from rank bins
#     if total > 0:
#         frac10 = topk_bins[10] / total
#         frac100 = topk_bins[100] / total
#         overall = min(1.0, 0.75 * frac10 + 0.35 * frac100)
#     else:
#         overall = 0.0

#     return {
#         "per_token": per_token,
#         "bins": topk_bins,
#         "total": total,
#         "score": overall,
#         "ppl": ppl,
#         "burstiness": burstiness,
#     }

# def chunked_scan(text: str) -> Dict:
#     enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
#     ids = enc["input_ids"][0]
#     if ids.size(0) <= MAX_TOKENS_PER_PASS:
#         input_ids = ids.unsqueeze(0).to(DEVICE)
#         return scan_chunk(input_ids)

#     # Long text: slide over with stride
#     all_tokens: List[Dict] = []
#     agg_bins = {10: 0, 100: 0, 1000: 0, "rest": 0}
#     agg_total = 0
#     scores = []
#     ppls = []
#     bursts = []

#     start = 0
#     while start < ids.size(0):
#         end = min(start + MAX_TOKENS_PER_PASS, ids.size(0))
#         chunk_ids = ids[start:end].unsqueeze(0).to(DEVICE)
#         result = scan_chunk(chunk_ids)

#         # Keep all tokens except the very first one of the document
#         # (but that nuance is small—fine for MVP)
#         all_tokens.extend(result["per_token"])
#         for k in agg_bins:
#             agg_bins[k] += result["bins"][k]
#         agg_total += result["total"]
#         scores.append(result["score"])
#         ppls.append(result["ppl"])
#         bursts.append(result["burstiness"])

#         if end == ids.size(0):
#             break
#         start = end - STRIDE  # overlap

#     # Aggregate across chunks
#     if agg_total > 0:
#         frac10 = agg_bins[10] / agg_total
#         frac100 = agg_bins[100] / agg_total
#         overall = min(1.0, 0.75 * frac10 + 0.35 * frac100)
#     else:
#         overall = 0.0

#     return {
#         "per_token": all_tokens,
#         "bins": agg_bins,
#         "total": agg_total,
#         "score": overall,
#         "ppl": sum(ppls) / len(ppls) if ppls else 0.0,
#         "burstiness": sum(bursts) / len(bursts) if bursts else 0.0,
#     }

#     # -------- CSV logging --------
# LOG_PATH = os.getenv("SCAN_LOG_PATH", "scan_logs.csv")

# _FIELDNAMES = [
#     "ts", "text_sha256", "text_len_chars", "text_len_tokens",
#     "model_name", "device", "dtype",
#     "overall_score", "top10_frac", "top100_frac", "ppl", "burstiness",
#     "ai_likelihood_calibrated", "tag"  # tag left blank for you to label later
# ]

# def _ensure_log_header(path: str):
#     if not os.path.exists(path) or os.path.getsize(path) == 0:
#         with open(path, "w", newline="") as f:
#             csv.DictWriter(f, fieldnames=_FIELDNAMES).writeheader()

# def _sha256_text(s: str) -> str:
#     return hashlib.sha256(s.encode("utf-8")).hexdigest()

# def log_scan_row(row: dict):
#     pathlib.Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
#     _ensure_log_header(LOG_PATH)
#     with open(LOG_PATH, "a", newline="") as f:
#         w = csv.DictWriter(f, fieldnames=_FIELDNAMES)
#         w.writerow(row)


# # -------------------- Route --------------------
# @app.post("/scan")
# def scan(inp: ScanIn):
#     text = inp.text.strip()
#     if not text:
#         return {
#             "overall_score": 0.0,
#             "per_token": [],
#             "explanation": "Empty text",
#             "ppl": None,
#             "burstiness": None,
#         }

#     out = chunked_scan(text)

#     # Build explanation
#     total = max(1, out["total"])
#     bins = out["bins"]
#     exp = (
#         f"{round(100*(bins[10]/total))}% of tokens in Top-10; "
#         f"{round(100*(bins[100]/total))}% in Top-100. "
#         f"Perplexity≈{out['ppl']:.1f}; Burstiness≈{out['burstiness']:.3f}. "
#         "Higher Top-10 fractions and lower perplexity typically correlate with model-like text."
#     )

#     total = max(1, out["total"])
#     bins = out["bins"]
#     frac10 = bins[10] / total
#     frac100 = bins[100] / total

#     # --- simple calibrated probability (same logic your frontend uses; optional) ---
#     # Map perplexity & burstiness to 0..1 “AI-ness” helpers
#     fPpl   = max(0, min(1, (25 - (out['ppl'] or 25)) / 20))
#     fBurst = max(0, min(1, (8 - (out['burstiness'] or 8)) / 6))
#     z = (2.3 * out["score"]) + (0.9 * fPpl) + (0.6 * fBurst) + (0.5 * frac10) - 1.5
#     calP = 1 / (1 + math.exp(-4 * z))
#     calP = max(0, min(1, calP))

#     # --- write CSV row ---
#     log_scan_row({
#         "ts": int(time.time()),
#         "text_sha256": _sha256_text(text),
#         "text_len_chars": len(text),
#         "text_len_tokens": total,
#         "model_name": MODEL_NAME,
#         "device": str(DEVICE),
#         "dtype": str(DTYPE),
#         "overall_score": round(out["score"], 6),
#         "top10_frac": round(frac10, 6),
#         "top100_frac": round(frac100, 6),
#         "ppl": round(out["ppl"], 6),
#         "burstiness": round(out["burstiness"], 6),
#         "ai_likelihood_calibrated": round(calP, 6),
#         "tag": (inp.tag or "")      # <- use the value sent from the UI
#     })



#     exp = (
#         f"{round(100*frac10)}% of tokens in Top-10; "
#         f"{round(100*frac100)}% in Top-100. "
#         f"Perplexity≈{out['ppl']:.1f}; Burstiness≈{out['burstiness']:.3f}. "
#         "Higher Top-10 fractions and lower perplexity typically correlate with model-like text."
#     )

#     return {
#         "overall_score": out["score"],
#         "per_token": out["per_token"],
#         "explanation": exp,
#         "ppl": out["ppl"],
#         "burstiness": out["burstiness"],
#         "bins": bins,
#         "total": total,
#         "model_name": MODEL_NAME,
#         "device": str(DEVICE),
#         "dtype": str(DTYPE),
#         "calibrated_prob": calP
#     }
