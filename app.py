# app.py — AI Text Scanner with classic-literature guard (single-file, no extra modules)

import os, csv, time, hashlib, pathlib, math, re
from typing import List, Dict, Optional

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- PDF (ReportLab) ----
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.barcharts import VerticalBarChart

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
app = FastAPI(title="AI Text Scanner (Classic Guard)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Model (safe load) --------------------
tokenizer = None
model = None
model_load_error = None

print("[server] loading model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    model.to(DEVICE)
    if DTYPE == torch.float16 and hasattr(model, "half"):
        model = model.half()
    elif hasattr(model, "float"):
        model = model.float()

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[server] model loaded OK on", DEVICE, "dtype", DTYPE)

except Exception as e:
    model_load_error = str(e)
    print("[server] FAILED TO LOAD MODEL:", model_load_error)

# -------------------- IO Models --------------------
class ScanIn(BaseModel):
    text: str
    tag: Optional[str] = None  # optional; if contains "classic" we apply the tighter cap

# -------------------- Core scoring helpers --------------------
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

    def _scan_ids(slice_ids: torch.Tensor) -> Dict:
        return scan_chunk(slice_ids.to(DEVICE))

    if ids.size(0) <= MAX_TOKENS_PER_PASS:
        input_ids = ids.unsqueeze(0)
        return _scan_ids(input_ids)

    all_tokens: List[Dict] = []
    agg_bins = {10: 0, 100: 0, 1000: 0, "rest": 0}
    agg_total = 0
    scores, ppls, bursts = [], [], []

    start = 0
    while start < ids.size(0):
        end = min(start + MAX_TOKENS_PER_PASS, ids.size(0))
        chunk_ids = ids[start:end].unsqueeze(0)
        result = _scan_ids(chunk_ids)

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

# -------------------- Category / Classic detection (in-file, no extras) --------------------
ARCHAIC_TOKENS = r"\b(thou|thee|thy|thine|ye|hath|doth|dost|art|shalt|whence|wherefore|ere|oft|nay|aye)\b"
CLASSIC_CUES   = r"\b(whilst|whereupon|thereof|therein|therewith|herein|hereby|forthwith|betwixt|methinks|thereupon|wherein)\b"

def looks_classic_like(text: str, metrics: Dict, frac10: float) -> bool:
    arch_hits = len(re.findall(ARCHAIC_TOKENS, text, flags=re.I))
    classic_hits = len(re.findall(CLASSIC_CUES,   text, flags=re.I))
    ppl   = metrics.get("ppl", 99.0)
    burst = metrics.get("burstiness", 99.0)
    return (arch_hits >= 2 or classic_hits >= 2) and (ppl <= 20.0) and (burst <= 9.0) and (frac10 >= 0.55)

def tag_says_classic(tag: Optional[str]) -> bool:
    if not tag:
        return False
    t = tag.lower()
    return any(k in t for k in ("classic", "pre-1920", "public_domain", "public-domain", "pre1920"))

def category_note_for_report(is_classic: bool) -> str:
    return ("Classic literature: dampened predictability; avoiding false positives."
            if is_classic else "Default calibration.")

# -------------------- Logging + PDF --------------------
LOG_PATH = os.path.join(".", "scan_logs.csv")

def create_pdf_summary(row: dict):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(row["ts"]))
    pdf_path = os.path.join(os.path.dirname(LOG_PATH), f"scan_summary_{timestamp}.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    likelihood_pct = int(row["ai_likelihood_calibrated"] * 100)
    verdict = ("Almost certainly AI-generated" if likelihood_pct >= 85 else
               "Very likely AI-generated" if likelihood_pct >= 70 else
               "Possibly AI-generated" if likelihood_pct >= 50 else
               "Likely human-written")

    story.append(Paragraph("<b>AI Text Scan Report</b>", styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(f"<b>Verdict:</b> {verdict}", styles["Heading2"]))
    story.append(Paragraph(f"<b>Likelihood:</b> {likelihood_pct}%", styles["Normal"]))
    story.append(Paragraph(f"<b>Detected style:</b> {row.get('category','other').replace('_',' ')} "
                           f"(conf {int(100*row.get('category_conf',0))}%)", styles["Normal"]))
    story.append(Paragraph(f"<i>{row.get('category_note','')}</i>", styles["Normal"]))
    story.append(Spacer(1, 0.15 * inch))

    data = [
        ["Metric", "Value"],
        ["Predictability Score", f"{row['overall_score']:.2f}"],
        ["Top-10 Tokens", f"{row['top10_frac']*100:.1f}%"],
        ["Top-100 Tokens", f"{row['top100_frac']*100:.1f}%"],
        ["Perplexity", f"{row['ppl']:.1f}"],
        ["Burstiness", f"{row['burstiness']:.2f}"],
        ["Model", row["model_name"]],
        ["Device", row["device"]],
        ["Text Length", f"{row['text_len_chars']} chars / {row['text_len_tokens']} tokens"],
        ["Tag", row["tag"] or "(none)"],
    ]
    table = Table(data, colWidths=[2.5 * inch, 3.5 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>Likelihood Indicator</b>", styles["Heading3"]))
    color = (colors.red if likelihood_pct >= 85 else
             colors.orange if likelihood_pct >= 60 else
             colors.yellow if likelihood_pct >= 40 else
             colors.green)
    d = Drawing(420, 40)
    d.add(Rect(0, 10, 4 * likelihood_pct, 20, fillColor=color))
    d.add(Rect(0, 10, 400, 20, fillColor=None, strokeColor=colors.black))
    d.add(String(410, 15, f"{likelihood_pct}%", fontSize=12))
    story.append(d)
    story.append(Spacer(1, 0.25 * inch))

    chart = VerticalBarChart()
    chart.x, chart.y = 50, 30
    chart.height, chart.width = 150, 400
    chart.data = [[row["top10_frac"]*100, row["top100_frac"]*100, row["ppl"], row["burstiness"]]]
    chart.categoryAxis.categoryNames = ["Top-10 %", "Top-100 %", "Perplexity", "Burstiness"]
    chart.bars[0].fillColor = colors.darkblue
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(100, row["ppl"] + 10)
    chart.valueAxis.valueStep = 20
    d2 = Drawing(500, 200)
    d2.add(chart)
    story.append(d2)

    doc.build(story)
    return pdf_path

def log_scan_row(row: dict):
    LOG_PATH = os.path.join(".", "scan_logs.csv")
    pathlib.Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    csv_exists = os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if not csv_exists:
            writer.writerow(["# Legend:", "Each row = one scan result.",
                             "Likelihood % = probability text was AI-generated.",
                             "Verdict interprets likelihood; Category is auto-detected (with classic guard)."])
            writer.writerow([])
            writer.writerow([
                "Timestamp","Verdict","Likelihood %","Predictability Score",
                "Top-10 %","Top-100 %","Perplexity","Burstiness",
                "Model","Device","Chars","Tokens","Text Hash","Tag",
                "Category","Category Conf","Category Note"
            ])

        verdict = ("Likely AI-generated"
                   if row["ai_likelihood_calibrated"] >= 0.6
                   else "Likely human-written")

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
            row["tag"] or "",
            row.get("category","other"),
            f"{int(100*row.get('category_conf',0))}%",
            row.get("category_note",""),
        ])

    pdf_path = create_pdf_summary(row)
    print(f"[+] PDF report created: {pdf_path}")

# -------------------- Main route --------------------
@app.post("/scan")
def scan(inp: ScanIn):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail=f"Model unavailable: {model_load_error}")

    text = inp.text.strip()
    if not text:
        return {"overall_score": 0.0, "per_token": [], "explanation": "Empty text"}

    out = chunked_scan(text)

    total = max(1, out["total"])
    bins = out["bins"]
    frac10 = bins[10] / total
    frac100 = bins[100] / total

    # crude “score” + helpers
    fPpl   = max(0, min(1, (25 - (out['ppl'] or 25)) / 20))
    fBurst = max(0, min(1, (8 - (out['burstiness'] or 8)) / 6))

    # category-ish detection (only classic vs other for now)
    is_classic_auto = looks_classic_like(text, out, frac10)
    cat = "classic_literature" if is_classic_auto else "other"
    cat_conf = 0.75 if is_classic_auto else 0.5

    # basic logistic combiner
    z = (2.0 * out["score"]) + (0.9 * fPpl) + (0.8 * fBurst) + (0.35 * frac10) - 1.6
    calP = 1 / (1 + math.exp(-4 * z))
    calP = max(0.0, min(1.0, calP))

    # --- Classic guard caps ---
    user_says_classic = tag_says_classic(inp.tag)
    machine_artifact  = ((out["burstiness"] <= 4.0 and out["ppl"] <= 12.0) or (frac10 >= 0.90))

    if (is_classic_auto and not machine_artifact):
        calP = min(calP, 0.25)   # auto classic cap
    if (user_says_classic and not machine_artifact):
        calP = min(calP, 0.10)   # user-tagged classic cap

    # --- Classic literature hard catch & override ---
    classic_keywords = re.findall(r"\b(thee|thou|thy|shall|whilst|hath|doth|ere|nay|aye|captain|monsieur|sir)\b",
                                  text, flags=re.I)
    proper_nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    dialogue = re.findall(r"[“\"].+?[”\"]", text)

    proper_noun_ratio = len(proper_nouns) / max(1, len(text.split()))
    dialogue_ratio = len(dialogue) / max(1, len(text.splitlines()))

    classic_signal = (len(classic_keywords) >= 3 or proper_noun_ratio > 0.02 or dialogue_ratio > 0.15)
    human_lit_metrics = (
        15 <= out['ppl'] <= 45 and
        4 <= out['burstiness'] <= 12 and
        frac10 < 0.90
    )

    if classic_signal and human_lit_metrics and not machine_artifact:
        calP = max(0, calP - 0.65)  # strong push toward human
        cat = "classic_literature"
        cat_conf = max(cat_conf, 0.8)
        note = "Classic-literature safeguard triggered — overridden to human"
        print("[guard] CLASSIC OVERRIDE APPLIED")
    else:
        note = category_note_for_report(is_classic_auto)

    # Build row AFTER all adjustments
    row = {
        "ts": int(time.time()),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
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
        "tag": (inp.tag or ""),
        "category": cat,
        "category_conf": round(cat_conf, 3),
        "category_note": note,
    }
    log_scan_row(row)

    percent = round(calP * 100)
    summary = ("almost certain." if percent >= 85 else
               "high likelihood." if percent >= 70 else
               "moderate likelihood." if percent >= 50 else
               "low likelihood.")

    cap_msgs = []
    if is_classic_auto and not machine_artifact:
        cap_msgs.append("classic-guard (auto)")
    if user_says_classic and not machine_artifact:
        cap_msgs.append("classic-guard (tag)")
    cap_note = f" Caps applied: {', '.join(cap_msgs)}." if cap_msgs else ""

    exp = (
        f"{round(100*frac10)}% of tokens in Top-10; "
        f"{round(100*frac100)}% in Top-100. "
        f"Perplexity≈{out['ppl']:.1f}; Burstiness≈{out['burstiness']:.3f}. "
        "Higher Top-10 fractions and lower perplexity typically correlate with model-like text. "
        f"Likelihood this was AI-generated: {percent}% — {summary} "
        f"Detected: {cat.replace('_',' ')} (conf≈{cat_conf:.0%}). "
        f"{note}{cap_note}"
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
        "calibrated_prob": calP,
        "category": cat,
        "category_conf": cat_conf,
        "category_note": note,
    }

# -------------------- Serve static UI --------------------
@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = pathlib.Path(__file__).parent / "index.html"
    return index_path.read_text(encoding="utf-8")

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)
