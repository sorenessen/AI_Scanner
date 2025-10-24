# app.py — AI Text Scanner with Category Detection (incl. classic_literature),
# category-aware calibration, readable CSV logging, and per-scan PDF reports.

import os, csv, time, hashlib, pathlib, math, re
from typing import List, Dict, Optional
from pydantic import BaseModel

from fastapi import FastAPI
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
app = FastAPI(title="AI Text Scanner (Category-Aware + Classic Literature Fix)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Model --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Use 'dtype' to avoid deprecation warning
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE)
model.eval().to(DEVICE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------- IO Models --------------------
class ScanIn(BaseModel):
    text: str
    tag: Optional[str] = None

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

# -------------------- Category detection --------------------
ARCHAIC_TOKENS = r"\b(thou|thee|thy|thine|ye|hath|doth|dost|art|shalt|whence|wherefore|ere|oft|nay|aye)\b"
CLASSIC_CUES = r"\b(whilst|whereupon|thereof|therein|therewith|herein|hereby|forthwith|betwixt|methinks|thereupon|wherein)\b"
ACADEMIC_CUES = [
    r"\babstract\b", r"\bintroduction\b", r"\bmethod(s|ology)?\b", r"\bresults?\b",
    r"\bdiscussion\b", r"\bconclusion(s)?\b", r"\breferences\b", r"\bdoi:\b",
    r"\b(et al\.|ibid\.|cf\.)\b", r"\(\d{4}\)", r"\[[0-9]{1,3}\]"
]
JOURNALISM_CUES = [
    r"\baccording to\b", r"\bspokes(person|man|woman)\b", r"\bofficials?\b",
    r"\binterview\b", r"\breported\b", r"\bpress (conference|release)\b",
    r"\b([A-Z][a-z]+ ){1,3}\([A-Z]{2,}\)\s?—"
]
TECH_CUES = [
    r"\bAPI\b", r"\bSDK\b", r"\bthroughput\b", r"\blatency\b", r"\bBig\-O\b",
    r"\bHTTP/\d\.\d\b", r"\bRFC\s?\d+\b", r"\balgorithm\b", r"def [a-zA-Z_]+\(", r"\bclass [A-Z]\w+\b"
]
POETRY_LINEBREAK_RATIO_CUTOFF = 0.20
SHORT_AVG_LINE_LEN = 60

def _ratio(n, d):
    return (n / d) if d else 0.0

def detect_category(text: str) -> dict:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    avg_line_len = sum(len(l) for l in lines)/len(lines) if lines else len(text)
    no_punct_ends = sum(1 for l in lines if not re.search(r"[\.!?;:—-]\s*$", l.strip()))
    enj_ratio = _ratio(no_punct_ends, max(1, len(lines)))
    arch_hits = len(re.findall(ARCHAIC_TOKENS, text, flags=re.I))
    classic_hits = len(re.findall(CLASSIC_CUES, text, flags=re.I))
    acad_hits = sum(bool(re.search(p, text, flags=re.I)) for p in ACADEMIC_CUES)
    jour_hits = sum(bool(re.search(p, text, flags=re.I)) for p in JOURNALISM_CUES)
    tech_hits = sum(bool(re.search(p, text)) for p in TECH_CUES)

    if arch_hits >= 3 and enj_ratio >= 0.15:
        return {"category": "archaic_poetry", "confidence": 0.85, "signals": {"arch": arch_hits, "enj": enj_ratio}}
    if acad_hits >= 3 and len(lines) >= 6:
        return {"category": "academic", "confidence": 0.8, "signals": {"acad": acad_hits}}
    if jour_hits >= 2:
        return {"category": "journalism", "confidence": 0.7, "signals": {"jour": jour_hits}}
    if tech_hits >= 2:
        return {"category": "technical", "confidence": 0.75, "signals": {"tech": tech_hits}}
    if enj_ratio >= POETRY_LINEBREAK_RATIO_CUTOFF and avg_line_len <= SHORT_AVG_LINE_LEN:
        return {"category": "modern_poetry", "confidence": 0.7, "signals": {"enj": enj_ratio}}
    # First-person narrative clues (light)
    if re.search(r"\b(I|we|my|our)\b.*\b(said|thought|felt|remember|walked|looked)\b", text, flags=re.I):
        # If classic cues are present, tip toward classic literature here (pre-metric)
        if classic_hits >= 2:
            return {"category": "classic_literature", "confidence": 0.7, "signals": {"classic": classic_hits}}
        return {"category": "creative_narrative", "confidence": 0.6, "signals": {}}
    if re.search(r"\bI think\b|\bmy view\b|\bin my opinion\b", text, flags=re.I):
        return {"category": "blog_opinion", "confidence": 0.6, "signals": {}}
    # Classic cues without FP to poetry
    if classic_hits >= 3:
        return {"category": "classic_literature", "confidence": 0.65, "signals": {"classic": classic_hits}}
    return {"category": "other", "confidence": 0.5, "signals": {}}

def category_adjustments(cat: str) -> dict:
    # defaults
    w = {"score": 2.3, "ppl": 0.9, "burst": 0.6, "t10": 0.5, "bias": -1.5}
    note = "Default calibration."
    if cat == "journalism":
        w = {"score": 2.0, "ppl": 0.9, "burst": 0.7, "t10": 0.4, "bias": -1.3}
        note = "Journalism: edited prose; stricter threshold."
    elif cat == "academic":
        w = {"score": 1.9, "ppl": 1.0, "burst": 0.9, "t10": 0.35, "bias": -1.25}
        note = "Academic: citations inflate predictability; weight burstiness more."
    elif cat == "technical":
        w = {"score": 1.8, "ppl": 0.9, "burst": 0.7, "t10": 0.35, "bias": -1.2}
        note = "Technical: repetitive terminology; avoid false positives."
    elif cat == "modern_poetry":
        w = {"score": 1.2, "ppl": 0.7, "burst": 1.4, "t10": 0.25, "bias": -1.8}
        note = "Modern poetry: emphasize burstiness; down-weight predictability."
    elif cat == "archaic_poetry":
        w = {"score": 1.0, "ppl": 0.7, "burst": 1.6, "t10": 0.2, "bias": -2.0}
        note = "Archaic poetry: lexical correction to avoid false positives."
    elif cat == "creative_narrative":
        w = {"score": 1.6, "ppl": 0.9, "burst": 1.1, "t10": 0.35, "bias": -1.55}
        note = "Creative narrative: style variation expected."
    elif cat == "blog_opinion":
        w = {"score": 1.8, "ppl": 0.9, "burst": 0.9, "t10": 0.35, "bias": -1.45}
        note = "Opinion/blog: conversational; moderate thresholds."
    elif cat == "classic_literature":
        # NEW: dampen AI-likelihood for Twain-like prose (low ppl + low burstiness)
        w = {"score": 1.2, "ppl": 0.7, "burst": 1.4, "t10": 0.25, "bias": -2.05}
        note = "Classic literature: dampened predictability; emphasize stylistic variation to avoid false positives."
    return {"weights": w, "note": note}

# -------------------- Logging + PDF --------------------
LOG_PATH = os.path.join(".", "scan_logs.csv")

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

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

    # Likelihood indicator bar
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

    # Metrics bar chart
    story.append(Paragraph("<b>Metrics Overview</b>", styles["Heading3"]))
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

    expected = {
        "journalism": "Expected: Top-10 60–80%, PPL 10–25, Burstiness 4–7. High predictability is normal.",
        "academic": "Expected: Top-10 55–75%, PPL 12–30, Burstiness 5–8. Citations raise predictability.",
        "technical": "Expected: Top-10 55–75%, PPL 12–28, Burstiness 5–8. Repetition is normal.",
        "modern_poetry": "Expected: Top-10 30–55%, PPL 20–60, Burstiness 7–12 (high).",
        "archaic_poetry": "Expected: Top-10 35–60%, PPL 18–55, Burstiness 8–13. Archaic lexicon lowers AI-likelihood.",
        "creative_narrative": "Expected: Top-10 40–65%, PPL 16–45, Burstiness 6–11.",
        "classic_literature": "Expected: Top-10 55–75%, PPL 12–30, Burstiness 5–9. Older diction lowers burstiness; avoid false positives.",
        "blog_opinion": "Expected: Top-10 45–65%, PPL 15–35, Burstiness 6–10.",
        "other": "Expected: Mixed; interpret with caution.",
    }
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(f"<i>{expected.get(row.get('category','other'), expected['other'])}</i>", styles["Normal"]))

    doc.build(story)
    return pdf_path

def log_scan_row(row: dict):
    """Readable CSV (single schema) + per-scan PDF."""
    pathlib.Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    csv_exists = os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if not csv_exists:
            writer.writerow([
                "# Legend:", "Each row = one scan result.",
                "Likelihood % = probability text was AI-generated.",
                "Verdict interprets likelihood; Category is auto-detected."
            ])
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

# -------------------- Route --------------------
@app.post("/scan")
def scan(inp: ScanIn):
    text = inp.text.strip()
    if not text:
        return {
            "overall_score": 0.0,
            "per_token": [],
            "explanation": "Empty text",
        }

    out = chunked_scan(text)

    total = max(1, out["total"])
    bins = out["bins"]
    frac10 = bins[10] / total
    frac100 = bins[100] / total

    # features for calibration
    fPpl   = max(0, min(1, (25 - (out['ppl'] or 25)) / 20))
    fBurst = max(0, min(1, (8 - (out['burstiness'] or 8)) / 6))

    # --- Category detection ---
    cat_info = detect_category(text)
    cat = cat_info["category"]
    cat_conf = cat_info["confidence"]

    # --- Classic-literature override using metrics (NEW) ---
    classic_metric_trigger = (
        out["ppl"] <= 15.0
        and out["burstiness"] <= 7.0
        and frac10 >= 0.65
    )
    if cat in ("creative_narrative", "other") and classic_metric_trigger:
        cat = "classic_literature"
        classic_hits = len(
            re.findall(
                r"\b(whilst|whereupon|thereof|therein|therewith|herein|hereby|forthwith|betwixt|methinks|thereupon|wherein)\b",
                text,
                flags=re.I,
            )
        )
        cat_conf = (
            0.75 if classic_hits >= 1
            else max(cat_conf, 0.65)
        )

    # Get category-specific weights
    adj = category_adjustments(cat)
    w = adj["weights"]

    # Calibrated probability (category-aware)
    z = (
        (w["score"] * out["score"])
        + (w["ppl"] * fPpl)
        + (w["burst"] * fBurst)
        + (w["t10"] * frac10)
        + w["bias"]
    )
    calP = 1 / (1 + math.exp(-4 * z))
    calP = max(0, min(1, calP))

    # Log to CSV and generate PDF
    row = {
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
        "tag": (inp.tag or ""),
        "category": cat,
        "category_conf": round(cat_conf, 3),
        "category_note": adj["note"],
    }
    log_scan_row(row)

    percent = round(calP * 100)
    summary = (
        "almost certain."
        if percent >= 85 else
        "high likelihood."
        if percent >= 70 else
        "moderate likelihood."
        if percent >= 50 else
        "low likelihood."
    )

    exp = (
        f"{round(100*frac10)}% of tokens in Top-10; "
        f"{round(100*frac100)}% in Top-100. "
        f"Perplexity≈{out['ppl']:.1f}; Burstiness≈{out['burstiness']:.3f}. "
        "Higher Top-10 fractions and lower perplexity typically correlate with model-like text. "
        f"Likelihood this was AI-generated: {percent}% — {summary} "
        f"Detected style: {cat.replace('_',' ')} (conf≈{cat_conf:.0%}). "
        f"{adj['note']}"
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
        "category_note": adj["note"],
    }

# -----------------------------------------------------------------
# NEW: Serve index.html at "/" so the browser UI loads
# -----------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_index():
    """
    Serve the local index.html so visiting http://127.0.0.1:8080/
    loads the scanner UI.
    """
    index_path = pathlib.Path(__file__).parent / "index.html"
    return index_path.read_text(encoding="utf-8")

# Optional: silence favicon.ico 404 spam in logs
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)
