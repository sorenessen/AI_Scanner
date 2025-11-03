# app.py — AI Text Scanner with classic-literature + nonsense-verse guards (single file)

import os, csv, time, hashlib, pathlib, math, re, string
from typing import List, Dict, Optional
from statistics import mean, pstdev

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

# ---------- Stylometry Layer v2 ----------
FUNCTION_WORDS = set("""
the be to of and a in that have I it for not on with he as you do at this 
but his by from they we say her she or an will my one all would there their
is was are were been being am you your me mine our ours who whom which what
if then else when where why how so than into up down over under out in on
more most some any no yes can could should would might may must just very
much many few every each other another again before after during without
within between about above below through across against because therefore
said says say asked replied
""".split())

def stylometric_fingerprint(text: str, token_logps: List[float]) -> Dict[str, float]:
    words = [w.strip(string.punctuation).lower() for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return {"func_ratio":0.0,"hapax_ratio":0.0,"sent_mean":0.0,"sent_var":0.0,
                "mlps":0.0,"mlps_var":0.0,"punct_entropy":0.0}

    func_ratio = sum(1 for w in words if w in FUNCTION_WORDS) / len(words)

    freq = {}
    for w in words: freq[w] = freq.get(w, 0) + 1
    hapax_ratio = sum(1 for _, c in freq.items() if c == 1) / max(1, len(freq))

    sents = re.split(r'[.!?]+', text)
    lengths = [len(s.split()) for s in sents if s.strip()]
    sent_mean = mean(lengths) if lengths else 0.0
    sent_var  = pstdev(lengths) if len(lengths) > 1 else 0.0

    if len(token_logps) > 5:
        slopes = [token_logps[i+1]-token_logps[i] for i in range(len(token_logps)-1)]
        mlps = mean(slopes)
        mlps_var = pstdev(slopes) if len(slopes) > 1 else 0.0
    else:
        mlps, mlps_var = 0.0, 0.0

    punct = [c for c in text if c in ".,;:!?-—"]
    if punct:
        uniq = list(set(punct))
        freqs = [punct.count(p)/len(punct) for p in uniq]
        punct_entropy = -sum(p*math.log(max(p,1e-12)) for p in freqs)
    else:
        punct_entropy = 0.0

    return {"func_ratio":float(func_ratio),"hapax_ratio":float(hapax_ratio),
            "sent_mean":float(sent_mean),"sent_var":float(sent_var),
            "mlps":float(mlps),"mlps_var":float(mlps_var),
            "punct_entropy":float(punct_entropy)}

# -------------------- Config --------------------
VERSION = "0.1.4"  # bumped to reflect confidence hardening
MODEL_NAME = os.getenv("REF_MODEL", "EleutherAI/gpt-neo-1.3B")
MAX_TOKENS_PER_PASS = 768
STRIDE = 128
USE_FP16 = True

# --- Confidence hardening knobs ---
MIN_TOKENS_STRONG = int(os.getenv("MIN_TOKENS_STRONG", "180"))  # tokens for full-strength confidence
SHORT_CAP = os.getenv("SHORT_CAP", "0")  # set to "1" to cap short excerpts

def _binom_ci_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.5
    var = p * (1 - p) / n
    return z * math.sqrt(max(var, 0.0))

def _shrink_toward_half(prob: float, reliability: float) -> float:
    # reliability in [0,1]; 0 -> 0.5; 1 -> unchanged
    return 0.5 + (prob - 0.5) * max(0.0, min(1.0, reliability))

# -------------------- Device setup --------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
DTYPE = torch.float16 if (USE_FP16 and (DEVICE.type in {"cuda","mps"})) else torch.float32

# -------------------- App + CORS --------------------
app = FastAPI(title="AI Text Scanner (Classic + Nonsense Guards)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
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
    if DTYPE == torch.float16 and hasattr(model,"half"): model = model.half()
    elif hasattr(model,"float"): model = model.float()
    model.eval()
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print("[server] model loaded OK on", DEVICE, "dtype", DTYPE)
except Exception as e:
    model_load_error = str(e)
    print("[server] FAILED TO LOAD MODEL:", model_load_error)

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
    topk_bins = {10:0, 100:0, 1000:0, "rest":0}
    total = max(0, ids.size(0)-1)
    logprobs = []

    for i in range(1, ids.size(0)):
        next_logits = logits[0, i-1]
        probs = torch.softmax(next_logits, dim=-1)
        actual_id = ids[i].item()
        p_actual = probs[actual_id].item()
        rank = int((probs > p_actual).sum().item() + 1)

        if rank <= 10: topk_bins[10] += 1
        elif rank <= 100: topk_bins[100] += 1
        elif rank <= 1000: topk_bins[1000] += 1
        else: topk_bins["rest"] += 1

        tok_str = tokenizer.convert_ids_to_tokens([actual_id])[0]
        per_token.append({"t": tok_str, "rank": rank, "p": p_actual})
        logprobs.append(math.log(max(p_actual, 1e-12)))

    ppl = math.exp(-sum(logprobs) / max(1, len(logprobs)))
    mean_lp = sum(logprobs) / max(1, len(logprobs))
    burstiness = sum((lp - mean_lp) ** 2 for lp in logprobs) / max(1, len(logprobs))

    if total > 0:
        frac10 = topk_bins[10]/total
        frac100 = topk_bins[100]/total
        overall = min(1.0, 0.75*frac10 + 0.35*frac100)
    else:
        overall = 0.0

    return {"per_token": per_token, "bins": topk_bins, "total": total,
            "score": overall, "ppl": ppl, "burstiness": burstiness}

def chunked_scan(text: str) -> Dict:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]

    def _scan_ids(slice_ids: torch.Tensor) -> Dict:
        return scan_chunk(slice_ids.to(DEVICE))

    if ids.size(0) <= MAX_TOKENS_PER_PASS:
        return _scan_ids(ids.unsqueeze(0))

    all_tokens: List[Dict] = []
    agg_bins = {10:0, 100:0, 1000:0, "rest":0}
    agg_total = 0
    ppls, bursts, scores = [], [], []

    start = 0
    while start < ids.size(0):
        end = min(start + MAX_TOKENS_PER_PASS, ids.size(0))
        result = _scan_ids(ids[start:end].unsqueeze(0))

        all_tokens.extend(result["per_token"])
        for k in agg_bins: agg_bins[k] += result["bins"][k]
        agg_total += result["total"]
        scores.append(result["score"]); ppls.append(result["ppl"]); bursts.append(result["burstiness"])
        if end == ids.size(0): break
        start = end - STRIDE

    if agg_total > 0:
        frac10 = agg_bins[10]/agg_total
        frac100 = agg_bins[100]/agg_total
        overall = min(1.0, 0.75*frac10 + 0.35*frac100)
    else:
        overall = 0.0

    return {"per_token": all_tokens, "bins": agg_bins, "total": agg_total,
            "score": overall, "ppl": (sum(ppls)/len(ppls) if ppls else 0.0),
            "burstiness": (sum(bursts)/len(bursts) if bursts else 0.0)}

# -------------------- Category / Guards --------------------
ARCHAIC_TOKENS = r"\b(thou|thee|thy|thine|ye|hath|doth|dost|art|shalt|whence|wherefore|ere|oft|nay|aye)\b"
CLASSIC_CUES   = r"\b(whilst|whereupon|thereof|therein|therewith|herein|hereby|forthwith|betwixt|methinks|thereupon|wherein|monsieur|captain|sir)\b"
NAUTICAL_CUES  = r"\b(anchor|yard[s]?|mast[s]?|port[- ]?hole|pilot|flag[s]?|chain|rattl(?:e|ing)|deck|crew|yards ashore|half[- ]?mast)\b"

def prose_classic_signals(text: str) -> dict:
    quotes = re.findall(r"[“\"]([^”\"]+)[”\"]", text)
    lines  = [ln.strip() for ln in text.splitlines() if ln.strip()]
    dialog_density = (len(quotes) / max(1, len(lines)))
    words = re.findall(r"\b[A-Za-z][a-z]+\b", text)
    proper_nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    proper_ratio = len(proper_nouns) / max(1, len(words))
    punct_hits = len(re.findall(r"[;—–]", text)) / max(1, len(lines))
    nautical_hits = len(re.findall(NAUTICAL_CUES, text, flags=re.I))
    return {
        "dialog_density": dialog_density,
        "proper_ratio":   proper_ratio,
        "punct_rate":     punct_hits,
        "nautical_hits":  nautical_hits
    }

def looks_classic_like(text: str, metrics: Dict, frac10: float) -> bool:
    """Classic prose heuristic: one+ cue plus human-edited prose signals."""
    arch_hits    = len(re.findall(ARCHAIC_TOKENS, text, flags=re.I))
    classic_hits = len(re.findall(CLASSIC_CUES,   text, flags=re.I))
    cues = arch_hits + classic_hits
    sigs  = prose_classic_signals(text)
    ppl   = metrics.get("ppl", 99.0)
    burst = metrics.get("burstiness", 99.0)
    prose_ok = (
        sigs["dialog_density"] >= 0.10 or
        sigs["proper_ratio"]   >= 0.020 or
        sigs["nautical_hits"]  >= 2
    )
    return (
        cues >= 1 and
        prose_ok and
        ppl   <= 24.0 and
        burst <= 12.0 and
        frac10 >= 0.55
    )

def tag_says_classic(tag: Optional[str]) -> bool:
    if not tag: return False
    t = tag.lower()
    return any(k in t for k in ("classic","pre-1920","public_domain","public-domain","pre1920"))

def category_note_for_report(is_classic: bool) -> str:
    return ("Classic literature: dampened predictability; avoiding false positives."
            if is_classic else "Default calibration.")

# --- Nonsense / Surreal Verse helpers ---
CARROLL_LEXICON = re.compile(
    r"\b(jabberwock|bandersnatch|jubjub|borogove|outgrabe|brillig|slithy|toves|"
    r"gyre|gimble|mome|raths|vorpal|frabjous|manxome|galumph|tulgey|uffish)\b", re.I)

def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w: return 0
    groups = re.findall(r"[aeiouy]+", w)
    syl = max(1, len(groups))
    if w.endswith("e") and syl > 1 and not w.endswith(("le","ye")): syl -= 1
    return max(1, syl)

def _rhyme_key(word: str, min_tail=2, max_tail=4) -> str:
    w = re.sub(r"[^a-z]", "", word.lower())
    if len(w) <= min_tail: return w
    m = re.search(r"[aeiouy][a-z]*$", w)
    return (m.group(0) if m else w)[-max_tail:]

def rhyme_density(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3: return 0.0
    endings = []
    for ln in lines:
        toks = [t.strip(string.punctuation) for t in ln.split()]
        endings.append(_rhyme_key(toks[-1]) if toks else "")
    counts = {}
    for r in endings: counts[r] = counts.get(r, 0) + 1
    pairs = sum(c*(c-1)//2 for c in counts.values())
    possible = len(endings)*(len(endings)-1)//2
    return pairs / max(1, possible)

def meter_periodicity(text: str) -> float:
    lines = [ln for ln in (l.strip() for l in text.splitlines()) if ln]
    if len(lines) < 3: return 1.0
    syls = []
    for ln in lines:
        words = [w.strip(string.punctuation) for w in ln.split()]
        syls.append(sum(_syllables(w) for w in words))
    m = mean(syls) if syls else 0.0
    s = pstdev(syls) if len(syls) > 1 else 0.0
    return (s/m) if m > 0 else 1.0

def invented_word_ratio(text: str) -> float:
    words = [w.strip(string.punctuation) for w in re.findall(r"[A-Za-z'-]+", text)]
    if not words: return 0.0
    COMMON = set(w.lower() for w in FUNCTION_WORDS)
    novel = 0
    for w in words:
        lw = w.lower()
        if len(lw) <= 3 or w[0].isupper() or lw in COMMON: continue
        if re.search(r"[bcdfghjklmnpqrstvwxz]{3,}", lw) or re.search(r"(zz|zx|xq|qk|kk)", lw):
            novel += 1; continue
        if re.search(r"(wock|snatch|mome|borog|jubjub|frum|vorpal|tulgey|uffish|bander)", lw):
            novel += 1; continue
        if not re.search(r"[aeiouy]", lw):
            novel += 1
    return novel / max(1, len(words))

def semantic_discontinuity(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines: return 0.0
    no_punct = sum(1 for ln in lines if not re.search(r"[.!?;:]", ln))
    shortish = sum(1 for ln in lines if len(ln.split()) <= 8)
    return 0.5*(no_punct/max(1,len(lines))) + 0.5*(shortish/max(1,len(lines)))

def looks_nonsense_verse(text: str, metrics: Dict, frac10: float, frac100: float) -> Dict:
    # Carroll lexicon shortcut (deterministic)
    lex_hits = len(CARROLL_LEXICON.findall(text))
    if lex_hits >= 1:
        r = {
            "rhyme_density": round(rhyme_density(text), 3),
            "meter_cv": round(meter_periodicity(text), 3),
            "invented_ratio": round(invented_word_ratio(text), 3),
            "lex_hits": int(lex_hits),
            "semantic_disc": round(semantic_discontinuity(text), 3),
            "ppl": round(metrics.get("ppl", 99.0), 3),
            "burst": round(metrics.get("burstiness", 99.0), 3),
            "top10": round(frac10, 3),
            "top100": round(frac100, 3),
        }
        return {"hit": True, "signals": r}

    # Otherwise use general nonsense-verse pattern
    r_density = rhyme_density(text)
    cv_meter  = meter_periodicity(text)
    inv_ratio = invented_word_ratio(text)
    sem_disc  = semantic_discontinuity(text)
    ppl   = metrics.get("ppl", 99.0)
    burst = metrics.get("burstiness", 99.0)

    generic_rule = (
        r_density >= 0.22 and
        cv_meter  <= 0.60 and
        (inv_ratio >= 0.06) and
        sem_disc  >= 0.25 and
        frac10    >= 0.75 and
        ppl       <= 14.0 and
        burst     <= 8.0
    )

    hit = bool(generic_rule)
    return {"hit": hit, "signals":{
        "rhyme_density": round(r_density,3),
        "meter_cv": round(cv_meter,3),
        "invented_ratio": round(inv_ratio,3),
        "lex_hits": 0,
        "semantic_disc": round(sem_disc,3),
        "ppl": round(ppl,3),
        "burst": round(burst,3),
        "top10": round(frac10,3),
        "top100": round(frac100,3),
    }}

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
        ["Metric","Value"],
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
    table = Table(data, colWidths=[2.5*inch, 3.5*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.grey),
        ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("BOTTOMPADDING",(0,0),(-1,0),8),
        ("BACKGROUND",(0,1),(-1,-1),colors.beige),
        ("GRID",(0,0),(-1,-1),0.25,colors.black),
    ]))
    story.append(table); story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Likelihood Indicator</b>", styles["Heading3"]))
    color = (colors.red if likelihood_pct >= 85 else
             colors.orange if likelihood_pct >= 60 else
             colors.yellow if likelihood_pct >= 40 else
             colors.green)
    d = Drawing(420, 40)
    d.add(Rect(0,10, 4*likelihood_pct, 20, fillColor=color))
    d.add(Rect(0,10, 400, 20, fillColor=None, strokeColor=colors.black))
    d.add(String(410,15, f"{likelihood_pct}%", fontSize=12))
    story.append(d); story.append(Spacer(1, 0.25*inch))
    chart = VerticalBarChart()
    chart.x, chart.y = 50, 30
    chart.height, chart.width = 150, 400
    chart.data = [[row["top10_frac"]*100, row["top100_frac"]*100, row["ppl"], row["burstiness"]]]
    chart.categoryAxis.categoryNames = ["Top-10 %","Top-100 %","Perplexity","Burstiness"]
    chart.bars[0].fillColor = colors.darkblue
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(100, row["ppl"] + 10)
    chart.valueAxis.valueStep = 20
    d2 = Drawing(500, 200); d2.add(chart); story.append(d2)
    doc.build(story); return pdf_path

def log_scan_row(row: dict):
    pathlib.Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    csv_exists = os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(["# Legend:","Each row = one scan result.",
                             "Likelihood % = probability text was AI-generated.",
                             "Verdict interprets likelihood; Category is auto-detected (with classic/nonsense guards)."])
            writer.writerow([])
            writer.writerow(["Timestamp","Verdict","Likelihood %","Predictability Score",
                             "Top-10 %","Top-100 %","Perplexity","Burstiness",
                             "Model","Device","Chars","Tokens","Text Hash","Tag",
                             "Category","Category Conf","Category Note"])
        verdict = ("Likely AI-generated" if row["ai_likelihood_calibrated"] >= 0.6 else "Likely human-written")
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["ts"])),
            verdict, f"{int(row['ai_likelihood_calibrated']*100)}%",
            f"{row['overall_score']:.2f}",
            f"{row['top10_frac']*100:.1f}%", f"{row['top100_frac']*100:.1f}%",
            f"{row['ppl']:.1f}", f"{row['burstiness']:.2f}",
            row["model_name"], row["device"],
            row["text_len_chars"], row["text_len_tokens"],
            row["text_sha256"][:8], row["tag"] or "",
            row.get("category","other"), f"{int(100*row.get('category_conf',0))}%",
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
    frac10 = bins[10]/total
    frac100 = bins[100]/total

    # helpers
    fPpl   = max(0, min(1, (25 - (out['ppl'] or 25)) / 20))
    fBurst = max(0, min(1, (8 - (out['burstiness'] or 8)) / 6))

    # --- Evidence strength / reliability ---
    ci10 = _binom_ci_halfwidth(frac10, total)
    len_factor = min(1.0, total / float(MIN_TOKENS_STRONG))   # 0..1
    shape_factor = 1.0 - min(0.6, ci10 * 1.8)                 # damp when CI wide
    reliability = max(0.15, len_factor * shape_factor)        # floor

    # classic detector
    is_classic_auto = looks_classic_like(text, out, frac10)
    cat = "classic_literature" if is_classic_auto else "other"
    cat_conf = 0.75 if is_classic_auto else 0.5

    # base combiner
    z = (2.0*out["score"]) + (0.9*fPpl) + (0.8*fBurst) + (0.35*frac10) - 1.6
    calP = 1/(1+math.exp(-4*z))
    calP = max(0.0, min(1.0, calP))

    # MUCH stricter artifact block (only for obvious machine text)
    machine_artifact = (
        frac10 >= 0.985 and
        frac100 <= 0.03 and
        out["ppl"] <= 2.5 and
        out["burstiness"] <= 3.5
    )

    # classic caps
    user_says_classic = tag_says_classic(inp.tag)
    if (is_classic_auto and not machine_artifact): calP = min(calP, 0.25)
    if (user_says_classic and not machine_artifact): calP = min(calP, 0.10)

    # default note (can be overwritten by guards below)
    note = category_note_for_report(is_classic_auto)

    # stronger classic-prose override for dialog-heavy 19th-century chapters
    if is_classic_auto and not machine_artifact:
        sigs = prose_classic_signals(text)
        strong_prose = (
            sigs["dialog_density"] >= 0.14 or
            (sigs["proper_ratio"] >= 0.025 and sigs["punct_rate"] >= 0.10) or
            sigs["nautical_hits"] >= 3
        )
        if strong_prose:
            calP = min(calP, 0.12)
            cat = "classic_literature"
            cat_conf = max(cat_conf, 0.85)
            note = ("Classic-prose safeguard — dialog_density="
                    f"{sigs['dialog_density']:.3f}, proper_ratio={sigs['proper_ratio']:.3f}, "
                    f"punct/line={sigs['punct_rate']:.3f}, nautical_hits={sigs['nautical_hits']}")

    # nonsense verse guard (ALWAYS allowed to run; ignores machine_artifact)
    nonsense = looks_nonsense_verse(text, out, frac10, frac100)
    if nonsense["hit"]:
        calP = min(calP, 0.02)     # force human for rhymed nonsense
        cat = "nonsense_literature"
        cat_conf = max(cat_conf, 0.90)
        note = ("Nonsense/Surreal verse safeguard — "
                f"rhyme={nonsense['signals']['rhyme_density']}, "
                f"meter_cv={nonsense['signals']['meter_cv']}, "
                f"invented={nonsense['signals']['invented_ratio']}, "
                f"lex_hits={nonsense['signals']['lex_hits']}, "
                f"sem_disc={nonsense['signals']['semantic_disc']}")
        print("[guard] NONSENSE-VERSE OVERRIDE APPLIED", nonsense["signals"])

    # Final reliability shrink (prevents overconfident extremes on thin evidence)
    calP = _shrink_toward_half(calP, reliability)

    # Optional short-excerpt cap
    if SHORT_CAP == "1" and total < 160 and not machine_artifact:
        calP = min(calP, 0.35)

    # stylometry diagnostics
    token_logps = [math.log(max(t.get("p", 0.0), 1e-12)) for t in out["per_token"]]
    stylo = stylometric_fingerprint(text, token_logps)

    # row
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

    percent = round(calP*100)
    summary = ("almost certain." if percent >= 85 else
               "high likelihood." if percent >= 70 else
               "moderate likelihood." if percent >= 50 else
               "low likelihood.")
    cap_msgs = []
    if is_classic_auto and not machine_artifact: cap_msgs.append("classic-guard (auto)")
    if user_says_classic and not machine_artifact: cap_msgs.append("classic-guard (tag)")
    if nonsense["hit"]: cap_msgs.append("nonsense-verse guard")
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

    return {"overall_score": out["score"], "per_token": out["per_token"],
            "explanation": exp, "ppl": out["ppl"], "burstiness": out["burstiness"],
            "bins": bins, "total": total, "model_name": MODEL_NAME,
            "device": str(DEVICE), "dtype": str(DTYPE),
            "calibrated_prob": calP, "category": cat,
            "category_conf": cat_conf, "category_note": note,
            "stylometry": stylo, "nonsense_signals": nonsense["signals"]}

# -------------------- Serve static UI --------------------
@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = pathlib.Path(__file__).parent / "index.html"
    return index_path.read_text(encoding="utf-8")

@app.get("/version")
def version():
    return {"version": VERSION, "model": MODEL_NAME, "device": str(DEVICE), "dtype": str(DTYPE)}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)
