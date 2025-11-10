# app.py — AI Text Scanner (pattern-classic + nonsense + ensemble + abstain + PD-dampener + runtime config)

import os, csv, time, hashlib, pathlib, math, re, string, json, glob, random
from typing import List, Dict, Optional, Tuple
from statistics import mean, pstdev

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi import UploadFile, File, Query

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

# --- Live verification scratch state (in-memory) ---
LAST_SCAN_STYLO = None   # set on each /scan
LAST_SCAN_META  = None   # store token counts etc.


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
    punct = [c for c in text if c in ".,;:!?-—–"]
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

# -------------------- Config (env defaults) --------------------
VERSION = "0.3.1"  # adds /config runtime settings

MODEL_NAME = os.getenv("REF_MODEL", "EleutherAI/gpt-neo-1.3B")
SECOND_MODEL_ENV = os.getenv("SECOND_MODEL", "distilgpt2")  # optional
ENABLE_SECOND_MODEL_ENV = os.getenv("ENABLE_SECOND_MODEL", "0") == "1"

MAX_TOKENS_PER_PASS = int(os.getenv("MAX_TOKENS_PER_PASS", "768"))
STRIDE = int(os.getenv("STRIDE", "128"))
USE_FP16 = os.getenv("USE_FP16", "1") == "1"

# Confidence + abstain (defaults)
ABSTAIN_LOW_DEF  = float(os.getenv("ABSTAIN_LOW",  "0.35"))
ABSTAIN_HIGH_DEF = float(os.getenv("ABSTAIN_HIGH", "0.65"))

# Hardening defaults
MIN_TOKENS_STRONG_DEF = int(os.getenv("MIN_TOKENS_STRONG", "180"))
SHORT_CAP_DEF = (os.getenv("SHORT_CAP", "0") == "1")
MAX_CONF_SHORT_DEF    = float(os.getenv("MAX_CONF_SHORT", "0.35"))

BOOTSTRAP_SAMPLES = int(os.getenv("BOOTSTRAP_SAMPLES", "24"))
BOOTSTRAP_WINDOW  = int(os.getenv("BOOTSTRAP_WINDOW",  "256"))
MAX_CONF_UNSTABLE_DEF = float(os.getenv("MAX_CONF_UNSTABLE", "0.35"))
NON_EN_CAP_DEF        = float(os.getenv("NON_EN_CAP",        "0.15"))
EN_THRESH_DEF         = float(os.getenv("EN_THRESH",         "0.70"))

# Modes
DEFAULT_MODE = os.getenv("MODE", "Balanced").strip().title()
if DEFAULT_MODE not in ("Balanced","Strict","Academic"):
    DEFAULT_MODE = "Balanced"

# Public-domain statistical dampener
PD_FINGERPRINT_DIR = os.getenv("PD_FINGERPRINT_DIR", "./pd_fingerprints")
PD_NGRAM_N = int(os.getenv("PD_NGRAM_N", "5"))
PD_DAMP_THRESHOLD = float(os.getenv("PD_DAMP_THRESHOLD", "0.12"))
PD_MAX_CONF = float(os.getenv("PD_MAX_CONF", "0.25"))

# -------------------- Runtime settings (UI-tweakable) --------------------
class ScanSettings(BaseModel):
    mode: str = DEFAULT_MODE                         # Balanced | Strict | Academic
    short_cap: bool = SHORT_CAP_DEF
    min_tokens_strong: int = MIN_TOKENS_STRONG_DEF
    use_ensemble: bool = ENABLE_SECOND_MODEL_ENV     # takes effect only if secondary is loaded
    non_en_cap: float = NON_EN_CAP_DEF
    en_thresh: float = EN_THRESH_DEF
    max_conf_unstable: float = MAX_CONF_UNSTABLE_DEF
    max_conf_short: float = MAX_CONF_SHORT_DEF
    abstain_low: float = ABSTAIN_LOW_DEF
    abstain_high: float = ABSTAIN_HIGH_DEF

SETTINGS = ScanSettings()

def _mode_multipliers(mode: str) -> Tuple[float,float,float]:
    # (sensitivity_boost, abstain_delta, artifact_bias)
    m = (mode or DEFAULT_MODE).strip().title()
    if m == "Strict":    return (1.10, -0.03, 1.05)
    if m == "Academic":  return (0.90,  0.05, 0.90)
    return (1.00, 0.00, 1.00)

def _binom_ci_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    if n <= 0: return 0.5
    var = p * (1 - p) / n
    return z * math.sqrt(max(var, 0.0))

def _shrink_toward_half(prob: float, reliability: float) -> float:
    return 0.5 + (prob - 0.5) * max(0.0, min(1.0, reliability))

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

# -------------------- Device setup --------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
DTYPE = torch.float16 if (USE_FP16 and (DEVICE.type in {"cuda","mps"})) else torch.float32

# -------------------- App + CORS --------------------
app = FastAPI(title="AI Text Scanner (Pattern Classic + Nonsense + Ensemble)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# -------------------- Model (safe load) --------------------
tokenizer = None
model = None
tokenizer2 = None
model2 = None
model_load_error = None
SECONDARY_READY = False
SECOND_MODEL = SECOND_MODEL_ENV

print("[server] loading model(s)...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    if (DTYPE == torch.float16) and hasattr(model, "half"):
        model = model.half()
    elif hasattr(model, "float"):
        model = model.float()
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[server] primary model OK on", DEVICE, "dtype", DTYPE)

    if ENABLE_SECOND_MODEL_ENV and SECOND_MODEL:
        try:
            tokenizer2 = AutoTokenizer.from_pretrained(SECOND_MODEL)
            model2 = AutoModelForCausalLM.from_pretrained(SECOND_MODEL)
            model2.to(DEVICE)
            if (DTYPE == torch.float16) and hasattr(model2, "half"):
                model2 = model2.half()
            elif hasattr(model2, "float"):
                model2 = model2.float()
            model2.eval()
            if tokenizer2.pad_token is None:
                tokenizer2.pad_token = tokenizer2.eos_token
            # GPT-Neo and (di)stilgpt2 share GPT-2 vocab size (usually 50257); verify anyway:
            SECONDARY_READY = (model.config.vocab_size == model2.config.vocab_size)
            if SECONDARY_READY:
                print(f"[server] second model OK: {SECOND_MODEL}")
            else:
                print(f"[server] second model vocab mismatch "
                      f"({model.config.vocab_size} vs {model2.config.vocab_size}) — disabling ensemble.")
                model2 = None
        except Exception as e2:
            print("[server] failed to load secondary model:", str(e2))
            model2 = None
            SECONDARY_READY = False

except Exception as e:
    model_load_error = str(e)
    print("[server] FAILED TO LOAD MODEL:", model_load_error)

# -------------------- IO Models --------------------
class ScanIn(BaseModel):
    text: str
    tag: Optional[str] = None
    mode: Optional[str] = None  # optionally override for a single call

# -------------------- Core scoring helpers (model-agnostic) --------------------
def scan_chunk_with(model_obj, tok, input_ids: torch.Tensor) -> Dict:
    model_obj.eval()
    with torch.no_grad():
        out = model_obj(input_ids=input_ids)
        logits = out.logits  # [1, T, V]
    ids = input_ids[0]
    per_token = []
    topk_bins = {10:0, 100:0, 1000:0, "rest":0}
    total = max(0, ids.size(0)-1)
    logprobs = []

    for i in range(1, ids.size(0)):
        next_logits = logits[0, i-1]
        probs = torch.softmax(next_logits, dim=-1)
        actual_id = int(ids[i].item())
        p_actual = float(probs[actual_id].item())
        rank = int((probs > p_actual).sum().item() + 1)

        if rank <= 10: topk_bins[10] += 1
        elif rank <= 100: topk_bins[100] += 1
        elif rank <= 1000: topk_bins[1000] += 1
        else: topk_bins["rest"] += 1

        tok_str = tok.convert_ids_to_tokens([actual_id])[0]
        per_token.append({"t": tok_str, "rank": rank, "p": p_actual})
        logprobs.append(math.log(max(p_actual, 1e-12)))

    ppl = math.exp(-sum(logprobs) / max(1, len(logprobs))) if logprobs else float("inf")
    mean_lp = (sum(logprobs)/len(logprobs)) if logprobs else 0.0
    burstiness = (sum((lp - mean_lp) ** 2 for lp in logprobs) / max(1, len(logprobs))) if logprobs else 0.0

    if total > 0:
        frac10 = topk_bins[10]/total
        frac100 = topk_bins[100]/total
        overall = min(1.0, 0.75*frac10 + 0.35*frac100)
    else:
        overall = 0.0

    return {"per_token": per_token, "bins": topk_bins, "total": total,
            "score": overall, "ppl": ppl, "burstiness": burstiness}

def _chunked_scan_with(model_obj, tok, text: str) -> Dict:
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0].to(DEVICE)

    def _scan_ids(slice_ids: torch.Tensor) -> Dict:
        return scan_chunk_with(model_obj, tok, slice_ids.unsqueeze(0))

    if ids.size(0) <= MAX_TOKENS_PER_PASS:
        return _scan_ids(ids)

    all_tokens: List[Dict] = []
    agg_bins = {10:0, 100:0, 1000:0, "rest":0}
    agg_total = 0
    ppls, bursts, scores = [], [], []

    start = 0
    T = ids.size(0)
    while start < T:
        end = min(start + MAX_TOKENS_PER_PASS, T)
        result = _scan_ids(ids[start:end])

        all_tokens.extend(result["per_token"])
        for k in agg_bins: agg_bins[k] += result["bins"][k]
        agg_total += result["total"]
        scores.append(result["score"]); ppls.append(result["ppl"]); bursts.append(result["burstiness"])
        if end == T: break
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

# Simple wrappers
def chunked_scan_primary(text: str) -> Dict:
    return _chunked_scan_with(model, tokenizer, text)

def chunked_scan_secondary(text: str) -> Optional[Dict]:
    if not (SECONDARY_READY and model2 and tokenizer2): return None
    # keep it a bit cheaper for the tiny/distil model
    global MAX_TOKENS_PER_PASS
    mtok = MAX_TOKENS_PER_PASS
    try:
        MAX_TOKENS_PER_PASS = min(mtok, 512)
        return _chunked_scan_with(model2, tokenizer2, text)
    finally:
        MAX_TOKENS_PER_PASS = mtok

# -------------------- Pattern-based Classic Style --------------------
def prose_classic_signals(text: str) -> dict:
    quotes = re.findall(r"[“\"]([^”\"]+)[”\"]", text)
    lines  = [ln.strip() for ln in text.splitlines() if ln.strip()]
    dialog_density = (len(quotes) / max(1, len(lines)))
    words = re.findall(r"\b[A-Za-z][a-z]+\b", text)
    proper_nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    proper_ratio = len(proper_nouns) / max(1, len(words))
    semi_dash = len(re.findall(r"[;:—–]", text))
    commas    = text.count(",")
    exclam_q  = len(re.findall(r"[!?]", text))
    lines_n   = max(1, len(lines))
    sent_like = max(1, len(re.split(r'[.!?]+', text)))
    clause_density = (commas + semi_dash) / float(sent_like)
    semi_dash_rate = semi_dash / float(lines_n)
    exclamq_rate   = exclam_q / float(lines_n)
    return {
        "dialog_density": dialog_density,
        "proper_ratio":   proper_ratio,
        "clause_density": clause_density,
        "semi_dash_rate": semi_dash_rate,
        "exclamq_rate":   exclamq_rate,
        "sent_like":      sent_like,
    }

def classic_style_score(text: str, stylo: Dict, sigs: Dict) -> float:
    sm   = stylo.get("sent_mean", 0.0)
    sv   = stylo.get("sent_var",  0.0)
    pe   = stylo.get("punct_entropy", 0.0)
    hr   = stylo.get("hapax_ratio",  0.0)
    fr   = stylo.get("func_ratio",   0.0)
    dd   = sigs["dialog_density"]
    cd   = sigs["clause_density"]
    sdr  = sigs["semi_dash_rate"]
    f_len   = _clamp((sm - 16.0) / 16.0, 0.0, 1.0)
    f_var   = _clamp((10.0 - abs(10.0 - sv)) / 10.0, 0.0, 1.0)
    f_clause= _clamp(cd / 1.2, 0.0, 1.0)
    f_semi  = _clamp(sdr / 0.06, 0.0, 1.0)
    f_dialog= _clamp(1.0 - abs(dd - 0.18)/0.18, 0.0, 1.0)
    f_vocab = _clamp((0.16 - abs(0.16 - hr))/0.16, 0.0, 1.0)
    f_pent  = _clamp(pe / 1.6, 0.0, 1.0)
    f_func  = _clamp(1.0 - abs(fr - 0.52)/0.15, 0.0, 1.0)
    score = (
        0.18*f_len + 0.10*f_var + 0.20*f_clause +
        0.12*f_semi + 0.18*f_dialog + 0.10*f_vocab +
        0.07*f_pent + 0.05*f_func
    )
    return _clamp(score, 0.0, 1.0)

def tag_says_classic(tag: Optional[str]) -> bool:
    if not tag: return False
    t = tag.lower()
    return any(k in t for k in ("classic","pre-1920","public_domain","public-domain","pre1920"))

def category_note_for_report(is_classic: bool, style_score: float = None) -> str:
    if is_classic:
        base = "Classic-style safeguard: structure/rhythm match; avoiding false positives."
        if style_score is not None:
            return f"{base} style_score≈{style_score:.2f}"
        return base
    return "Default calibration."

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
    m = re.search(r"[aeiouy][a-z]*$", w)
    return (m.group(0) if m else w)[-max_tail:] if len(w) > min_tail else w

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

# -------------------- Public-Domain Overlap Dampener --------------------
def _load_pd_fingerprints() -> List[Dict]:
    fps = []
    try:
        for path in glob.glob(os.path.join(PD_FINGERPRINT_DIR, "*.json")):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, dict) and "ngrams" in obj and "N" in obj:
                    fps.append(obj)
    except Exception as e:
        print("[pd] load error:", e)
    print(f"[pd] loaded {len(fps)} fingerprints from {PD_FINGERPRINT_DIR}")
    return fps

PD_FPS = _load_pd_fingerprints()

def _make_ngrams(text: str, n: int) -> set:
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    grams = set()
    if len(toks) < n: return grams
    for i in range(len(toks)-n+1):
        grams.add(" ".join(toks[i:i+n]))
    return grams

def pd_overlap_score(text: str, n: int = PD_NGRAM_N) -> float:
    if not PD_FPS: return 0.0
    grams = _make_ngrams(text, n)
    if not grams: return 0.0
    best = 0.0
    for fp in PD_FPS:
        ref = set(fp.get("ngrams", {}).keys())
        inter = len(grams & ref)
        union = len(grams | ref)
        if union > 0:
            j = inter / union
            if j > best: best = j
    return best

# -------------------- LLM Fingerprint (v0.3.0 stub) --------------------
# Centroids are lightweight “style vectors” per model family.
# File format (one JSON per file in dir):
# {"family": "gpt4", "n": 120, "vector": [0.12, 0.34, ...]}  # fixed length

MODEL_CENTROID_DIR = os.getenv("MODEL_CENTROID_DIR", "./model_centroids")
FPRINT_MIN_TOKENS = int(os.getenv("FPRINT_MIN_TOKENS", "180"))

def _centroid_dir_ensure():
    pathlib.Path(MODEL_CENTROID_DIR).mkdir(parents=True, exist_ok=True)

def _load_model_centroids() -> List[Dict]:
    _centroid_dir_ensure()
    cents = []
    for path in glob.glob(os.path.join(MODEL_CENTROID_DIR, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "vector" in obj and "family" in obj:
                vec = obj["vector"]
                if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
                    cents.append({
                        "family": str(obj["family"]),
                        "vector": [float(x) for x in vec],
                        "n": int(obj.get("n", 0)),
                        "filename": os.path.basename(path)
                    })
        except Exception as e:
            print("[fingerprint] bad centroid:", path, e)
    print(f"[fingerprint] loaded {len(cents)} centroids from {MODEL_CENTROID_DIR}")
    return cents

MODEL_CENTROIDS = _load_model_centroids()

def _cosine_sim_lite(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a)) or 1e-9
    db = math.sqrt(sum(y*y for y in b)) or 1e-9
    s = num / (da * db)
    return max(0.0, min(1.0, s))

def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    Z = sum(exps) or 1.0
    return [e / Z for e in exps]

def _style_vector(text: str, scan_primary: Dict, stylo: Dict, sigs: Dict) -> List[float]:
    """
    Fixed-length 12D vector. Keep order/length stable across releases.
    """
    total = max(1, scan_primary.get("total", 0))
    bins = scan_primary.get("bins", {10:0, 100:0, "rest":0})
    frac10 = bins.get(10, 0)/total
    frac100 = bins.get(100, 0)/total
    return [
        float(stylo.get("func_ratio", 0.0)),
        float(stylo.get("hapax_ratio", 0.0)),
        float(stylo.get("sent_mean", 0.0)),
        float(stylo.get("sent_var", 0.0)),
        float(stylo.get("punct_entropy", 0.0)),
        float(sigs.get("clause_density", 0.0)),
        float(sigs.get("semi_dash_rate", 0.0)),
        float(sigs.get("dialog_density", 0.0)),
        float(scan_primary.get("ppl", 0.0)),
        float(scan_primary.get("burstiness", 0.0)),
        float(frac10),
        float(frac100),
    ]

def compute_llm_fingerprint(text: str, scan_primary: Dict, stylo: Dict, sigs: Dict) -> Dict:
    """
    Returns similarity distribution to known model families.
    If no centroids are present (or sample too short), returns {available: False}.
    """
    if scan_primary.get("total", 0) < FPRINT_MIN_TOKENS:
        return {"available": False, "reason": f"too_short(<{FPRINT_MIN_TOKENS} tokens)"}
    if not MODEL_CENTROIDS:
        return {"available": False, "reason": "no_centroids"}

    v = _style_vector(text, scan_primary, stylo, sigs)
    sims = [ _cosine_sim_lite(v, c["vector"]) for c in MODEL_CENTROIDS ]
    probs = _softmax(sims)

    fams = [c["family"] for c in MODEL_CENTROIDS]
    sim_map = {fams[i]: round(float(sims[i]), 4) for i in range(len(fams))}
    prob_map = {fams[i]: round(float(probs[i]), 4) for i in range(len(fams))}

    # Confidence = length adequacy + (top margin) + vector norm sanity
    top_idx = max(range(len(sims)), key=lambda i: sims[i])
    top = sims[top_idx]; second = sorted(sims)[-2] if len(sims) >= 2 else 0.0
    margin = max(0.0, top - second)
    length_ok = min(1.0, scan_primary.get("total", 0) / float(FPRINT_MIN_TOKENS))
    confidence = round(0.5*length_ok + 0.5*min(1.0, margin/0.15), 3)

    # Convert to a normalized 0..1 score where higher ~ more human-like distribution:
    # If you include a 'human' centroid, that will dominate. If not, we reward high entropy.
    dist_entropy = -sum(p*math.log(p+1e-12) for p in probs) / math.log(len(probs)) if len(probs) > 1 else 0.0
    human_score = round(float(dist_entropy), 3)

    return {
        "available": True,
        "nearest_family": fams[top_idx],
        "similarity": sim_map,
        "distribution": prob_map,
        "confidence": confidence,
        "human_score": human_score  # 0..1 (higher looks less like a single AI family)
    }


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

# -------------------- Semantic Drift (v0.3.0) --------------------
def _paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

def _bow_embed(paras: List[str]) -> List[Dict[str, int]]:
    # simple bag-of-words per paragraph (lowercase words only)
    vecs = []
    for p in paras:
        toks = re.findall(r"[a-z']+", p.lower())
        d = {}
        for t in toks:
            d[t] = d.get(t, 0) + 1
        vecs.append(d)
    return vecs

def _cosine_dict(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    num = sum(a.get(k,0)*b.get(k,0) for k in keys)
    da = math.sqrt(sum(v*v for v in a.values())) or 1e-9
    db = math.sqrt(sum(v*v for v in b.values())) or 1e-9
    s = num/(da*db)
    return max(0.0, min(1.0, s))

def compute_semantic_drift(text: str) -> Dict:
    paras = _paragraphs(text)
    if len(paras) <= 1:
        # shim for short inputs
        return {
            "available": False,
            "reason": "single_paragraph",
            "avg_adjacent_sim": 1.0,
            "std_adjacent_sim": 0.0,
            "low_drops": [],
            "risk": 0.5,
        }

    vecs = _bow_embed(paras)
    sims = [_cosine_dict(a,b) for a,b in zip(vecs, vecs[1:])]
    avg = sum(sims)/len(sims)
    var = sum((s-avg)*(s-avg) for s in sims)/len(sims)
    std = math.sqrt(var)

    # flag large downward jumps
    drops = []
    for i in range(1, len(sims)):
        delta = sims[i] - sims[i-1]
        if delta < -0.35:  # abrupt shift
            drops.append({"at_paragraph": i+1, "delta": round(float(delta),3)})

    # risk heuristic: too-flat (AI-ish) OR many big drops (incoherent)
    flat_penalty = 1.0 - min(1.0, std/0.18)         # std <~0.18 is suspiciously uniform
    jump_penalty = min(1.0, len(drops)*0.25)        # many big drops also suspicious
    # Lower risk is better (human); convert to human-style score later if needed
    risk = _clamp(0.5*flat_penalty + 0.5*jump_penalty, 0.0, 1.0)

    # normalize to a 0..1 "human-like" score for consistency with other signals
    human_score = 1.0 - risk

    return {
        "available": True,
        "avg_adjacent_sim": round(float(avg),3),
        "std_adjacent_sim": round(float(std),3),
        "low_drops": drops,
        "risk": round(float(risk),3),
        "score": round(float(human_score),3)
    }


# -------------------- Helpers used in /scan --------------------
def _english_confidence(txt: str) -> float:
    toks = re.findall(r"[A-Za-z']+", txt)
    if not toks: return 0.0
    lower = [t.lower() for t in toks]
    ascii_ratio = sum(t.isascii() and re.match(r"^[a-z']+$", t) is not None for t in lower) / len(lower)
    fw_hits = sum(1 for t in lower if t in FUNCTION_WORDS)
    fw_ratio = fw_hits / max(1, len(lower))
    return max(0.0, min(1.0, 0.6*ascii_ratio + 0.4*min(1.0, fw_ratio/0.10)))

def _bootstrap_instability(per_token: List[Dict], token_logps: List[float]) -> float:
    n = len(per_token)
    if n < 2*BOOTSTRAP_WINDOW:
        return 1.0 - min(1.0, n / float(2*BOOTSTRAP_WINDOW))
    stats = []
    for _ in range(BOOTSTRAP_SAMPLES):
        start = random.randint(0, max(1, n-BOOTSTRAP_WINDOW))
        end   = start + BOOTSTRAP_WINDOW
        window = per_token[start:end]
        ranks  = [t.get("rank", 10**9) for t in window]
        top10  = sum(1 for r in ranks if r <= 10) / max(1, len(ranks))
        lps    = token_logps[start:end] if token_logps else []
        mlp    = (sum(lps)/len(lps)) if lps else 0.0
        stats.append((top10, mlp))
    if len(stats) < 4:
        return 0.5
    top10s = [s[0] for s in stats]
    mlps   = [s[1] for s in stats]
    def _cv(arr):
        m = sum(arr)/len(arr)
        if m == 0: return 1.0
        v = sum((x-m)*(x-m) for x in arr)/len(arr)
        return min(1.0, math.sqrt(v)/abs(m))
    cv = 0.75*_cv(top10s) + 0.25*_cv([abs(x) for x in mlps])
    return _clamp(cv, 0.0, 1.0)

def _combine_models(primary_prob: float, secondary_prob: Optional[float], use_ensemble: bool) -> float:
    if (secondary_prob is None) or (not use_ensemble):
        return primary_prob
    # Weighted vote + disagreement tempering
    w1, w2 = 0.65, 0.35
    mix = _clamp(w1*primary_prob + w2*secondary_prob, 0.0, 1.0)
    if abs(primary_prob - secondary_prob) >= 0.30:
        mix = 0.5 + (mix-0.5)*0.8
    return mix

# -------------------- API: runtime config --------------------
@app.get("/config")
def get_config():
    return {
        "settings": SETTINGS.model_dump(),
        "second_model_available": bool(SECONDARY_READY),
    }

@app.post("/config")
def set_config(s: ScanSettings):
    global SETTINGS
    # sanitize mode
    if s.mode not in ("Balanced","Strict","Academic"):
        s.mode = "Balanced"
    # clamp numeric fields
    s.en_thresh = _clamp(s.en_thresh, 0.0, 1.0)
    s.non_en_cap = _clamp(s.non_en_cap, 0.0, 1.0)
    s.max_conf_unstable = _clamp(s.max_conf_unstable, 0.0, 1.0)
    s.max_conf_short = _clamp(s.max_conf_short, 0.0, 1.0)
    s.abstain_low = _clamp(s.abstain_low, 0.0, 1.0)
    s.abstain_high = _clamp(s.abstain_high, 0.0, 1.0)
    if s.abstain_low > s.abstain_high:
        s.abstain_low, s.abstain_high = s.abstain_high, s.abstain_low
    # don't allow enabling ensemble if model2 isn't ready
    if s.use_ensemble and not SECONDARY_READY:
        s.use_ensemble = False
    SETTINGS = s
    return {"ok": True, "settings": SETTINGS.model_dump()}

    # -------------------- PD Fingerprints Management API --------------------
def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name or "")

def _pd_dir_ensure():
    pathlib.Path(PD_FINGERPRINT_DIR).mkdir(parents=True, exist_ok=True)

@app.get("/pd/list")
def pd_list():
    """
    Return what's on disk and what's currently loaded.
    """
    _pd_dir_ensure()
    disk = []
    for path in glob.glob(os.path.join(PD_FINGERPRINT_DIR, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            disk.append({
                "filename": os.path.basename(path),
                "name": obj.get("name") or os.path.basename(path),
                "N": obj.get("N"),
                "ngrams_count": len((obj.get("ngrams") or {})),
            })
        except Exception as e:
            disk.append({
                "filename": os.path.basename(path),
                "name": os.path.basename(path),
                "error": str(e)
            })

    loaded = [{
        "name": fp.get("name") or "(unnamed)",
        "N": fp.get("N"),
        "ngrams_count": len((fp.get("ngrams") or {})),
    } for fp in PD_FPS]

    return {"dir": PD_FINGERPRINT_DIR, "disk": disk, "loaded": loaded}

@app.post("/pd/reload")
def pd_reload():
    """
    Reload PD fingerprints from disk.
    """
    global PD_FPS
    PD_FPS = _load_pd_fingerprints()
    return {"ok": True, "count": len(PD_FPS)}

@app.post("/pd/upload")
async def pd_upload(file: UploadFile = File(...)):
    """
    Accept a JSON fingerprint file with structure:
    {"name": "...", "N": <int>, "ngrams": {"...": count, ...}}
    """
    _pd_dir_ensure()
    data = await file.read()
    try:
        obj = json.loads(data.decode("utf-8"))
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Invalid JSON: {e}"}, status_code=400)

    if not (isinstance(obj, dict) and "ngrams" in obj and "N" in obj and isinstance(obj["ngrams"], dict)):
        return JSONResponse({"ok": False, "error": "Fingerprint must be an object with keys: 'ngrams' (dict) and 'N' (int)."}, status_code=400)

    # choose filename
    base = obj.get("name") or file.filename or f"pd_{int(time.time())}.json"
    base = _safe_name(base)
    if not base.endswith(".json"):
        base += ".json"
    dest = os.path.join(PD_FINGERPRINT_DIR, base)

    try:
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Failed to write file: {e}"}, status_code=500)

    # reload PD cache
    global PD_FPS
    PD_FPS = _load_pd_fingerprints()

    return {"ok": True, "saved_as": base, "loaded_count": len(PD_FPS)}

@app.delete("/pd/delete")
def pd_delete(filename: str = Query(..., description="Filename under PD_FINGERPRINT_DIR")):
    """
    Delete a fingerprint JSON from disk by filename (basename only).
    """
    _pd_dir_ensure()
    name = _safe_name(os.path.basename(filename))
    path = os.path.join(PD_FINGERPRINT_DIR, name)
    if not os.path.isfile(path):
        return JSONResponse({"ok": False, "error": "File not found."}, status_code=404)

    try:
        os.remove(path)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Failed to delete: {e}"}, status_code=500)

    # reload PD cache
    global PD_FPS
    PD_FPS = _load_pd_fingerprints()
    return {"ok": True, "deleted": name, "loaded_count": len(PD_FPS)}

# -------------------- Fingerprint Centroid Management --------------------
@app.get("/fingerprint/centroids")
def fp_list():
    return {
        "dir": MODEL_CENTROID_DIR,
        "count": len(MODEL_CENTROIDS),
        "families": [c["family"] for c in MODEL_CENTROIDS]
    }

@app.post("/fingerprint/reload")
def fp_reload():
    global MODEL_CENTROIDS
    MODEL_CENTROIDS = _load_model_centroids()
    return {"ok": True, "count": len(MODEL_CENTROIDS)}

# -------------------- Explain Mode (v0.3.1) --------------------
def _label_band(p: float) -> str:
    if p < 0.35:   return "Looks human"
    if p <= 0.65:  return "Inconclusive"
    return "Likely AI"

def _friendly_pct(x: float) -> str:
    try:    return f"{int(round(100*x))}%"
    except: return "—"

def _explain_from(resp: dict) -> dict:
    """
    Produce a plain-language explanation object from the /scan response.
    Designed to be UI-ready and non-technical by default.
    """
    ai_p = float(resp.get("calibrated_prob", 0.5))
    verdict = resp.get("verdict") or _label_band(ai_p)
    drift   = (resp.get("semantic_drift") or {})
    fp      = (resp.get("llm_fingerprint") or {})
    pd_j    = float(resp.get("pd_overlap_j", 0.0))
    cat     = (resp.get("category") or "other").replace("_", " ")

    bullets = []
    notes   = []
    fixes   = []

    # Headline
    if ai_p < 0.35:
        headline = "This looks human and original."
    elif ai_p <= 0.65:
        headline = "Mixed signals — not clearly AI or human."
    else:
        headline = "This likely contains AI-generated writing."

    # Why we think that (positive evidence)
    if fp.get("available"):
        nf = fp.get("nearest_family")
        human_score = fp.get("human_score")
        if nf == "human_baseline" or (human_score is not None and human_score >= 0.7):
            bullets.append("Your overall style matches typical human variation.")
        else:
            bullets.append(f"Closest style match: {nf} (not definitive).")

    if drift.get("available"):
        sc = drift.get("score", 0.5)
        if sc >= 0.45 and sc <= 0.65:
            bullets.append("Topic flow and tone shifts look natural for human writing.")
        elif sc < 0.35:
            bullets.append("Paragraphs are unusually uniform (possible model pattern).")
            fixes.append("Vary sentence length and vocabulary between paragraphs.")
        else:
            bullets.append("Paragraphs shift topics a lot (could be human brainstorming).")

    # Public-domain overlap note
    if pd_j >= 0.12:
        notes.append("We detected overlap with public-domain phrasing; we limited the AI score to avoid a false positive.")
    else:
        notes.append("No meaningful match with public-domain phrasing.")

    # Practical next steps depending on band
    if ai_p > 0.65:
        fixes.extend([
            "Add brief personal specifics (dates, places, unique details).",
            "Break up uniform sentences; mix short and long forms.",
            "Include a brief anecdote or reflection in your own voice."
        ])
    elif ai_p <= 0.65 and ai_p >= 0.35:
        fixes.append("If needed, add a few personal details or examples to strengthen human signal.")

    # Compact teacher-style summary
    teacher = {
        "headline": headline,
        "ai_likelihood": _friendly_pct(ai_p),
        "band": _label_band(ai_p),
        "nearest_style": (fp.get("nearest_family") if fp.get("available") else None),
        "drift_score": (drift.get("score") if drift.get("available") else None),
        "pd_overlap_j": round(pd_j, 3),
        "category": cat,
    }

    return {
        "headline": headline,
        "ai_likelihood_pct": _friendly_pct(ai_p),
        "band": _label_band(ai_p),
        "why": bullets,
        "notes": notes,
        "what_to_fix": fixes,
        "teacher_report": teacher
    }

# -------------------- Main route --------------------
@app.post("/scan")
def scan(inp: ScanIn):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail=f"Model unavailable: {model_load_error}")

    # Pull runtime settings
    mode = (inp.mode or SETTINGS.mode).strip().title()
    short_cap_on = SETTINGS.short_cap
    min_tokens_strong = SETTINGS.min_tokens_strong
    non_en_cap = SETTINGS.non_en_cap
    en_thresh = SETTINGS.en_thresh
    max_conf_unstable = SETTINGS.max_conf_unstable
    max_conf_short = SETTINGS.max_conf_short
    abstain_low = SETTINGS.abstain_low
    abstain_high = SETTINGS.abstain_high
    use_ensemble = SETTINGS.use_ensemble and SECONDARY_READY

    sens_boost, abstain_delta, artifact_bias = _mode_multipliers(mode)

    text = inp.text.strip()
    if not text:
        return {"overall_score": 0.0, "per_token": [], "explanation": "Empty text"}

    out1 = chunked_scan_primary(text)
    total = max(1, out1["total"])
    bins = out1["bins"]
    frac10 = bins[10]/total
    frac100 = bins[100]/total

    fPpl   = _clamp((25 - (out1['ppl'] or 25)) / 20, 0, 1)
    fBurst = _clamp((8  - (out1['burstiness'] or 8)) / 6, 0, 1)

    # Evidence / reliability
    ci10 = _binom_ci_halfwidth(frac10, total)
    len_factor = min(1.0, total / float(min_tokens_strong))
    shape_factor = 1.0 - min(0.6, ci10 * 1.8)
    reliability = max(0.15, len_factor * shape_factor)

    token_logps = [math.log(max(t.get("p", 0.0), 1e-12)) for t in out1["per_token"]]
    stylo = stylometric_fingerprint(text, token_logps)
    sigs  = prose_classic_signals(text)
    cscore = classic_style_score(text, stylo, sigs)

    # NEW (v0.3.0): LLM fingerprint computation (safe if no centroids / too short)
    fp = compute_llm_fingerprint(text, out1, stylo, sigs)

    # NEW (v0.3.0): Semantic drift
    drift = compute_semantic_drift(text)

    # primary calibration
    z = sens_boost * ((2.0*out1["score"]) + (0.9*fPpl) + (0.8*fBurst) + (0.35*frac10) - 1.6)
    calP1 = _clamp(1/(1+math.exp(-4*z)), 0.0, 1.0)

    # optional second model
    calP2 = None
    out2 = None
    if use_ensemble:
        out2 = chunked_scan_secondary(text)
        if out2:
            total2 = max(1, out2["total"])
            bins2 = out2["bins"]
            frac10_2 = bins2[10]/total2
            fPpl2   = _clamp((25 - (out2['ppl'] or 25)) / 20, 0, 1)
            fBurst2 = _clamp((8  - (out2['burstiness'] or 8)) / 6, 0, 1)
            z2 = sens_boost * ((2.0*out2["score"]) + (0.9*fPpl2) + (0.8*fBurst2) + (0.35*frac10_2) - 1.6)
            calP2 = _clamp(1/(1+math.exp(-4*z2)), 0.0, 1.0)

    calP = _combine_models(calP1, calP2, use_ensemble)

    # Artifact gate
    machine_artifact = (
        frac10 >= (0.985*artifact_bias) and
        frac100 <= 0.03 and
        out1["ppl"] <= (2.5/ artifact_bias) and
        out1["burstiness"] <= (3.5/ artifact_bias)
    )

    # Classic style decision
    looks_classic = (
        cscore >= 0.62 and
        6.0 <= (out1["ppl"] or 99.0) <= 28.0 and
        (out1["burstiness"] or 99.0) <= 12.0 and
        frac10 >= 0.45 and
        not machine_artifact
    )

    if looks_classic:
        cat = "classic_literature"
        cat_conf = 0.88 if cscore >= 0.72 else 0.78
        note = category_note_for_report(True, cscore)
        calP = min(calP, 0.15 if cscore >= 0.72 else 0.22)
    else:
        cat = "other"; cat_conf = 0.5
        note = category_note_for_report(False)

    if tag_says_classic(inp.tag) and not machine_artifact:
        calP = min(calP, 0.10)
        note += " | user-tag classic"

    # Nonsense guard
    nonsense = looks_nonsense_verse(text, out1, frac10, frac100)
    if nonsense["hit"]:
        calP = min(calP, 0.02)
        cat = "nonsense_literature"
        cat_conf = max(cat_conf, 0.90)
        note = ("Nonsense/Surreal verse safeguard — "
                f"rhyme={nonsense['signals']['rhyme_density']}, "
                f"meter_cv={nonsense['signals']['meter_cv']}, "
                f"invented={nonsense['signals']['invented_ratio']}, "
                f"lex_hits={nonsense['signals']['lex_hits']}, "
                f"sem_disc={nonsense['signals']['semantic_disc']}")

    # Short excerpt attenuation
    if short_cap_on and out1["total"] < min_tokens_strong and not machine_artifact:
        scale = max(0.25, out1["total"] / float(min_tokens_strong))
        calP = min(calP * scale, max_conf_short)
        note += f" | short-excerpt cap: {out1['total']} tokens"

    # Bootstrap instability shrink/cap
    instab = _bootstrap_instability(out1["per_token"], token_logps)
    if instab > 0.20 and not machine_artifact:
        calP *= 1.0 / (1.0 + 2.0*instab)
        calP = min(calP, max_conf_unstable)
        note += f" | instability cv≈{instab:.2f}"

    # Non-English cap (unless explicitly tagged classic)
    en_conf = _english_confidence(text)
    if en_conf < en_thresh and not tag_says_classic(inp.tag) and not machine_artifact:
        calP = min(calP, non_en_cap)
        note += f" | non-English cap (p_en≈{en_conf:.2f})"

    # Public-domain overlap dampener
    pd_score = pd_overlap_score(text, PD_NGRAM_N)
    if pd_score >= PD_DAMP_THRESHOLD and not machine_artifact:
        calP = min(calP, PD_MAX_CONF)
        note += f" | PD-overlap dampener (J≈{pd_score:.3f})"

    # Final reliability shrink
    calP = _shrink_toward_half(calP, reliability)

    # Abstain band + thermometer
    low  = max(0.0, abstain_low  + abstain_delta)
    high = min(1.0, abstain_high - abstain_delta)
    verdict = ("Likely human-written" if calP < low else
               "Inconclusive — human & model signals mixed" if calP <= high else
               "Likely AI-generated")
    thermometer_blocks = int(round(calP*10))
    thermo = "▇"*thermometer_blocks + "▒"*(10-thermometer_blocks)

    # row
    row = {
        "ts": int(time.time()),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text_len_chars": len(text),
        "text_len_tokens": out1["total"],
        "model_name": MODEL_NAME + (f" + {SECOND_MODEL}" if (use_ensemble and calP2 is not None) else ""),
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "overall_score": round(out1["score"], 6),
        "top10_frac": round(frac10, 6),
        "top100_frac": round(frac100, 6),
        "ppl": round(out1["ppl"], 6),
        "burstiness": round(out1["burstiness"], 6),
        "ai_likelihood_calibrated": round(calP, 6),
        "tag": (inp.tag or ""),
        "category": cat,
        "category_conf": round(cat_conf, 3),
        "category_note": note + f" | mode={mode}",
    }
    log_scan_row(row)

    percent = round(calP*100)
    exp = (
        f"Confidence {percent}% [{thermo}] — {verdict}. "
        f"{round(100*frac10)}% tokens in Top-10; {round(100*frac100)}% in Top-100. "
        f"PPL≈{out1['ppl']:.1f}; Burst≈{out1['burstiness']:.3f}. "
        f"Detected: {cat.replace('_',' ')} (conf≈{cat_conf:.0%}). {note}"
    )

    resp = {
        "overall_score": out1["score"], "per_token": out1["per_token"],
        "explanation": exp, "ppl": out1["ppl"], "burstiness": out1["burstiness"],
        "bins": bins, "total": out1["total"], "model_name": row["model_name"],
        "device": str(DEVICE), "dtype": str(DTYPE),
        "calibrated_prob": calP, "category": cat,
        "category_conf": cat_conf, "category_note": note,
        "stylometry": stylo, "nonsense_signals": nonsense["signals"],
        "mode": mode, "verdict": verdict, "thermometer": thermo,
        "pd_overlap_j": round(pd_score, 4),
        "llm_fingerprint": fp,
        "semantic_drift": drift
    }
    if out2:
        resp["secondary"] = {
            "ppl": out2["ppl"], "burstiness": out2["burstiness"],
            "total": out2["total"], "score": out2["score"]
        }

    # Attach friendly explanation (v0.3.1)
    try:
        resp["explain"] = _explain_from(resp)
    except Exception:
        resp["explain"] = {
            "headline": "Summary unavailable",
            "band": _label_band(resp.get("calibrated_prob", 0.5)),
            "why": [],
            "notes": ["Explain mode encountered an error."],
            "what_to_fix": []
        }

    return resp

    # -------------------- Live Typing Verification (Hybrid streaming) --------------------
import uuid

LIVE_SESSIONS: Dict[str, Dict] = {}

def _norm(x, lo, hi):
    if hi <= lo: return 0.0
    x = (x - lo) / (hi - lo)
    return max(0.0, min(1.0, x))

def _stylometry_vector(sty: Dict[str, float]) -> List[float]:
    # Order matters; keep stable across ref vs sample
    return [
        float(sty.get("func_ratio", 0.0)),
        float(sty.get("hapax_ratio", 0.0)),
        float(sty.get("sent_mean", 0.0)),
        float(sty.get("sent_var", 0.0)),
        float(sty.get("mlps", 0.0)),
        float(sty.get("mlps_var", 0.0)),
        float(sty.get("punct_entropy", 0.0)),
    ]

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    import math
    if not a or not b or len(a) != len(b): return 0.0
    num = sum(x*y for x,y in zip(a,b))
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(y*y for y in b))
    if da == 0 or db == 0: return 0.0
    return max(0.0, min(1.0, num/(da*db)))

def _compare_reference_vs_sample(ref_text: str, live_text: str) -> Dict[str, float]:
    """
    Compute a composite 'match' between the pasted reference text and the user's live-typed text.
    Uses stylometry and model rhythm (ppl/burst/top10/top100) from the primary scanner.
    Returns a dict with sub-scores and an overall match in [0,1].
    """
    # Stylometry signatures
    ref_scan = chunked_scan_primary(ref_text)
    live_scan = chunked_scan_primary(live_text) if live_text.strip() else {"total":0, "ppl":0.0, "burstiness":0.0, "bins":{10:0,100:0,"rest":0}}
    ref_lps = [math.log(max(t.get("p", 0.0), 1e-12)) for t in ref_scan.get("per_token",[])]
    live_lps= [math.log(max(t.get("p", 0.0), 1e-12)) for t in live_scan.get("per_token",[])]

    ref_sty = stylometric_fingerprint(ref_text, ref_lps)
    live_sty= stylometric_fingerprint(live_text, live_lps)

    # Vector similarity
    sim_sty = _cosine_similarity(_stylometry_vector(ref_sty), _stylometry_vector(live_sty))

    # Rhythm resemblance (normalize diffs)
    # Typical ranges (rough): ppl∈[5,60], burst∈[0,20], top10/top100 in [0,1]
    def _top_fracs(scan):
        total = max(1, scan.get("total", 0))
        b = scan.get("bins", {10:0,100:0,"rest":0})
        return b.get(10,0)/total, b.get(100,0)/total

    r_top10, r_top100 = _top_fracs(ref_scan)
    s_top10, s_top100 = _top_fracs(live_scan)

    ppl_ref, ppl_live = float(ref_scan.get("ppl", 0.0)), float(live_scan.get("ppl", 0.0))
    burst_ref, burst_live = float(ref_scan.get("burstiness", 0.0)), float(live_scan.get("burstiness", 0.0))

    # Convert differences into similarities
    ppl_sim   = 1.0 - _norm(abs(ppl_ref - ppl_live), 0.0, 40.0)
    burst_sim = 1.0 - _norm(abs(burst_ref - burst_live), 0.0, 12.0)
    t10_sim   = 1.0 - _norm(abs(r_top10 - s_top10), 0.0, 0.40)
    t100_sim  = 1.0 - _norm(abs(r_top100 - s_top100), 0.0, 0.50)

    # Short sample penalty (need ~80+ tokens for stable matching)
    len_penalty = 1.0
    if live_scan.get("total", 0) < 80:
        len_penalty = 0.85
    if live_scan.get("total", 0) < 40:
        len_penalty = 0.70

    # Composite (weights sum ≈ 1.0 before length penalty)
    match = (
        0.45 * sim_sty +
        0.18 * ppl_sim +
        0.18 * burst_sim +
        0.10 * t10_sim +
        0.09 * t100_sim
    ) * len_penalty

    match = _clamp(match, 0.0, 1.0)

    return {
        "sim_sty": round(sim_sty, 4),
        "sim_ppl": round(ppl_sim, 4),
        "sim_burst": round(burst_sim, 4),
        "sim_top10": round(t10_sim, 4),
        "sim_top100": round(t100_sim, 4),
        "len_penalty": round(len_penalty, 3),
        "match": round(match, 6),
        "live_tokens": int(live_scan.get("total", 0))
    }

class SampleStartIn(BaseModel):
    reference_text: str
    duration_sec: Optional[int] = 90  # UI timer; not enforced server-side beyond guidance

class SampleSubmitIn(BaseModel):
    session_id: str
    text_chunk: str
    done: Optional[bool] = False

class SampleFinalizeIn(BaseModel):
    session_id: str

@app.post("/auth/sample/start")
def auth_sample_start(inp: SampleStartIn):
    ref = inp.reference_text.strip()
    if not ref:
        raise HTTPException(status_code=400, detail="reference_text is empty")
    sid = str(uuid.uuid4())
    LIVE_SESSIONS[sid] = {
        "ts": time.time(),
        "duration_sec": int(inp.duration_sec or 90),
        "reference": ref,
        "accum": [],
        "final": None
    }
    return {"session_id": sid, "duration_sec": LIVE_SESSIONS[sid]["duration_sec"]}

@app.post("/auth/sample/submit")
def auth_sample_submit(inp: SampleSubmitIn):
    sess = LIVE_SESSIONS.get(inp.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")
    chunk = (inp.text_chunk or "").strip()
    if chunk:
        sess["accum"].append(chunk)
    live_text = " ".join(sess["accum"]).strip()
    # Provide progressive feedback (tokens typed + rough similarity)
    if live_text:
        cmpd = _compare_reference_vs_sample(sess["reference"], live_text)
        prog = {
            "live_tokens": cmpd["live_tokens"],
            "rough_match_0_1": cmpd["match"]
        }
    else:
        prog = {"live_tokens": 0, "rough_match_0_1": 0.0}

    if inp.done:
        # Finalize now
        cmpd = _compare_reference_vs_sample(sess["reference"], live_text)
        sess["final"] = cmpd
        return {"ok": True, "final": True, "result": cmpd}
    return {"ok": True, "final": False, "progress": prog}

@app.post("/auth/sample/finalize")
def auth_sample_finalize(inp: SampleFinalizeIn):
    sess = LIVE_SESSIONS.get(inp.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")
    live_text = " ".join(sess["accum"]).strip()
    cmpd = _compare_reference_vs_sample(sess["reference"], live_text) if live_text else {
        "sim_sty":0,"sim_ppl":0,"sim_burst":0,"sim_top10":0,"sim_top100":0,"len_penalty":0,"match":0,"live_tokens":0
    }
    sess["final"] = cmpd

    # Simple verdict thresholds
    m = cmpd["match"]
    if m >= 0.70 and cmpd["live_tokens"] >= 80:
        verdict = "Strong match"
        note = "Live stylometry & rhythm align with the reference sample."
    elif m >= 0.50 and cmpd["live_tokens"] >= 60:
        verdict = "Moderate match"
        note = "Core style signals align; consider longer sample to strengthen."
    else:
        verdict = "Weak/No match"
        note = "Insufficient alignment or sample too short to confirm."

    return {
        "session_id": inp.session_id,
        "match_score": round(m*100),
        "verdict": verdict,
        "explanation": note,
        "details": cmpd
    }

# -------------------- Demo route --------------------
DEMO_TEXTS = [
    {
        "label": "Human — journal paragraph",
        "text": ("I missed my train this morning and walked the long way instead. "
                 "The sidewalks were still wet from last night’s storm, and the maples "
                 "had the sweet, sharp smell they get after the first cold snap. I kept "
                 "rehearsing the question I should have asked yesterday and didn’t."),
        "expect": "Likely Human"
    },
    {
        "label": "Classic-style prose (public-domain-like)",
        "text": ("“You mustn’t be uneasy,” said the gentleman, turning aside the curtain "
                 "with a grave composure, “for the hour is not yet come; and if the wind "
                 "keeps steady we shall have our answer before the bells have done.”"),
        "expect": "Human w/ classic safeguard"
    },
    {
        "label": "AI — generic essay tone",
        "text": ("In conclusion, it is important to recognize that technology has both "
                 "advantages and disadvantages. By thoughtfully balancing innovation and "
                 "ethics, society can create a future that is inclusive and sustainable."),
        "expect": "Likely AI"
    },
    {
        "label": "Nonsense-verse",
        "text": ("’Twas brillig, and the slithy toves did gyre and gimble in the wabe; "
                 "all mimsy were the borogoves, and the mome raths outgrabe."),
        "expect": "Nonsense guard → Human"
    },
    {
        "label": "AI-rewrite of human",
        "text": ("This week I explored an AI music tool after reading a study suggesting "
                 "listeners often can’t tell AI compositions from human ones. I chose a song "
                 "that matters to me and compared the generated piece to the original to "
                 "evaluate structure, pacing, and emotional expression."),
        "expect": "Inconclusive / Likely AI"
    }
]

@app.get("/demo")
def demo():
    sample = random.sample(DEMO_TEXTS, k=min(3, len(DEMO_TEXTS)))
    return JSONResponse(sample)

# -------------------- Serve static UI --------------------
@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = pathlib.Path(__file__).parent / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return """
    <html><head><title>AI Text Scanner</title></head>
    <body style="font-family: sans-serif; max-width: 720px; margin: 40px;">
      <h1>AI Text Scanner</h1>
      <p>POST <code>/scan</code> with JSON {"text": "...", "tag": "optional", "mode":"Balanced|Strict|Academic"}</p>
      <p>GET <code>/config</code> to view settings; POST <code>/config</code> to change them.</p>
      <p>GET <code>/demo</code> for sample texts. GET <code>/version</code> for build info.</p>
    </body></html>
    """

@app.get("/version")
def version():
    return {
        "version": VERSION,
        "model": MODEL_NAME,
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "ensemble": (SETTINGS.use_ensemble and bool(SECONDARY_READY)),
        "secondary_model": (SECOND_MODEL if (SETTINGS.use_ensemble and SECONDARY_READY) else None),
        "mode": SETTINGS.mode,
        "fingerprint_centroids": len(MODEL_CENTROIDS),
    }

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)
