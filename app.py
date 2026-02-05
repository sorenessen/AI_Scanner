# app.py — AI Text Scanner (pattern-classic + nonsense + ensemble + abstain + PD-dampener + runtime config)
# Notarization/PyInstaller-safe: never writes into the .app bundle, uses ~/Library/Application Support/CopyCat

from __future__ import annotations

import multiprocessing

import csv
import glob
import hashlib
import json
import math
import os
import pathlib
import random
import re
import io
import string
import sys
import time
import uuid
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- PDF (ReportLab) ----
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import docx  # python-docx
except Exception:
    docx = None


# =============================================================================
# Paths (PyInstaller-safe)
# =============================================================================

def resource_path(rel: str) -> pathlib.Path:
    """
    Returns a real filesystem path to a bundled resource in both dev and PyInstaller frozen mode.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = pathlib.Path(sys._MEIPASS)
    else:
        base = pathlib.Path(__file__).parent
    return base / rel


def user_data_dir() -> pathlib.Path:
    """
    Writable location for packaged apps.

    macOS:  ~/Library/Application Support/CopyCat
    Windows: %LOCALAPPDATA%\\CopyCat   (fallback: %APPDATA%\\CopyCat)
    Linux:  ~/.local/share/CopyCat
    """
    if sys.platform.startswith("win"):
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(pathlib.Path.home() / "AppData" / "Local")
        p = pathlib.Path(base) / "CopyCat"
    elif sys.platform == "darwin":
        p = pathlib.Path.home() / "Library" / "Application Support" / "CopyCat"
    else:
        p = pathlib.Path(os.getenv("XDG_DATA_HOME") or (pathlib.Path.home() / ".local" / "share")) / "CopyCat"

    p.mkdir(parents=True, exist_ok=True)
    return p



def user_documents_dir() -> pathlib.Path:
    r"""
    Best-effort 'Documents' directory across macOS + Windows + Linux.

    macOS:    ~/Documents
    Windows:  %USERPROFILE%\Documents, or OneDrive 'Documents' if redirected
    Linux:    ~/Documents (fallback if XDG_DOCUMENTS_DIR not available)
    """
    # Windows: OneDrive can redirect Documents; prefer it if present.
    if sys.platform.startswith("win"):
        userprofile = os.getenv("USERPROFILE") or str(pathlib.Path.home())
        # Common OneDrive env vars (personal / business)
        onedrive = (
            os.getenv("OneDriveCommercial")
            or os.getenv("OneDrive")
            or os.getenv("OneDriveConsumer")
        )
        candidates = []
        if onedrive:
            candidates.append(pathlib.Path(onedrive) / "Documents")
        candidates.append(pathlib.Path(userprofile) / "Documents")
        # Fallback: home/Documents
        candidates.append(pathlib.Path.home() / "Documents")
        for c in candidates:
            try:
                c.mkdir(parents=True, exist_ok=True)
                return c
            except Exception:
                continue
        return pathlib.Path.home()

    # macOS + Linux
    docs = pathlib.Path.home() / "Documents"
    try:
        docs.mkdir(parents=True, exist_ok=True)
    except Exception:
        return pathlib.Path.home()
    return docs




def reports_output_dir() -> pathlib.Path:
    r"""
    Choose a writable Reports directory for PDF outputs.

    Windows note:
      Many users keep "CopyCat" under OneDrive root (OneDrive\CopyCat\Reports),
      while others expect it under Documents (Documents\CopyCat\Reports).
      We prefer the OneDrive root when available, then fall back to Documents.

    You can override this completely with env var COPYCAT_REPORTS_DIR.
    """
    override = os.environ.get("COPYCAT_REPORTS_DIR")
    if override:
        try:
            return ensure_dir(pathlib.Path(override).expanduser())
        except Exception:
            # fall through to defaults
            pass

    docs = user_documents_dir()

    if sys.platform.startswith("win"):
        # OneDrive root (most common)
        onedrive = (
            os.environ.get("OneDriveCommercial")
            or os.environ.get("OneDriveConsumer")
            or os.environ.get("OneDrive")
        )
        if onedrive:
            od = pathlib.Path(onedrive)
            if od.exists():
                try:
                    return ensure_dir(od / "CopyCat" / "Reports")
                except Exception:
                    pass

    return ensure_dir(docs / "CopyCat" / "Reports")

def ensure_dir(p: pathlib.Path) -> pathlib.Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


APP_DATA_DIR = user_data_dir()
PD_USER_DIR = ensure_dir(APP_DATA_DIR / "pd_fingerprints")
CENTROIDS_USER_DIR = ensure_dir(APP_DATA_DIR / "model_centroids")
LOG_PATH = str(APP_DATA_DIR / "scan_logs.csv")


REPORTS_DIR = reports_output_dir()
# --- Diagnostics: print resolved reports directory at startup (helps verify Windows/OneDrive vs Documents)
try:
    import logging as _logging
    _logging.getLogger("copycat").info("Reports dir resolved to: %s", str(REPORTS_DIR))
except Exception as e:
    print("LOGO DRAW ERROR:", e)

print(f"[CopyCat] Reports dir resolved to: {REPORTS_DIR}")

PD_BUNDLE_DIR = resource_path("pd_fingerprints")
CENTROIDS_BUNDLE_DIR = resource_path("model_centroids")

# Optional env overrides (still should be writable if you expect uploads)
PD_FINGERPRINT_DIR = pathlib.Path(os.getenv("PD_FINGERPRINT_DIR") or str(PD_USER_DIR))
MODEL_CENTROID_DIR = pathlib.Path(os.getenv("MODEL_CENTROID_DIR") or str(CENTROIDS_USER_DIR))
ensure_dir(PD_FINGERPRINT_DIR)
ensure_dir(MODEL_CENTROID_DIR)


# =============================================================================
# Stylometry Layer v2
# =============================================================================

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

# (Optional) live verification scratch state
LAST_SCAN_STYLO: Optional[Dict[str, float]] = None
LAST_SCAN_META: Optional[Dict[str, float]] = None


def stylometric_fingerprint(text: str, token_logps: List[float]) -> Dict[str, float]:
    words = [w.strip(string.punctuation).lower() for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return {
            "func_ratio": 0.0,
            "hapax_ratio": 0.0,
            "sent_mean": 0.0,
            "sent_var": 0.0,
            "mlps": 0.0,
            "mlps_var": 0.0,
            "punct_entropy": 0.0,
        }

    func_ratio = sum(1 for w in words if w in FUNCTION_WORDS) / len(words)

    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    hapax_ratio = sum(1 for _, c in freq.items() if c == 1) / max(1, len(freq))

    sents = re.split(r"[.!?]+", text)
    lengths = [len(s.split()) for s in sents if s.strip()]
    sent_mean = mean(lengths) if lengths else 0.0
    sent_var = pstdev(lengths) if len(lengths) > 1 else 0.0

    if len(token_logps) > 5:
        slopes = [token_logps[i + 1] - token_logps[i] for i in range(len(token_logps) - 1)]
        mlps = mean(slopes)
        mlps_var = pstdev(slopes) if len(slopes) > 1 else 0.0
    else:
        mlps, mlps_var = 0.0, 0.0

    punct = [c for c in text if c in ".,;:!?-—–"]
    if punct:
        uniq = list(set(punct))
        freqs = [punct.count(p) / len(punct) for p in uniq]
        punct_entropy = -sum(p * math.log(max(p, 1e-12)) for p in freqs)
    else:
        punct_entropy = 0.0

    return {
        "func_ratio": float(func_ratio),
        "hapax_ratio": float(hapax_ratio),
        "sent_mean": float(sent_mean),
        "sent_var": float(sent_var),
        "mlps": float(mlps),
        "mlps_var": float(mlps_var),
        "punct_entropy": float(punct_entropy),
    }


# =============================================================================
# Config (env defaults)
# =============================================================================

VERSION = "0.5.0"  # adds /config runtime settings

MODEL_NAME = os.getenv("REF_MODEL", "EleutherAI/gpt-neo-1.3B")
SECOND_MODEL_ENV = os.getenv("SECOND_MODEL", "distilgpt2")  # optional
ENABLE_SECOND_MODEL_ENV = os.getenv("ENABLE_SECOND_MODEL", "0") == "1"

MAX_TOKENS_PER_PASS = int(os.getenv("MAX_TOKENS_PER_PASS", "768"))
STRIDE = int(os.getenv("STRIDE", "128"))
USE_FP16 = os.getenv("USE_FP16", "1") == "1"

# Confidence + abstain
ABSTAIN_LOW_DEF = float(os.getenv("ABSTAIN_LOW", "0.35"))
ABSTAIN_HIGH_DEF = float(os.getenv("ABSTAIN_HIGH", "0.65"))

# Hardening
MIN_TOKENS_STRONG_DEF = int(os.getenv("MIN_TOKENS_STRONG", "180"))
SHORT_CAP_DEF = (os.getenv("SHORT_CAP", "0") == "1")
MAX_CONF_SHORT_DEF = float(os.getenv("MAX_CONF_SHORT", "0.35"))

BOOTSTRAP_SAMPLES = int(os.getenv("BOOTSTRAP_SAMPLES", "24"))
BOOTSTRAP_WINDOW = int(os.getenv("BOOTSTRAP_WINDOW", "256"))
MAX_CONF_UNSTABLE_DEF = float(os.getenv("MAX_CONF_UNSTABLE", "0.35"))
NON_EN_CAP_DEF = float(os.getenv("NON_EN_CAP", "0.15"))
EN_THRESH_DEF = float(os.getenv("EN_THRESH", "0.70"))

DEFAULT_MODE = os.getenv("MODE", "Balanced").strip().title()
if DEFAULT_MODE not in ("Balanced", "Strict", "Academic"):
    DEFAULT_MODE = "Balanced"

# Public-domain statistical dampener
PD_NGRAM_N = int(os.getenv("PD_NGRAM_N", "5"))
PD_DAMP_THRESHOLD = float(os.getenv("PD_DAMP_THRESHOLD", "0.12"))
PD_MAX_CONF = float(os.getenv("PD_MAX_CONF", "0.25"))

# PDF logging
ENABLE_PDF = os.getenv("ENABLE_PDF", "1") == "1"


class ScanSettings(BaseModel):
    mode: str = DEFAULT_MODE  # Balanced | Strict | Academic
    short_cap: bool = SHORT_CAP_DEF
    min_tokens_strong: int = MIN_TOKENS_STRONG_DEF
    use_ensemble: bool = ENABLE_SECOND_MODEL_ENV  # only if secondary is loaded
    non_en_cap: float = NON_EN_CAP_DEF
    en_thresh: float = EN_THRESH_DEF
    max_conf_unstable: float = MAX_CONF_UNSTABLE_DEF
    max_conf_short: float = MAX_CONF_SHORT_DEF
    abstain_low: float = ABSTAIN_LOW_DEF
    abstain_high: float = ABSTAIN_HIGH_DEF


SETTINGS = ScanSettings()


def _mode_multipliers(mode: str) -> Tuple[float, float, float]:
    # (sensitivity_boost, abstain_delta, artifact_bias)
    m = (mode or DEFAULT_MODE).strip().title()
    if m == "Strict":
        return (1.10, -0.03, 1.05)
    if m == "Academic":
        return (0.90, 0.05, 0.90)
    return (1.00, 0.00, 1.00)


def _binom_ci_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.5
    var = p * (1 - p) / n
    return z * math.sqrt(max(var, 0.0))


def _shrink_toward_half(prob: float, reliability: float) -> float:
    r = max(0.0, min(1.0, reliability))
    return 0.5 + (prob - 0.5) * r


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =============================================================================
# Device setup
# =============================================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

DTYPE = torch.float16 if (USE_FP16 and (DEVICE.type in {"cuda", "mps"})) else torch.float32


# =============================================================================
# App + CORS + Static
# =============================================================================

app = FastAPI(title="AI Text Scanner (Pattern Classic + Nonsense + Ensemble)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")

app.mount(
  "/reports",
  StaticFiles(directory="/Users/sorenessen/Documents/CopyCat/Reports"),
  name="reports"
)



def _resolve_static_dir() -> pathlib.Path:
    """Resolve where UI static/ assets live in both dev + bundled builds.

    Resolution order:
      1) $STATIC_DIR (explicit override)
      2) PyInstaller bundle root via resource_path("static")
      3) Project checkout next to this file: <repo>/static
      4) macOS .app bundle: CopyCat.app/Contents/Resources/static
      5) Legacy fallback: CopyCat.app/Contents/Frameworks/static
    """
    env = os.getenv("STATIC_DIR")
    if env:
        try:
            p = pathlib.Path(env)
            if p.exists() and p.is_dir():
                return p
        except Exception:
            pass

    candidates = [
        resource_path("static"),
        pathlib.Path(__file__).resolve().parent / "static",
        pathlib.Path(sys.executable).resolve().parent.parent / "Resources" / "static",
        pathlib.Path(sys.executable).resolve().parent.parent / "Frameworks" / "static",
    ]
    for p in candidates:
        try:
            if p.exists() and p.is_dir():
                return p
        except Exception:
            pass
    return candidates[0]


STATIC_DIR = _resolve_static_dir()
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    print("[server] WARNING: static directory missing:", str(STATIC_DIR))


# =============================================================================
# UI Routes (serve bundled index.html)
# =============================================================================

def _resolve_ui_html() -> pathlib.Path:
    """Find index.html in both dev + PyInstaller frozen mode."""
    p = resource_path("index.html")
    if p.exists():
        return p
    # Fallback: next to this file (dev checkout) or current working dir
    here = pathlib.Path(__file__).resolve().parent
    for cand in (here / "index.html", pathlib.Path.cwd() / "index.html"):
        try:
            if cand.exists():
                return cand
        except Exception:
            pass
    return p  # best-effort

@app.get("/ui")
def ui():
    p = _resolve_ui_html()
    if not p.exists():
        return HTMLResponse(
            "<h2>CopyCat UI missing</h2><p>index.html was not found/bundled.</p>",
            status_code=500,
        )
    return FileResponse(str(p))

@app.get("/m")
def mobile_home():
    return FileResponse("static/mobile/index.html")

# =============================================================================
# Model load (safe)
# =============================================================================

tokenizer: Optional[AutoTokenizer] = None
model: Optional[AutoModelForCausalLM] = None
tokenizer2: Optional[AutoTokenizer] = None
model2: Optional[AutoModelForCausalLM] = None
model_load_error: Optional[str] = None
SECONDARY_READY = False
SECOND_MODEL = SECOND_MODEL_ENV

print("[server] loading model(s)...")

SECONDARY_READY = False
model2 = None
tokenizer2 = None

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

            SECONDARY_READY = True
            print(f"[server] second model OK: {SECOND_MODEL}")

        except Exception as e2:
            print("[server] failed to load secondary model:", str(e2))
            model2 = None
            tokenizer2 = None
            SECONDARY_READY = False
    else:
        model2 = None
        tokenizer2 = None
        SECONDARY_READY = False

except Exception as e:
    model_load_error = str(e)
    print("[server] FAILED TO LOAD MODEL:", model_load_error)



# =============================================================================
# IO Models
# =============================================================================

class ScanIn(BaseModel):
    text: str
    tag: Optional[str] = None
    mode: Optional[str] = None  # override for a single call


class DriftIn(BaseModel):
    text: str


class DriftCompareIn(BaseModel):
    scan_text: str
    live_text: str


# =============================================================================
# Core scoring helpers (model-agnostic)
# =============================================================================

def scan_chunk_with(model_obj, tok, input_ids: torch.Tensor) -> Dict:
    model_obj.eval()
    with torch.no_grad():
        out = model_obj(input_ids=input_ids)
        logits = out.logits  # [1, T, V]

    ids = input_ids[0]
    per_token: List[Dict] = []
    topk_bins = {10: 0, 100: 0, 1000: 0, "rest": 0}
    total = max(0, ids.size(0) - 1)
    logprobs: List[float] = []

    for i in range(1, ids.size(0)):
        next_logits = logits[0, i - 1]
        probs = torch.softmax(next_logits, dim=-1)
        actual_id = int(ids[i].item())
        p_actual = float(probs[actual_id].item())
        rank = int((probs > p_actual).sum().item() + 1)

        if rank <= 10:
            topk_bins[10] += 1
        elif rank <= 100:
            topk_bins[100] += 1
        elif rank <= 1000:
            topk_bins[1000] += 1
        else:
            topk_bins["rest"] += 1

        tok_str = tok.convert_ids_to_tokens([actual_id])[0]
        per_token.append({"t": tok_str, "rank": rank, "p": p_actual})
        logprobs.append(math.log(max(p_actual, 1e-12)))

    ppl = math.exp(-sum(logprobs) / max(1, len(logprobs))) if logprobs else float("inf")
    mean_lp = (sum(logprobs) / len(logprobs)) if logprobs else 0.0
    burstiness = (
        (sum((lp - mean_lp) ** 2 for lp in logprobs) / max(1, len(logprobs)))
        if logprobs
        else 0.0
    )

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


def _chunked_scan_with(
    model_obj,
    tok,
    text: str,
    *,
    max_tokens_per_pass: int,
    stride: int,
) -> Dict:
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0].to(DEVICE)

    def _scan_ids(slice_ids: torch.Tensor) -> Dict:
        return scan_chunk_with(model_obj, tok, slice_ids.unsqueeze(0))

    if ids.size(0) <= max_tokens_per_pass:
        return _scan_ids(ids)

    all_tokens: List[Dict] = []
    agg_bins = {10: 0, 100: 0, 1000: 0, "rest": 0}
    agg_total = 0
    ppls: List[float] = []
    bursts: List[float] = []

    start = 0
    T = ids.size(0)

    while start < T:
        end = min(start + max_tokens_per_pass, T)
        result = _scan_ids(ids[start:end])

        all_tokens.extend(result["per_token"])
        for k in agg_bins:
            agg_bins[k] += result["bins"][k]
        agg_total += result["total"]
        ppls.append(float(result["ppl"]))
        bursts.append(float(result["burstiness"]))

        if end == T:
            break
        start = max(0, end - stride)

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
        "ppl": (sum(ppls) / len(ppls) if ppls else 0.0),
        "burstiness": (sum(bursts) / len(bursts) if bursts else 0.0),
    }


def chunked_scan_primary(text: str) -> Dict:
    if model is None or tokenizer is None:
        raise RuntimeError("Primary model not loaded")
    return _chunked_scan_with(
        model,
        tokenizer,
        text,
        max_tokens_per_pass=MAX_TOKENS_PER_PASS,
        stride=STRIDE,
    )


def chunked_scan_secondary(text: str) -> Optional[Dict]:
    if not (SECONDARY_READY and model2 is not None and tokenizer2 is not None):
        return None
    # keep it cheaper for distil/tiny models
    max_tokens = min(MAX_TOKENS_PER_PASS, 512)
    return _chunked_scan_with(
        model2,
        tokenizer2,
        text,
        max_tokens_per_pass=max_tokens,
        stride=min(STRIDE, 128),
    )


# =============================================================================
# Pattern-based Classic Style
# =============================================================================

def prose_classic_signals(text: str) -> dict:
    quotes = re.findall(r"[“\"]([^”\"]+)[”\"]", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    dialog_density = (len(quotes) / max(1, len(lines)))
    words = re.findall(r"\b[A-Za-z][a-z]+\b", text)
    proper_nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    proper_ratio = len(proper_nouns) / max(1, len(words))
    semi_dash = len(re.findall(r"[;:—–]", text))
    commas = text.count(",")
    exclam_q = len(re.findall(r"[!?]", text))
    lines_n = max(1, len(lines))
    sent_like = max(1, len(re.split(r"[.!?]+", text)))
    clause_density = (commas + semi_dash) / float(sent_like)
    semi_dash_rate = semi_dash / float(lines_n)
    exclamq_rate = exclam_q / float(lines_n)
    return {
        "dialog_density": dialog_density,
        "proper_ratio": proper_ratio,
        "clause_density": clause_density,
        "semi_dash_rate": semi_dash_rate,
        "exclamq_rate": exclamq_rate,
        "sent_like": sent_like,
    }


def classic_style_score(text: str, stylo: Dict, sigs: Dict) -> float:
    sm = stylo.get("sent_mean", 0.0)
    sv = stylo.get("sent_var", 0.0)
    pe = stylo.get("punct_entropy", 0.0)
    hr = stylo.get("hapax_ratio", 0.0)
    fr = stylo.get("func_ratio", 0.0)
    dd = sigs["dialog_density"]
    cd = sigs["clause_density"]
    sdr = sigs["semi_dash_rate"]

    f_len = _clamp((sm - 16.0) / 16.0, 0.0, 1.0)
    f_var = _clamp((10.0 - abs(10.0 - sv)) / 10.0, 0.0, 1.0)
    f_clause = _clamp(cd / 1.2, 0.0, 1.0)
    f_semi = _clamp(sdr / 0.06, 0.0, 1.0)
    f_dialog = _clamp(1.0 - abs(dd - 0.18) / 0.18, 0.0, 1.0)
    f_vocab = _clamp((0.16 - abs(0.16 - hr)) / 0.16, 0.0, 1.0)
    f_pent = _clamp(pe / 1.6, 0.0, 1.0)
    f_func = _clamp(1.0 - abs(fr - 0.52) / 0.15, 0.0, 1.0)

    score = (
        0.18 * f_len
        + 0.10 * f_var
        + 0.20 * f_clause
        + 0.12 * f_semi
        + 0.18 * f_dialog
        + 0.10 * f_vocab
        + 0.07 * f_pent
        + 0.05 * f_func
    )
    return _clamp(score, 0.0, 1.0)


def tag_says_classic(tag: Optional[str]) -> bool:
    if not tag:
        return False
    t = tag.lower()
    return any(k in t for k in ("classic", "pre-1920", "public_domain", "public-domain", "pre1920"))


def category_note_for_report(is_classic: bool, style_score: Optional[float] = None) -> str:
    if is_classic:
        base = "Classic-style safeguard: structure/rhythm match; avoiding false positives."
        if style_score is not None:
            return f"{base} style_score≈{style_score:.2f}"
        return base
    return "Default calibration."


# =============================================================================
# Nonsense / Surreal Verse helpers
# =============================================================================

CARROLL_LEXICON = re.compile(
    r"\b(jabberwock|bandersnatch|jubjub|borogove|outgrabe|brillig|slithy|toves|"
    r"gyre|gimble|mome|raths|vorpal|frabjous|manxome|galumph|tulgey|uffish)\b",
    re.I,
)


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    syl = max(1, len(groups))
    if w.endswith("e") and syl > 1 and not w.endswith(("le", "ye")):
        syl -= 1
    return max(1, syl)


def _rhyme_key(word: str, min_tail: int = 2, max_tail: int = 4) -> str:
    w = re.sub(r"[^a-z]", "", word.lower())
    m = re.search(r"[aeiouy][a-z]*$", w)
    tail = (m.group(0) if m else w)
    if len(tail) <= min_tail:
        return tail
    return tail[-max_tail:]


def rhyme_density(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return 0.0
    endings: List[str] = []
    for ln in lines:
        toks = [t.strip(string.punctuation) for t in ln.split()]
        endings.append(_rhyme_key(toks[-1]) if toks else "")
    counts: Dict[str, int] = {}
    for r in endings:
        counts[r] = counts.get(r, 0) + 1
    pairs = sum(c * (c - 1) // 2 for c in counts.values())
    possible = len(endings) * (len(endings) - 1) // 2
    return pairs / max(1, possible)


def meter_periodicity(text: str) -> float:
    lines = [ln for ln in (l.strip() for l in text.splitlines()) if ln]
    if len(lines) < 3:
        return 1.0
    syls: List[int] = []
    for ln in lines:
        words = [w.strip(string.punctuation) for w in ln.split()]
        syls.append(sum(_syllables(w) for w in words))
    m = mean(syls) if syls else 0.0
    s = pstdev(syls) if len(syls) > 1 else 0.0
    return (s / m) if m > 0 else 1.0


def invented_word_ratio(text: str) -> float:
    words = [w.strip(string.punctuation) for w in re.findall(r"[A-Za-z'-]+", text)]
    if not words:
        return 0.0
    common = set(w.lower() for w in FUNCTION_WORDS)
    novel = 0
    for w in words:
        lw = w.lower()
        if len(lw) <= 3 or w[0].isupper() or lw in common:
            continue
        if re.search(r"[bcdfghjklmnpqrstvwxz]{3,}", lw) or re.search(r"(zz|zx|xq|qk|kk)", lw):
            novel += 1
            continue
        if re.search(r"(wock|snatch|mome|borog|jubjub|frum|vorpal|tulgey|uffish|bander)", lw):
            novel += 1
            continue
        if not re.search(r"[aeiouy]", lw):
            novel += 1
    return novel / max(1, len(words))


def semantic_discontinuity(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    no_punct = sum(1 for ln in lines if not re.search(r"[.!?;:]", ln))
    shortish = sum(1 for ln in lines if len(ln.split()) <= 8)
    return 0.5 * (no_punct / max(1, len(lines))) + 0.5 * (shortish / max(1, len(lines)))


def looks_nonsense_verse(text: str, metrics: Dict, frac10: float, frac100: float) -> Dict:
    lex_hits = len(CARROLL_LEXICON.findall(text))
    ppl = float(metrics.get("ppl", 99.0))
    burst = float(metrics.get("burstiness", 99.0))

    r_density = rhyme_density(text)
    cv_meter = meter_periodicity(text)
    inv_ratio = invented_word_ratio(text)
    sem_disc = semantic_discontinuity(text)

    hit = False
    if lex_hits >= 1:
        hit = True
    else:
        hit = bool(
            r_density >= 0.22
            and cv_meter <= 0.60
            and inv_ratio >= 0.06
            and sem_disc >= 0.25
            and frac10 >= 0.75
            and ppl <= 14.0
            and burst <= 8.0
        )

    return {
        "hit": hit,
        "signals": {
            "rhyme_density": round(r_density, 3),
            "meter_cv": round(cv_meter, 3),
            "invented_ratio": round(inv_ratio, 3),
            "lex_hits": int(lex_hits),
            "semantic_disc": round(sem_disc, 3),
            "ppl": round(ppl, 3),
            "burst": round(burst, 3),
            "top10": round(frac10, 3),
            "top100": round(frac100, 3),
        },
    }


# =============================================================================
# Public-Domain Overlap Dampener
# =============================================================================

def _load_pd_fingerprints() -> List[Dict]:
    fps: List[Dict] = []
    seen = set()

    # Prefer writable dir first, then bundle dir (bundle is read-only in .app)
    search_dirs: List[pathlib.Path] = []
    if PD_FINGERPRINT_DIR:
        search_dirs.append(PD_FINGERPRINT_DIR)
    if PD_BUNDLE_DIR and PD_BUNDLE_DIR.exists() and PD_BUNDLE_DIR != PD_FINGERPRINT_DIR:
        search_dirs.append(PD_BUNDLE_DIR)

    try:
        for d in search_dirs:
            for path in glob.glob(str(d / "*.json")):
                base = os.path.basename(path)
                if base in seen:
                    continue
                seen.add(base)
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict) and "ngrams" in obj and "N" in obj:
                    fps.append(obj)
    except Exception as e:
        print("[pd] load error:", e)

    print(f"[pd] loaded {len(fps)} fingerprints from writable={PD_FINGERPRINT_DIR} + bundle={PD_BUNDLE_DIR}")
    return fps


PD_FPS = _load_pd_fingerprints()


def _make_ngrams(text: str, n: int) -> set:
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    grams = set()
    if len(toks) < n:
        return grams
    for i in range(len(toks) - n + 1):
        grams.add(" ".join(toks[i : i + n]))
    return grams


def pd_overlap_score(text: str, n: int = PD_NGRAM_N) -> float:
    if not PD_FPS:
        return 0.0
    grams = _make_ngrams(text, n)
    if not grams:
        return 0.0
    best = 0.0
    for fp in PD_FPS:
        ref = set((fp.get("ngrams") or {}).keys())
        inter = len(grams & ref)
        union = len(grams | ref)
        if union > 0:
            j = inter / union
            if j > best:
                best = j
    return best


# =============================================================================
# LLM Fingerprint (centroid similarity)
# =============================================================================

FPRINT_MIN_TOKENS = int(os.getenv("FPRINT_MIN_TOKENS", "180"))


def _load_model_centroids() -> List[Dict]:
    cents: List[Dict] = []

    # Prefer writable dir first, then bundle dir
    search_dirs: List[pathlib.Path] = [MODEL_CENTROID_DIR]
    if CENTROIDS_BUNDLE_DIR.exists() and CENTROIDS_BUNDLE_DIR != MODEL_CENTROID_DIR:
        search_dirs.append(CENTROIDS_BUNDLE_DIR)

    for d in search_dirs:
        for path in glob.glob(str(d / "*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict) and "vector" in obj and "family" in obj:
                    vec = obj["vector"]
                    if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
                        cents.append(
                            {
                                "family": str(obj["family"]),
                                "vector": [float(x) for x in vec],
                                "n": int(obj.get("n", 0)),
                                "filename": os.path.basename(path),
                            }
                        )
            except Exception as e:
                print("[fingerprint] bad centroid:", path, e)

    print(f"[fingerprint] loaded {len(cents)} centroids from writable={MODEL_CENTROID_DIR} + bundle={CENTROIDS_BUNDLE_DIR}")
    return cents


MODEL_CENTROIDS = _load_model_centroids()


def _cosine_sim_lite(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1e-9
    db = math.sqrt(sum(y * y for y in b)) or 1e-9
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
    total = max(1, int(scan_primary.get("total", 0)))
    bins = scan_primary.get("bins", {10: 0, 100: 0, 1000: 0, "rest": 0})
    frac10 = float(bins.get(10, 0)) / total
    frac100 = float(bins.get(100, 0)) / total
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
    if int(scan_primary.get("total", 0)) < FPRINT_MIN_TOKENS:
        return {"available": False, "reason": f"too_short(<{FPRINT_MIN_TOKENS} tokens)"}
    if not MODEL_CENTROIDS:
        return {"available": False, "reason": "no_centroids"}

    v = _style_vector(text, scan_primary, stylo, sigs)
    sims = [_cosine_sim_lite(v, c["vector"]) for c in MODEL_CENTROIDS]
    probs = _softmax(sims)

    fams = [c["family"] for c in MODEL_CENTROIDS]
    sim_map = {fams[i]: round(float(sims[i]), 4) for i in range(len(fams))}
    prob_map = {fams[i]: round(float(probs[i]), 4) for i in range(len(fams))}

    top_idx = max(range(len(sims)), key=lambda i: sims[i])
    top = sims[top_idx]
    second = sorted(sims)[-2] if len(sims) >= 2 else 0.0
    margin = max(0.0, top - second)
    length_ok = min(1.0, int(scan_primary.get("total", 0)) / float(FPRINT_MIN_TOKENS))
    confidence = round(0.5 * length_ok + 0.5 * min(1.0, margin / 0.15), 3)

    dist_entropy = (
        -sum(p * math.log(p + 1e-12) for p in probs) / math.log(len(probs))
        if len(probs) > 1
        else 0.0
    )
    human_score = round(float(dist_entropy), 3)

    return {
        "available": True,
        "nearest_family": fams[top_idx],
        "similarity": sim_map,
        "distribution": prob_map,
        "confidence": confidence,
        "human_score": human_score,
    }


# =============================================================================
# Logging + PDF
# =============================================================================

def create_pdf_summary(row: dict) -> str:
    """
    Generate a polished PDF scan report (ReportLab/Platypus).

    Optional branding assets (drop these into your bundled `assets/` folder):
      - assets/copycat_logo.png
      - assets/calypso_logo.png

    If the files don't exist, the report renders without them.
    """
    timestamp = row.get("timestamp") or time.strftime("%Y-%m-%d_%H-%M-%S")
    pdf_path = str(REPORTS_DIR / f"scan_summary_{timestamp}.pdf")

    # -------------------------
    # Helpers
    # -------------------------
    def _as_float(v, default=None):
        try:
            return float(v)
        except Exception:
            return default

    def _as_int(v, default=None):
        try:
            return int(v)
        except Exception:
            return default

    def _pct(v, digits=1):
        f = _as_float(v)
        return "—" if f is None else f"{f:.{digits}f}%"

    def _num(v, digits=2):
        f = _as_float(v)
        return "—" if f is None else f"{f:.{digits}f}"

    def _short_hash(h: str, n: int = 12) -> str:
        if not h:
            return "—"
        h = str(h)
        return h[:n]

    # Logos (optional)
    def _resolve_asset(rel_path: str) -> pathlib.Path:
        # 1) PyInstaller _MEIPASS or next-to-this-file
        p = resource_path(rel_path)
        if p.exists():
            return p

        # 2) macOS .app bundle Resources
        if sys.platform == "darwin":
            try:
                mac_resources = pathlib.Path(sys.executable).resolve().parent.parent / "Resources"
                p2 = mac_resources / rel_path
                if p2.exists():
                    return p2
            except Exception:
                pass

        return p

    logo_copycat = _resolve_asset("assets/copycat_logo.png")
    logo_calypso = _resolve_asset("assets/calypso_logo.png")

    logos = [p for p in (logo_copycat, logo_calypso) if p and p.exists()]

    for lp in (logo_copycat, logo_calypso):
        try:
            if lp and os.path.exists(lp):
                logos.append(lp)
        except Exception:
            pass

    # -------------------------
    # Document + styles
    # -------------------------
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=0.72 * inch,
        rightMargin=0.72 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.75 * inch,
        title="CopyCat — AI Text Scan Report",
        author="CopyCat by Calypso Labs",
    )

    styles = getSampleStyleSheet()
    # Add a few nicer styles without depending on extra fonts
    styles.add(ParagraphStyle(
        name="CC_Title",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="CC_Subtitle",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="CC_H2",
        parent=styles["Heading2"],
        fontSize=13.5,
        leading=16,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor("#111827"),
    ))
    styles.add(ParagraphStyle(
        name="CC_Small",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#374151"),
    ))
    styles.add(ParagraphStyle(
        name="CC_Mono",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8.8,
        leading=11,
        textColor=colors.HexColor("#111827"),
    ))

    # Palette (keep it neutral + "trust blue")
    C_BG = colors.HexColor("#0B1020")
    C_BLUE = colors.HexColor("#2563EB")
    C_BLUE_SOFT = colors.HexColor("#DBEAFE")
    C_BORDER = colors.HexColor("#E5E7EB")
    C_TEXT = colors.HexColor("#111827")
    C_MUTED = colors.HexColor("#6B7280")

    # -------------------------
    # Header/footer (drawn on canvas)
    # -------------------------
    report_id = f"scan_summary_{timestamp}"
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    text_hash = row.get("text_hash") or row.get("hash") or ""

    def _draw_header_footer(c, d):
        w, h = letter
        # Top band
        c.saveState()
        c.setFillColor(C_BG)
        c.rect(0, h - 52, w, 52, stroke=0, fill=1)

        # Title in band
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(doc.leftMargin, h - 34, "CopyCat — AI Text Scan Report")

        # Optional tiny logos on the right
        x_right = w - doc.rightMargin
        y_logo = h - 45
        if logos:
            # Render at most 2, right-aligned
            size_h = 22
            gap = 8
            x = x_right

            for lp in reversed(logos[:2]):
                try:
                    from reportlab.lib.utils import ImageReader
                    ir = ImageReader(str(lp))
                    iw, ih = ir.getSize()
                    scale = size_h / float(ih)
                    size_w = iw * scale
                    x -= size_w
                    c.drawImage(ir, x, y_logo, width=size_w, height=size_h, mask="auto")
                    x -= gap
                except Exception:
                    pass

        # Footer
        c.setFillColor(C_MUTED)
        c.setFont("Helvetica", 8.5)
        footer_left = f"{report_id}  •  text_hash={_short_hash(text_hash)}"
        footer_right = f"Generated {generated_at}  •  Page {c.getPageNumber()}"
        c.drawString(doc.leftMargin, 18, footer_left)
        c.drawRightString(w - doc.rightMargin, 18, footer_right)
        c.restoreState()

    # -------------------------
    # Content
    # -------------------------
    # NOTE: the in-memory `row` uses internal keys (e.g. ai_likelihood_calibrated, top10_frac),
    # while the CSV uses human headers (e.g. "Likelihood %", "Top-10 %"). The PDF supports both.
    verdict = str(row.get("verdict") or row.get("Verdict") or "").strip()
    if not verdict:
        cal = _as_float(row.get("ai_likelihood_calibrated"))
        if cal is not None:
            verdict = "Likely AI-generated" if cal >= 0.6 else "Likely human-written"

    # Likelihood: prefer explicit percent, else derive from calibrated 0..1 likelihood
    likelihood = _as_float(row.get("likelihood_pct") or row.get("likelihood") or row.get("Likelihood %"))
    if likelihood is None:
        cal = _as_float(row.get("ai_likelihood_calibrated"))
        if cal is not None:
            likelihood = cal * 100.0

    overall = _as_float(row.get("overall_score") or row.get("Predictability Score"))

    # Top-k: accept either percent or fraction (0..1)
    top10 = _as_float(row.get("top10") or row.get("Top-10 %"))
    if top10 is None:
        frac = _as_float(row.get("top10_frac"))
        if frac is not None:
            top10 = frac * 100.0

    top100 = _as_float(row.get("top100") or row.get("Top-100 %"))
    if top100 is None:
        frac = _as_float(row.get("top100_frac"))
        if frac is not None:
            top100 = frac * 100.0

    ppl = _as_float(row.get("ppl") or row.get("Perplexity"))
    burst = _as_float(row.get("burstiness") or row.get("Burstiness"))

    model_name = row.get("model") or row.get("Model") or row.get("model_name") or row.get("modelName") or "—"
    device = row.get("device") or row.get("Device") or "—"

    n_chars = _as_int(row.get("chars") or row.get("Chars") or row.get("text_len_chars"))
    n_tokens = _as_int(row.get("tokens") or row.get("Tokens") or row.get("text_len_tokens"))

    tag = row.get("tag") or row.get("Tag") or ""
    category = row.get("category") or row.get("Category") or ""
    category_conf = row.get("category_conf") or row.get("Category Conf")
    category_note = row.get("category_note") or row.get("Category Note") or ""

    # A small, human-readable interpretation line
    def _interpret(lh: float | None) -> str:
        if lh is None:
            return "Likelihood could not be computed for this scan."
        if lh >= 85:
            return "Strong indicators of AI generation."
        if lh >= 65:
            return "Moderate indicators of AI generation."
        if lh >= 45:
            return "Mixed signal. Could be edited or hybrid."
        if lh >= 25:
            return "Leans human, with some AI-like regularity."
        return "Strong indicators of human-authored text."

    story = []

    # Cover heading
    story.append(Paragraph("CopyCat — AI Text Scan Report", styles["CC_Title"]))
    subtitle_bits = [
        f"<b>Verdict:</b> {verdict}",
        f"<b>Likelihood:</b> {_pct(likelihood, 1)}",
        f"<b>Report ID:</b> {report_id}",
    ]
    story.append(Paragraph(" &nbsp;&nbsp;•&nbsp;&nbsp; ".join(subtitle_bits), styles["CC_Subtitle"]))

    # At-a-glance table
    at_glance = [
        ["Likelihood", _pct(likelihood, 1), "Predictability", _num(overall, 2)],
        ["Top-10 %", _pct(top10, 1), "Top-100 %", _pct(top100, 1)],
        ["Perplexity", "—" if ppl is None else f"{ppl:.1f}", "Burstiness", _num(burst, 2)],
        ["Tokens", "—" if n_tokens is None else f"{n_tokens:,}", "Chars", "—" if n_chars is None else f"{n_chars:,}"],
        ["Model", str(model_name), "Device", str(device)],
    ]
    if tag or category:
        at_glance.append(["Tag", str(tag or "—"), "Category", str(category or "—")])

    t = Table(at_glance, colWidths=[1.1*inch, 2.0*inch, 1.2*inch, 2.0*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_BLUE_SOFT),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_TEXT),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, C_BORDER),
        ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.18 * inch))

    # Interpretation
    story.append(Paragraph("Likelihood Indicator", styles["CC_H2"]))
    story.append(Paragraph(_interpret(likelihood), styles["Normal"]))

    # Add calibration / category notes if present
    notes_bits = []
    if category:
        if category_conf is not None and str(category_conf).strip() != "":
            try:
                notes_bits.append(f"<b>Category:</b> {category} (conf {float(category_conf):.2f})")
            except Exception:
                notes_bits.append(f"<b>Category:</b> {category}")
        else:
            notes_bits.append(f"<b>Category:</b> {category}")
    if category_note:
        notes_bits.append(f"<b>Note:</b> {str(category_note)}")

    if notes_bits:
        story.append(Spacer(1, 0.08 * inch))
        story.append(Paragraph("<br/>".join(notes_bits), styles["CC_Small"]))

    # Metrics explanation cards
    story.append(Paragraph("What these metrics mean", styles["CC_H2"]))
    cards = [
        ["Predictability Score", "A composite score combining multiple signals. Higher usually means more model-like regularity.", _num(overall, 2)],
        ["Top-10 %", "Percent of tokens that fell within the model's top-10 predictions. Higher can indicate templated phrasing.", _pct(top10, 1)],
        ["Top-100 %", "Percent within the model's top-100 predictions. Used with Top-10% to detect “over-confident” runs.", _pct(top100, 1)],
        ["Perplexity", "Average token surprise. Very low can mean highly predictable / repetitive structure.", "—" if ppl is None else f"{ppl:.1f}"],
        ["Burstiness", "Variance of surprise across the text. Humans often show rhythm and unevenness.", _num(burst, 2)],
    ]
    # 2-column table layout: metric + value, with definition under it
    card_rows = []
    for name, desc, value in cards:
        card_rows.append([
            Paragraph(f"<b>{name}</b><br/><font color='#6B7280'>" + desc + "</font>", styles["CC_Small"]),
            Paragraph(f"<b>{value}</b>", styles["Normal"]),
        ])
    cards_tbl = Table(card_rows, colWidths=[5.2*inch, 1.1*inch])
    cards_tbl.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, C_BORDER),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(cards_tbl)

    # Chart: normalized bars
    story.append(Paragraph("Signal overview", styles["CC_H2"]))
    try:
        # Normalize metrics to 0–100 scale for a compact "radar-like" bar
        # For ppl/burst we use gentle transforms to keep the chart stable.
        v_top10 = 0 if top10 is None else max(0.0, min(100.0, top10))
        v_top100 = 0 if top100 is None else max(0.0, min(100.0, top100))
        v_ppl = 0 if ppl is None else max(0.0, min(100.0, (ppl / 100.0) * 100.0))  # assume ppl ~ 20-120 typical
        v_burst = 0 if burst is None else max(0.0, min(100.0, burst * 100.0))        # burst usually small-ish
        data = [[v_top10, v_top100, v_ppl, v_burst]]

        drawing = Drawing(420, 180)
        drawing.add(Rect(0, 0, 420, 180, fillColor=colors.white, strokeColor=C_BORDER, strokeWidth=1))

        chart = VerticalBarChart()
        chart.x = 40
        chart.y = 30
        chart.height = 120
        chart.width = 360
        chart.data = data
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = 100
        chart.valueAxis.valueStep = 20
        chart.categoryAxis.categoryNames = ["Top-10 %", "Top-100 %", "Perplexity*", "Burstiness*"]
        chart.bars[0].fillColor = C_BLUE
        chart.barLabels.nudge = 7
        chart.barLabelFormat = "%0.0f"
        chart.barLabels.fontSize = 8
        chart.barLabels.fillColor = C_TEXT
        chart.valueAxis.labels.fontSize = 8
        chart.categoryAxis.labels.fontSize = 8

        drawing.add(chart)
        drawing.add(String(42, 10, "*scaled to 0–100 for display", fontSize=7, fillColor=C_MUTED))
        story.append(drawing)
    except Exception:
        # Don't fail report generation over charting
        story.append(Paragraph("Signal chart unavailable for this run.", styles["CC_Small"]))

    # Raw details
    story.append(Paragraph("Technical details", styles["CC_H2"]))
    details = [
        ["Report ID", report_id],
        ["Generated", generated_at],
        ["Text hash", str(text_hash or "—")],
    ]
    if row.get("filepath"):
        details.append(["Input file", str(row.get("filepath"))])
    det_tbl = Table(details, colWidths=[1.2*inch, 5.1*inch])
    det_tbl.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, C_BORDER),
        ("BACKGROUND", (0, 0), (-1, 0), C_BLUE_SOFT),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(det_tbl)

    doc.build(story, onFirstPage=_draw_header_footer, onLaterPages=_draw_header_footer)
    return pdf_path


def log_scan_row(row: dict) -> None:
    pathlib.Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    csv_exists = os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(
                [
                    "# Legend:",
                    "Each row = one scan result.",
                    "Likelihood % = probability text was AI-generated.",
                    "Verdict interprets likelihood; Category is auto-detected (with classic/nonsense guards).",
                ]
            )
            writer.writerow([])
            writer.writerow(
                [
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
                    "Tag",
                    "Category",
                    "Category Conf",
                    "Category Note",
                ]
            )

        verdict = "Likely AI-generated" if float(row["ai_likelihood_calibrated"]) >= 0.6 else "Likely human-written"
        # Add convenience keys for the PDF generator (keeps report robust even if internals change)
        row.setdefault("verdict", verdict)
        row.setdefault("likelihood_pct", float(row["ai_likelihood_calibrated"]) * 100.0)
        row.setdefault("top10", float(row.get("top10_frac", 0.0)) * 100.0)
        row.setdefault("top100", float(row.get("top100_frac", 0.0)) * 100.0)
        row.setdefault("model", row.get("model_name"))
        row.setdefault("chars", row.get("text_len_chars"))
        row.setdefault("tokens", row.get("text_len_tokens"))
        writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["ts"])),
                verdict,
                f"{int(float(row['ai_likelihood_calibrated']) * 100)}%",
                f"{float(row['overall_score']):.2f}",
                f"{float(row['top10_frac']) * 100:.1f}%",
                f"{float(row['top100_frac']) * 100:.1f}%",
                f"{float(row['ppl']):.1f}",
                f"{float(row['burstiness']):.2f}",
                row["model_name"],
                row["device"],
                row["text_len_chars"],
                row["text_len_tokens"],
                str(row["text_sha256"])[:8],
                row.get("tag") or "",
                row.get("category", "other"),
                f"{int(100 * float(row.get('category_conf', 0)))}%",
                row.get("category_note", ""),
            ]
        )

    if ENABLE_PDF:
        pdf_path = create_pdf_summary(row)
        print(f"[+] PDF report created: {pdf_path}")


# =============================================================================
# Semantic Drift (simple BoW)
# =============================================================================

def _paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def _bow_embed(paras: List[str]) -> List[Dict[str, int]]:
    vecs: List[Dict[str, int]] = []
    for p in paras:
        toks = re.findall(r"[a-z']+", p.lower())
        d: Dict[str, int] = {}
        for t in toks:
            d[t] = d.get(t, 0) + 1
        vecs.append(d)
    return vecs


def _cosine_dict(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    num = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    da = math.sqrt(sum(v * v for v in a.values())) or 1e-9
    db = math.sqrt(sum(v * v for v in b.values())) or 1e-9
    s = num / (da * db)
    return max(0.0, min(1.0, s))


def compute_semantic_drift(text: str) -> Dict:
    paras = _paragraphs(text)
    if len(paras) <= 1:
        return {
            "available": False,
            "reason": "single_paragraph",
            "paragraphs": len(paras),
            "avg_adjacent_sim": 1.0,
            "std_adjacent_sim": 0.0,
            "adjacent_sim_series": [],
            "low_drops": [],
            "risk": 0.5,
            "score": 0.5,
        }

    vecs = _bow_embed(paras)
    sims = [_cosine_dict(a, b) for a, b in zip(vecs, vecs[1:])]
    avg = sum(sims) / len(sims)
    var = sum((s - avg) * (s - avg) for s in sims) / len(sims)
    std = math.sqrt(var)

    drops = []
    for i in range(1, len(sims)):
        delta = sims[i] - sims[i - 1]
        if delta < -0.35:
            drops.append({"at_paragraph": i + 1, "delta": round(float(delta), 3)})

    flat_penalty = 1.0 - min(1.0, std / 0.18)
    jump_penalty = min(1.0, len(drops) * 0.25)
    risk = _clamp(0.5 * flat_penalty + 0.5 * jump_penalty, 0.0, 1.0)
    human_score = 1.0 - risk

    return {
        "available": True,
        "paragraphs": len(paras),
        "avg_adjacent_sim": round(float(avg), 3),
        "std_adjacent_sim": round(float(std), 3),
        "adjacent_sim_series": [round(float(x), 3) for x in sims],
        "low_drops": drops,
        "risk": round(float(risk), 3),
        "score": round(float(human_score), 3),
    }


def _global_bow_sim(a_text: str, b_text: str) -> float:
    va = _bow_embed([a_text])[0] if a_text.strip() else {}
    vb = _bow_embed([b_text])[0] if b_text.strip() else {}
    return round(_cosine_dict(va, vb), 4)


# =============================================================================
# Helpers used in /scan
# =============================================================================

def _english_confidence(txt: str) -> float:
    toks = re.findall(r"[A-Za-z']+", txt)
    if not toks:
        return 0.0
    lower = [t.lower() for t in toks]
    ascii_ratio = sum(t.isascii() and (re.match(r"^[a-z']+$", t) is not None) for t in lower) / len(lower)
    fw_hits = sum(1 for t in lower if t in FUNCTION_WORDS)
    fw_ratio = fw_hits / max(1, len(lower))
    return max(0.0, min(1.0, 0.6 * ascii_ratio + 0.4 * min(1.0, fw_ratio / 0.10)))


def _bootstrap_instability(per_token: List[Dict], token_logps: List[float]) -> float:
    n = len(per_token)
    if n < 2 * BOOTSTRAP_WINDOW:
        return 1.0 - min(1.0, n / float(2 * BOOTSTRAP_WINDOW))

    stats: List[Tuple[float, float]] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        start = random.randint(0, max(1, n - BOOTSTRAP_WINDOW))
        end = start + BOOTSTRAP_WINDOW
        window = per_token[start:end]
        ranks = [t.get("rank", 10**9) for t in window]
        top10 = sum(1 for r in ranks if r <= 10) / max(1, len(ranks))
        lps = token_logps[start:end] if token_logps else []
        mlp = (sum(lps) / len(lps)) if lps else 0.0
        stats.append((top10, mlp))

    if len(stats) < 4:
        return 0.5

    top10s = [s[0] for s in stats]
    mlps = [s[1] for s in stats]

    def _cv(arr: List[float]) -> float:
        m = sum(arr) / len(arr)
        if m == 0:
            return 1.0
        v = sum((x - m) * (x - m) for x in arr) / len(arr)
        return min(1.0, math.sqrt(v) / abs(m))

    cv = 0.75 * _cv(top10s) + 0.25 * _cv([abs(x) for x in mlps])
    return _clamp(cv, 0.0, 1.0)


def _combine_models(primary_prob: float, secondary_prob: Optional[float], use_ensemble: bool) -> float:
    if (secondary_prob is None) or (not use_ensemble):
        return primary_prob
    w1, w2 = 0.65, 0.35
    mix = _clamp(w1 * primary_prob + w2 * secondary_prob, 0.0, 1.0)
    if abs(primary_prob - secondary_prob) >= 0.30:
        mix = 0.5 + (mix - 0.5) * 0.8
    return mix


# =============================================================================
# API: runtime config
# =============================================================================

@app.get("/config")
def get_config():
    return {
        "settings": SETTINGS.model_dump(),
        "second_model_available": bool(SECONDARY_READY),
        "paths": {
            "app_data": str(APP_DATA_DIR),
            "pd_fingerprints": str(PD_FINGERPRINT_DIR),
            "model_centroids": str(MODEL_CENTROID_DIR),
            "scan_log": str(LOG_PATH),
            "reports_dir": str(REPORTS_DIR),
        },
    }


@app.post("/config")
def set_config(s: ScanSettings):
    global SETTINGS

    if s.mode not in ("Balanced", "Strict", "Academic"):
        s.mode = "Balanced"

    s.en_thresh = _clamp(float(s.en_thresh), 0.0, 1.0)
    s.non_en_cap = _clamp(float(s.non_en_cap), 0.0, 1.0)
    s.max_conf_unstable = _clamp(float(s.max_conf_unstable), 0.0, 1.0)
    s.max_conf_short = _clamp(float(s.max_conf_short), 0.0, 1.0)
    s.abstain_low = _clamp(float(s.abstain_low), 0.0, 1.0)
    s.abstain_high = _clamp(float(s.abstain_high), 0.0, 1.0)

    if s.abstain_low > s.abstain_high:
        s.abstain_low, s.abstain_high = s.abstain_high, s.abstain_low

    if s.use_ensemble and not SECONDARY_READY:
        s.use_ensemble = False

    SETTINGS = s
    return {"ok": True, "settings": SETTINGS.model_dump()}


# =============================================================================
# PD Fingerprints Management API
# =============================================================================

def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name or "")


@app.get("/pd/list")
def pd_list():
    disk = []
    for path in glob.glob(str(PD_FINGERPRINT_DIR / "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            disk.append(
                {
                    "filename": os.path.basename(path),
                    "name": obj.get("name") or os.path.basename(path),
                    "N": obj.get("N"),
                    "ngrams_count": len((obj.get("ngrams") or {})),
                }
            )
        except Exception as e:
            disk.append({"filename": os.path.basename(path), "name": os.path.basename(path), "error": str(e)})

    loaded = [
        {"name": fp.get("name") or "(unnamed)", "N": fp.get("N"), "ngrams_count": len((fp.get("ngrams") or {}))}
        for fp in PD_FPS
    ]

    return {"dir": str(PD_FINGERPRINT_DIR), "disk": disk, "loaded": loaded}


@app.post("/pd/reload")
def pd_reload():
    global PD_FPS
    PD_FPS = _load_pd_fingerprints()
    return {"ok": True, "count": len(PD_FPS)}


@app.post("/pd/upload")
async def pd_upload(file: UploadFile = File(...)):
    data = await file.read()
    try:
        obj = json.loads(data.decode("utf-8"))
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Invalid JSON: {e}"}, status_code=400)

    if not (isinstance(obj, dict) and "ngrams" in obj and "N" in obj and isinstance(obj["ngrams"], dict)):
        return JSONResponse(
            {"ok": False, "error": "Fingerprint must be an object with keys: 'ngrams' (dict) and 'N' (int)."},
            status_code=400,
        )

    base = obj.get("name") or file.filename or f"pd_{int(time.time())}.json"
    base = _safe_name(base)
    if not base.endswith(".json"):
        base += ".json"

    dest = PD_FINGERPRINT_DIR / base
    try:
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Failed to write file: {e}"}, status_code=500)

    global PD_FPS
    PD_FPS = _load_pd_fingerprints()
    return {"ok": True, "saved_as": base, "loaded_count": len(PD_FPS)}


@app.delete("/pd/delete")
def pd_delete(filename: str = Query(..., description="Filename under PD_FINGERPRINT_DIR (basename only)")):
    name = _safe_name(os.path.basename(filename))
    path = PD_FINGERPRINT_DIR / name
    if not path.is_file():
        return JSONResponse({"ok": False, "error": "File not found."}, status_code=404)

    try:
        path.unlink()
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Failed to delete: {e}"}, status_code=500)

    global PD_FPS
    PD_FPS = _load_pd_fingerprints()
    return {"ok": True, "deleted": name, "loaded_count": len(PD_FPS)}


# =============================================================================
# Fingerprint Centroid Management
# =============================================================================

@app.get("/fingerprint/centroids")
def fp_list():
    return {
        "dir": str(MODEL_CENTROID_DIR),
        "count": len(MODEL_CENTROIDS),
        "families": [c["family"] for c in MODEL_CENTROIDS],
    }


@app.post("/fingerprint/reload")
def fp_reload():
    global MODEL_CENTROIDS
    MODEL_CENTROIDS = _load_model_centroids()
    return {"ok": True, "count": len(MODEL_CENTROIDS)}


# =============================================================================
# Explain Mode
# =============================================================================

def _label_band(p: float) -> str:
    if p < 0.35:
        return "Looks human"
    if p <= 0.65:
        return "Inconclusive"
    return "Likely AI"


def _friendly_pct(x: float) -> str:
    try:
        return f"{int(round(100 * x))}%"
    except Exception:
        return "—"


def _explain_from(resp: dict) -> dict:
    ai_p = float(resp.get("calibrated_prob", 0.5))
    drift = (resp.get("semantic_drift") or {})
    fp = (resp.get("llm_fingerprint") or {})
    pd_j = float(resp.get("pd_overlap_j", 0.0))
    cat = (resp.get("category") or "other").replace("_", " ")

    bullets: List[str] = []
    notes: List[str] = []
    fixes: List[str] = []

    if ai_p < 0.35:
        headline = "This looks human and original."
    elif ai_p <= 0.65:
        headline = "Mixed signals — not clearly AI or human."
    else:
        headline = "This likely contains AI-generated writing."

    if fp.get("available"):
        nf = fp.get("nearest_family")
        human_score = fp.get("human_score")
        if nf == "human_baseline" or (human_score is not None and float(human_score) >= 0.7):
            bullets.append("Your overall style matches typical human variation.")
        else:
            bullets.append(f"Closest style match: {nf} (not definitive).")

    if drift.get("available"):
        sc = float(drift.get("score", 0.5))
        if 0.45 <= sc <= 0.65:
            bullets.append("Topic flow and tone shifts look natural for human writing.")
        elif sc < 0.35:
            bullets.append("Paragraphs are unusually uniform (possible model pattern).")
            fixes.append("Vary sentence length and vocabulary between paragraphs.")
        else:
            bullets.append("Paragraphs shift topics a lot (could be human brainstorming).")

    if pd_j >= 0.12:
        notes.append("We detected overlap with public-domain phrasing; we limited the AI score to avoid a false positive.")
    else:
        notes.append("No meaningful match with public-domain phrasing.")

    if ai_p > 0.65:
        fixes.extend(
            [
                "Add brief personal specifics (dates, places, unique details).",
                "Break up uniform sentences; mix short and long forms.",
                "Include a brief anecdote or reflection in your own voice.",
            ]
        )
    elif 0.35 <= ai_p <= 0.65:
        fixes.append("If needed, add a few personal details or examples to strengthen human signal.")

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
        "teacher_report": teacher,
    }


# =============================================================================
# Main route: /scan
# =============================================================================

@app.post("/scan")
def scan(inp: ScanIn):
    global LAST_SCAN_STYLO, LAST_SCAN_META

    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail=f"Model unavailable: {model_load_error}")

    mode = (inp.mode or SETTINGS.mode).strip().title()
    if mode not in ("Balanced", "Strict", "Academic"):
        mode = "Balanced"

    short_cap_on = bool(SETTINGS.short_cap)
    min_tokens_strong = int(SETTINGS.min_tokens_strong)
    non_en_cap = float(SETTINGS.non_en_cap)
    en_thresh = float(SETTINGS.en_thresh)
    max_conf_unstable = float(SETTINGS.max_conf_unstable)
    max_conf_short = float(SETTINGS.max_conf_short)
    abstain_low = float(SETTINGS.abstain_low)
    abstain_high = float(SETTINGS.abstain_high)
    use_ensemble = bool(SETTINGS.use_ensemble and SECONDARY_READY)

    sens_boost, abstain_delta, artifact_bias = _mode_multipliers(mode)

    text = (inp.text or "").strip()
    if not text:
        return {"overall_score": 0.0, "per_token": [], "explanation": "Empty text"}

    out1 = chunked_scan_primary(text)
    total = max(1, int(out1["total"]))
    bins = out1["bins"]
    frac10 = float(bins[10]) / total
    frac100 = float(bins[100]) / total

    fPpl = _clamp((25 - float(out1["ppl"] or 25)) / 20, 0, 1)
    fBurst = _clamp((8 - float(out1["burstiness"] or 8)) / 6, 0, 1)

    # Reliability
    ci10 = _binom_ci_halfwidth(frac10, total)
    len_factor = min(1.0, total / float(max(1, min_tokens_strong)))
    shape_factor = 1.0 - min(0.6, ci10 * 1.8)
    reliability = max(0.15, len_factor * shape_factor)

    token_logps = [math.log(max(float(t.get("p", 0.0)), 1e-12)) for t in out1["per_token"]]
    stylo = stylometric_fingerprint(text, token_logps)
    sigs = prose_classic_signals(text)
    cscore = classic_style_score(text, stylo, sigs)

    fp = compute_llm_fingerprint(text, out1, stylo, sigs)
    drift = compute_semantic_drift(text)

    # Keep scratch state
    LAST_SCAN_STYLO = stylo
    LAST_SCAN_META = {
        "total_tokens": total,
        "frac10": frac10,
        "frac100": frac100,
        "ppl": float(out1["ppl"]),
        "burstiness": float(out1["burstiness"]),
    }

    # Primary calibration
    z = sens_boost * ((2.0 * float(out1["score"])) + (0.9 * fPpl) + (0.8 * fBurst) + (0.35 * frac10) - 1.6)
    calP1 = _clamp(1 / (1 + math.exp(-4 * z)), 0.0, 1.0)

    # Optional secondary
    calP2 = None
    out2 = None
    if use_ensemble:
        out2 = chunked_scan_secondary(text)
        if out2:
            total2 = max(1, int(out2["total"]))
            bins2 = out2["bins"]
            frac10_2 = float(bins2[10]) / total2
            fPpl2 = _clamp((25 - float(out2["ppl"] or 25)) / 20, 0, 1)
            fBurst2 = _clamp((8 - float(out2["burstiness"] or 8)) / 6, 0, 1)
            z2 = sens_boost * ((2.0 * float(out2["score"])) + (0.9 * fPpl2) + (0.8 * fBurst2) + (0.35 * frac10_2) - 1.6)
            calP2 = _clamp(1 / (1 + math.exp(-4 * z2)), 0.0, 1.0)

    calP = _combine_models(calP1, calP2, use_ensemble)

    # Artifact gate
    machine_artifact = (
        frac10 >= (0.985 * artifact_bias)
        and frac100 <= 0.03
        and float(out1["ppl"]) <= (2.5 / artifact_bias)
        and float(out1["burstiness"]) <= (3.5 / artifact_bias)
    )

    # Classic style decision
    looks_classic = (
        cscore >= 0.62
        and 6.0 <= float(out1["ppl"] or 99.0) <= 28.0
        and float(out1["burstiness"] or 99.0) <= 12.0
        and frac10 >= 0.45
        and not machine_artifact
    )

    if looks_classic:
        cat = "classic_literature"
        cat_conf = 0.88 if cscore >= 0.72 else 0.78
        note = category_note_for_report(True, cscore)
        calP = min(calP, 0.15 if cscore >= 0.72 else 0.22)
    else:
        cat = "other"
        cat_conf = 0.5
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
        note = (
            "Nonsense/Surreal verse safeguard — "
            f"rhyme={nonsense['signals']['rhyme_density']}, "
            f"meter_cv={nonsense['signals']['meter_cv']}, "
            f"invented={nonsense['signals']['invented_ratio']}, "
            f"lex_hits={nonsense['signals']['lex_hits']}, "
            f"sem_disc={nonsense['signals']['semantic_disc']}"
        )

    # Short excerpt attenuation
    if short_cap_on and total < min_tokens_strong and not machine_artifact:
        scale = max(0.25, total / float(max(1, min_tokens_strong)))
        calP = min(calP * scale, max_conf_short)
        note += f" | short-excerpt cap: {total} tokens"

    # Bootstrap instability shrink/cap
    instab = _bootstrap_instability(out1["per_token"], token_logps)
    if instab > 0.20 and not machine_artifact:
        calP *= 1.0 / (1.0 + 2.0 * instab)
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
    low = max(0.0, abstain_low + abstain_delta)
    high = min(1.0, abstain_high - abstain_delta)
    if low > high:
        low, high = high, low

    verdict = (
        "Likely human-written"
        if calP < low
        else "Inconclusive — human & model signals mixed"
        if calP <= high
        else "Likely AI-generated"
    )
    thermometer_blocks = int(round(calP * 10))
    thermo = "▇" * thermometer_blocks + "▒" * (10 - thermometer_blocks)

    row = {
        "ts": int(time.time()),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text_len_chars": len(text),
        "text_len_tokens": total,
        "model_name": MODEL_NAME + (f" + {SECOND_MODEL}" if (use_ensemble and calP2 is not None) else ""),
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "overall_score": round(float(out1["score"]), 6),
        "top10_frac": round(frac10, 6),
        "top100_frac": round(frac100, 6),
        "ppl": round(float(out1["ppl"]), 6),
        "burstiness": round(float(out1["burstiness"]), 6),
        "ai_likelihood_calibrated": round(float(calP), 6),
        "tag": (inp.tag or ""),
        "category": cat,
        "category_conf": round(float(cat_conf), 3),
        "category_note": note + f" | mode={mode}",
    }
    log_scan_row(row)

    percent = round(calP * 100)
    exp = (
        f"Confidence {percent}% [{thermo}] — {verdict}. "
        f"{round(100 * frac10)}% tokens in Top-10; {round(100 * frac100)}% in Top-100. "
        f"PPL≈{float(out1['ppl']):.1f}; Burst≈{float(out1['burstiness']):.3f}. "
        f"Detected: {cat.replace('_', ' ')} (conf≈{cat_conf:.0%}). {note}"
    )

    resp = {
        "overall_score": float(out1["score"]),
        "per_token": out1["per_token"],
        "explanation": exp,
        "ppl": float(out1["ppl"]),
        "burstiness": float(out1["burstiness"]),
        "bins": bins,
        "total": total,
        "model_name": row["model_name"],
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "calibrated_prob": float(calP),
        "category": cat,
        "category_conf": float(cat_conf),
        "category_note": note,
        "stylometry": stylo,
        "nonsense_signals": nonsense["signals"],
        "mode": mode,
        "verdict": verdict,
        "thermometer": thermo,
        "pd_overlap_j": round(float(pd_score), 4),
        "llm_fingerprint": fp,
        "semantic_drift": drift,
    }

    if out2:
        resp["secondary"] = {
            "ppl": float(out2["ppl"]),
            "burstiness": float(out2["burstiness"]),
            "total": int(out2["total"]),
            "score": float(out2["score"]),
        }

    try:
        resp["explain"] = _explain_from(resp)
    except Exception:
        resp["explain"] = {
            "headline": "Summary unavailable",
            "band": _label_band(resp.get("calibrated_prob", 0.5)),
            "why": [],
            "notes": ["Explain mode encountered an error."],
            "what_to_fix": [],
        }

     # --- Add report URL for mobile UI (so it can open the generated PDF)
    try:
        # match the report naming you already generate
        report_id = time.strftime("scan_summary_%Y-%m-%d_%H-%M-%S", time.localtime(row["ts"]))
        pdf_name = f"{report_id}.pdf"

        # only expose the URL if the file exists
        pdf_path = os.path.join(str(REPORTS_DIR), pdf_name)
        if os.path.exists(pdf_path):
            resp["report_file"] = Path(pdf_path).name
            resp["report_url"] = f"/reports/{resp['report_file']}"
    except Exception:
        pass

    return resp

@app.post("/scan/file")
async def scan_file(
    file: UploadFile = File(...),
    mode: str = Form("Balanced"),
    tag: str = Form(""),
):
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    # Basic size guard (adjust as you like)
    MAX_BYTES = 10 * 1024 * 1024  # 10 MB
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    text = ""

    if ext == ".txt":
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")

    elif ext == ".pdf":
        if PdfReader is None:
            raise HTTPException(status_code=500, detail="PDF support not installed (pypdf missing)")
        try:
            reader = PdfReader(io.BytesIO(raw))
            parts = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t.strip():
                    parts.append(t)
            text = "\n\n".join(parts)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")

    elif ext == ".docx":
        if docx is None:
            raise HTTPException(status_code=500, detail="DOCX support not installed (python-docx missing)")
        try:
            d = docx.Document(io.BytesIO(raw))
            parts = [p.text for p in d.paragraphs if (p.text or "").strip()]
            text = "\n".join(parts)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read DOCX: {e}")

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext} (use .txt, .pdf, .docx)")

    # Normalize
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        raise HTTPException(status_code=400, detail="No readable text extracted from file")

    # Reuse your existing scan() logic
    inp = ScanIn(text=text, mode=mode, tag=tag)
    resp = scan(inp)

    # Helpful metadata for UI
    resp["source_file"] = filename
    resp["source_ext"] = ext
    resp["source_chars"] = len(text)
    resp["_source_text"] = text

    return resp



# =============================================================================
# Live Typing Verification (Hybrid streaming)
# =============================================================================

LIVE_SESSIONS: Dict[str, Dict] = {}


def _norm(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    x = (x - lo) / (hi - lo)
    return max(0.0, min(1.0, x))


def _stylometry_vector(sty: Dict[str, float]) -> List[float]:
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
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 0.0
    db = math.sqrt(sum(y * y for y in b)) or 0.0
    if da == 0 or db == 0:
        return 0.0
    return max(0.0, min(1.0, num / (da * db)))


def _compare_reference_vs_sample(ref_text: str, live_text: str) -> Dict[str, float]:
    ref_scan = chunked_scan_primary(ref_text)
    if live_text.strip():
        live_scan = chunked_scan_primary(live_text)
    else:
        live_scan = {"total": 0, "ppl": 0.0, "burstiness": 0.0, "bins": {10: 0, 100: 0, 1000: 0, "rest": 0}, "per_token": []}

    ref_lps = [math.log(max(float(t.get("p", 0.0)), 1e-12)) for t in ref_scan.get("per_token", [])]
    live_lps = [math.log(max(float(t.get("p", 0.0)), 1e-12)) for t in live_scan.get("per_token", [])]

    ref_sty = stylometric_fingerprint(ref_text, ref_lps)
    live_sty = stylometric_fingerprint(live_text, live_lps)

    sim_sty = _cosine_similarity(_stylometry_vector(ref_sty), _stylometry_vector(live_sty))

    def _top_fracs(scan: Dict) -> Tuple[float, float]:
        total = max(1, int(scan.get("total", 0)))
        b = scan.get("bins", {10: 0, 100: 0, 1000: 0, "rest": 0})
        return float(b.get(10, 0)) / total, float(b.get(100, 0)) / total

    r_top10, r_top100 = _top_fracs(ref_scan)
    s_top10, s_top100 = _top_fracs(live_scan)

    ppl_ref, ppl_live = float(ref_scan.get("ppl", 0.0)), float(live_scan.get("ppl", 0.0))
    burst_ref, burst_live = float(ref_scan.get("burstiness", 0.0)), float(live_scan.get("burstiness", 0.0))

    ppl_sim = 1.0 - _norm(abs(ppl_ref - ppl_live), 0.0, 40.0)
    burst_sim = 1.0 - _norm(abs(burst_ref - burst_live), 0.0, 12.0)
    t10_sim = 1.0 - _norm(abs(r_top10 - s_top10), 0.0, 0.40)
    t100_sim = 1.0 - _norm(abs(r_top100 - s_top100), 0.0, 0.50)

    len_penalty = 1.0
    live_tokens = int(live_scan.get("total", 0))
    if live_tokens < 80:
        len_penalty = 0.85
    if live_tokens < 40:
        len_penalty = 0.70

    match = (
        0.45 * sim_sty
        + 0.18 * ppl_sim
        + 0.18 * burst_sim
        + 0.10 * t10_sim
        + 0.09 * t100_sim
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
        "live_tokens": live_tokens,
    }


class SampleStartIn(BaseModel):
    reference_text: str
    duration_sec: Optional[int] = 90


class SampleSubmitIn(BaseModel):
    session_id: str
    text_chunk: str
    done: Optional[bool] = False


class SampleFinalizeIn(BaseModel):
    session_id: str


@app.post("/auth/sample/start")
def auth_sample_start(inp: SampleStartIn):
    ref = (inp.reference_text or "").strip()
    if not ref:
        raise HTTPException(status_code=400, detail="reference_text is empty")
    sid = str(uuid.uuid4())
    LIVE_SESSIONS[sid] = {
        "ts": time.time(),
        "duration_sec": int(inp.duration_sec or 90),
        "reference": ref,
        "accum": [],
        "final": None,
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

    if inp.done:
        cmpd = _compare_reference_vs_sample(sess["reference"], live_text)
        sess["final"] = cmpd
        return {"ok": True, "final": True, "result": cmpd}

    if live_text:
        cmpd = _compare_reference_vs_sample(sess["reference"], live_text)
        prog = {"live_tokens": cmpd["live_tokens"], "rough_match_0_1": cmpd["match"]}
    else:
        prog = {"live_tokens": 0, "rough_match_0_1": 0.0}

    return {"ok": True, "final": False, "progress": prog}


@app.post("/auth/sample/finalize")
def auth_sample_finalize(inp: SampleFinalizeIn):
    sess = LIVE_SESSIONS.get(inp.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")

    live_text = " ".join(sess["accum"]).strip()
    cmpd = _compare_reference_vs_sample(sess["reference"], live_text) if live_text else {
        "sim_sty": 0,
        "sim_ppl": 0,
        "sim_burst": 0,
        "sim_top10": 0,
        "sim_top100": 0,
        "len_penalty": 0,
        "match": 0,
        "live_tokens": 0,
    }
    sess["final"] = cmpd

    m = float(cmpd["match"])
    if m >= 0.70 and int(cmpd["live_tokens"]) >= 80:
        verdict = "Strong match"
        note = "Live stylometry & rhythm align with the reference sample."
    elif m >= 0.50 and int(cmpd["live_tokens"]) >= 60:
        verdict = "Moderate match"
        note = "Core style signals align; consider longer sample to strengthen."
    else:
        verdict = "Weak/No match"
        note = "Insufficient alignment or sample too short to confirm."

    return {
        "session_id": inp.session_id,
        "match_score": round(m * 100),
        "verdict": verdict,
        "explanation": note,
        "details": cmpd,
    }


# =============================================================================
# Drift Diagnostics API
# =============================================================================

@app.post("/drift/analyze")
def drift_analyze(inp: DriftIn):
    t = (inp.text or "").strip()
    if not t:
        raise HTTPException(status_code=400, detail="text is empty")
    return compute_semantic_drift(t)


@app.post("/drift/compare")
def drift_compare(inp: DriftCompareIn):
    scan = (inp.scan_text or "").strip()
    live = (inp.live_text or "").strip()
    if not scan or not live:
        raise HTTPException(status_code=400, detail="scan_text and live_text are required")

    drift_scan = compute_semantic_drift(scan)
    drift_live = compute_semantic_drift(live)
    sim = _global_bow_sim(scan, live)

    paras = min(int(drift_scan.get("paragraphs", 0)), int(drift_live.get("paragraphs", 0)))
    risk_self = 0.5 * (float(drift_scan.get("risk", 0.5)) + float(drift_live.get("risk", 0.5)))
    risk_pair = 1.0 - sim
    risk = _clamp(0.5 * risk_self + 0.5 * risk_pair, 0.0, 1.0)

    return {
        "available": True,
        "paragraphs": paras,
        "similarity": round(sim, 3),
        "overlap": round(sim, 3),
        "risk": round(risk, 3),
        "score": round(1.0 - risk, 3),
    }


# =============================================================================
# Demo route
# =============================================================================

DEMO_TEXTS = [
    {
        "label": "Human — journal paragraph",
        "text": (
            "I missed my train this morning and walked the long way instead. "
            "The sidewalks were still wet from last night’s storm, and the maples "
            "had the sweet, sharp smell they get after the first cold snap. I kept "
            "rehearsing the question I should have asked yesterday and didn’t."
        ),
        "expect": "Likely Human",
    },
    {
        "label": "Classic-style prose (public-domain-like)",
        "text": (
            "“You mustn’t be uneasy,” said the gentleman, turning aside the curtain "
            "with a grave composure, “for the hour is not yet come; and if the wind "
            "keeps steady we shall have our answer before the bells have done.”"
        ),
        "expect": "Human w/ classic safeguard",
    },
    {
        "label": "AI — generic essay tone",
        "text": (
            "In conclusion, it is important to recognize that technology has both "
            "advantages and disadvantages. By thoughtfully balancing innovation and "
            "ethics, society can create a future that is inclusive and sustainable."
        ),
        "expect": "Likely AI",
    },
    {
        "label": "Nonsense-verse",
        "text": (
            "’Twas brillig, and the slithy toves did gyre and gimble in the wabe; "
            "all mimsy were the borogoves, and the mome raths outgrabe."
        ),
        "expect": "Nonsense guard → Human",
    },
    {
        "label": "AI-rewrite of human",
        "text": (
            "This week I explored an AI music tool after reading a study suggesting "
            "listeners often can’t tell AI compositions from human ones. I chose a song "
            "that matters to me and compared the generated piece to the original to "
            "evaluate structure, pacing, and emotional expression."
        ),
        "expect": "Inconclusive / Likely AI",
    },
]


@app.get("/demo")
def demo():
    sample = random.sample(DEMO_TEXTS, k=min(3, len(DEMO_TEXTS)))
    return JSONResponse(sample)


# =============================================================================
# Serve UI + misc
# =============================================================================

@app.get("/")
def root():
    return RedirectResponse(url="/ui")


@app.get("/version")
def version():
    return {
        "version": VERSION,
        "model": MODEL_NAME,
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "ensemble": bool(SETTINGS.use_ensemble and SECONDARY_READY),
        "secondary_model": (SECOND_MODEL if (SETTINGS.use_ensemble and SECONDARY_READY) else None),
        "mode": SETTINGS.mode,
        "fingerprint_centroids": len(MODEL_CENTROIDS),
        "paths": {
            "app_data": str(APP_DATA_DIR),
            "pd_fingerprints": str(PD_FINGERPRINT_DIR),
            "model_centroids": str(MODEL_CENTROID_DIR),
            "scan_log": str(LOG_PATH),
            "reports_dir": str(REPORTS_DIR),
        },
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/favicon.ico")
def favicon():
    # If you ship a favicon in /static, let the browser request it there;
    # this just avoids 404 noise.
    return Response(status_code=204)


def _run_dev_server():
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info"),
        reload=False,  # IMPORTANT inside packaged apps
    )

def _run_desktop():
    import threading
    import time
    import webview  # pywebview

    # IMPORTANT: this must be an .ico on Windows
    icon_path = str(resource_path("assets/copycat.ico"))

    def run_server():
        _run_dev_server()

    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(1.0)

    webview.create_window(
    "CopyCat",
    "http://127.0.0.1:8000",
    width=1100,
    height=780,
)

    webview.start()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except Exception:
        pass

    _run_desktop()
