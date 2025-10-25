import re
import time
import requests
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup
import numpy as np

from sentence_transformers import SentenceTransformer


########################################
# CONFIG
########################################

# Light, fast sentence embedding model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# How big a "chunk" is when we slice source pages and essay text.
MAX_TOKENS_PER_CHUNK = 120

# How many “risky” sentences we’ll investigate from the essay
MAX_QUERIES = 6

# How many URLs to collect (per query) from the web search stub
URLS_PER_QUERY = 5


########################################
# SENTENCE / CHUNK HELPERS
########################################

def split_into_sentences(text: str) -> List[str]:
    """
    Naive sentence splitter. Good enough for now.
    """
    # Normalize weird whitespace first
    clean = re.sub(r"\s+", " ", text.strip())
    # Split on ., ?, ! followed by space/cap or end of string
    parts = re.split(r'(?<=[\.\?\!])\s+(?=[A-Z0-9"\'])', clean)
    # Fallback: if that fails or essay is super informal, just cut by period
    if len(parts) <= 1:
        parts = re.split(r'[\.\n]+', clean)
    # Strip and drop empties
    return [p.strip() for p in parts if p.strip()]


def pick_risky_sentences(essay: str) -> List[str]:
    """
    Heuristic ranker:
    - Long, polished, factual-sounding sentences are 'riskier'
    - We score and keep top MAX_QUERIES of them
    """

    sentences = split_into_sentences(essay)

    risky_scored: List[Tuple[float, str]] = []

    for sent in sentences:
        words = sent.split()
        length_score = min(len(words) / 20.0, 2.0)  # boost if it's long (>20 words)

        formal_score = 0.0
        # 'academic-y' markers
        if re.search(r"\b(according to|moreover|furthermore|in conclusion|it is widely believed|it is widely accepted|data suggest|studies show)\b", sent, flags=re.I):
            formal_score += 1.0
        # numbers / years
        if re.search(r"\b(19|20)\d{2}\b", sent):  # years like 1998, 2022
            formal_score += 0.5
        if re.search(r"\b\d+(\.\d+)?%|\b\d+(\.\d+)?\b", sent):  # stats / quantities
            formal_score += 0.5
        # proper nouns: Org / Gov / Agency style
        if re.search(r"\b(IPCC|UNESCO|World Health Organization|United Nations|U\.S\. Department|NASA|Harvard|MIT|Stanford)\b", sent):
            formal_score += 0.75

        # AI-smoothness hints: "however," "moreover," chained clauses
        comma_count = sent.count(",")
        if comma_count >= 2 and len(words) > 18:
            formal_score += 0.5

        # total heuristic risk score
        score = length_score + formal_score

        risky_scored.append((score, sent))

    # sort descending by score
    risky_scored.sort(key=lambda x: x[0], reverse=True)

    # take top N sentences
    chosen = [s for (_, s) in risky_scored[:MAX_QUERIES]]

    return chosen


def chunk_text_block(text: str, max_tokens=MAX_TOKENS_PER_CHUNK) -> List[str]:
    """
    Break freeform text into ~max_tokens-word chunks for embedding.
    """
    words = text.split()
    chunks = []
    cur = []
    for w in words:
        cur.append(w)
        if len(cur) >= max_tokens:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


########################################
# SEARCH + SCRAPE
########################################

def build_search_queries(risky_sentences: List[str]) -> List[str]:
    """
    Turn risky sentences into trimmed search queries.
    We remove quotes, parentheses, etc. to make them more friendly for web search.
    """
    queries = []
    for sent in risky_sentences:
        q = sent
        # strip citations / refs
        q = re.sub(r"\[[0-9]{1,3}\]", " ", q)  # [12]
        q = re.sub(r"\([^)]+\)", " ", q)      # (Smith 2020)
        # collapse whitespace
        q = re.sub(r"\s+", " ", q).strip()
        # trim long sentences for search (keep first ~25 words)
        words = q.split()
        q = " ".join(words[:25])
        queries.append(q)
    return queries


def stub_search_web_for(query: str, max_urls=URLS_PER_QUERY) -> List[str]:
    """
    PLACEHOLDER.
    This is where you'd call a real web search API (Bing, Brave, etc.).
    For now, we just return some generic public URLs so pipeline runs.

    In production:
    - Send `query` to your web search API
    - Parse top results (skip PDFs/login/paywall)
    - Return list of URLs
    """
    # For now: always include Wikipedia as fallback
    urls = [
        "https://en.wikipedia.org/wiki/Global_warming",
        "https://en.wikipedia.org/wiki/Carbon_dioxide",
        "https://en.wikipedia.org/wiki/Climate_change",
    ]
    # dedupe and cap
    out = []
    for u in urls:
        if u not in out:
            out.append(u)
        if len(out) >= max_urls:
            break
    return out


def fetch_url_text(url: str, timeout=8) -> str:
    """
    Download and clean visible text from a URL.
    """
    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "StudentSelfCheck/1.0 (+local)"},
        )
        r.raise_for_status()
    except Exception as e:
        print(f"[fetch] FAIL {url} -> {e}")
        return ""

    soup = BeautifulSoup(r.text, "html.parser")
    # remove junk
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    chunks = []
    for node in soup.find_all(["p", "li", "pre", "article", "section"]):
        txt = node.get_text(separator=" ", strip=True)
        if txt:
            chunks.append(txt)

    text = "\n\n".join(chunks)
    return text


def gather_candidate_corpus(urls: List[str]) -> List[Dict[str, str]]:
    """
    Returns list of {url, chunk} for all fetched pages.
    """
    corpus = []
    for url in urls:
        page_text = fetch_url_text(url)
        if not page_text:
            continue
        for piece in chunk_text_block(page_text, max_tokens=MAX_TOKENS_PER_CHUNK):
            corpus.append({
                "url": url,
                "chunk": piece,
            })
    return corpus


########################################
# EMBEDDING + MATCHING
########################################

def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Embed list of strings as normalized vectors (no multiprocessing).
    """
    embs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=8,
        show_progress_bar=False,
    )
    return embs.astype("float32")


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Cosine sim for normalized vectors = dot product.
    A: [m, d], B: [n, d] -> [m, n]
    """
    return np.dot(A, B.T)


def match_essay_to_corpus(essay_text: str, model: SentenceTransformer,
                          candidate_corpus: List[Dict[str, str]]) -> List[Dict]:
    """
    Compare each essay chunk to each source chunk and return ranked matches.
    """
    essay_chunks = chunk_text_block(essay_text, max_tokens=MAX_TOKENS_PER_CHUNK)
    if not essay_chunks:
        return []

    corpus_chunks = [c["chunk"] for c in candidate_corpus]
    if not corpus_chunks:
        return []

    essay_embs = embed_texts(essay_chunks, model)
    corpus_embs = embed_texts(corpus_chunks, model)

    sim = cosine_sim_matrix(essay_embs, corpus_embs)
    results = []

    for ei, row in enumerate(sim):
        # top 3 matches for each essay chunk
        top_idx = np.argsort(-row)[:3]
        for idx in top_idx:
            score = float(row[idx])
            results.append({
                "essay_chunk": essay_chunks[ei],
                "score": score,
                "source_chunk": corpus_chunks[idx],
                "url": candidate_corpus[idx]["url"],
            })

    # sort global matches high → low
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


########################################
# PUBLIC ENTRY POINT
########################################

def analyze_plagiarism(essay_text: str) -> Dict:
    """
    Master pipeline:
    1. Pick risky sentences
    2. Build search queries
    3. (Stub) get URLs for each query
    4. Fetch + chunk those URLs
    5. Embed + compare
    6. Build student-facing advice

    Returns a dict you can send back from FastAPI.
    """

    ts_start = time.time()

    risky_sentences = pick_risky_sentences(essay_text)
    queries = build_search_queries(risky_sentences)

    # For all queries, gather candidate URLs
    all_urls = []
    for q in queries:
        cand_urls = stub_search_web_for(q, max_urls=URLS_PER_QUERY)
        for u in cand_urls:
            if u not in all_urls:
                all_urls.append(u)

    # Download & chunk source pages
    candidate_corpus = gather_candidate_corpus(all_urls)

    # Load embedder model (we load it fresh here; you could also cache it globally)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    raw_matches = match_essay_to_corpus(
        essay_text=essay_text,
        model=model,
        candidate_corpus=candidate_corpus
    )

    # Post-process into per-sentence "risk report":
    # We'll map each risky sentence to its top similarity finding, if any.
    sentence_reports = []
    for sent in risky_sentences:
        # find best match in raw_matches where essay_chunk overlaps this sentence
        # (simple heuristic: check if the risky sentence is contained in that essay chunk)
        best = None
        for m in raw_matches:
            if sent[:80] in m["essay_chunk"]:  # crude containment check
                if (best is None) or (m["score"] > best["score"]):
                    best = m
        if best:
            sim_score = best["score"]
            advice = (
                "This passage is very close to an online source. You should cite or rewrite."
                if sim_score >= 0.5 else
                "This passage is somewhat similar to an online source. Consider citing."
                if sim_score >= 0.35 else
                "Low similarity. Probably fine."
            )
            sentence_reports.append({
                "sentence": sent,
                "similarity": round(sim_score, 3),
                "advice": advice,
                "match_url": best["url"],
                "match_excerpt": best["source_chunk"][:300],
            })
        else:
            sentence_reports.append({
                "sentence": sent,
                "similarity": 0.0,
                "advice": "No close public match found. You're likely fine.",
                "match_url": None,
                "match_excerpt": None,
            })

    total_time = time.time() - ts_start

    return {
        "summary": {
            "risky_sentence_count": len(risky_sentences),
            "unique_source_urls_checked": len(all_urls),
            "elapsed_sec": round(total_time, 2),
        },
        "per_sentence": sentence_reports,
        "debug": {
            "queries": queries,
            "urls": all_urls,
        }
    }
