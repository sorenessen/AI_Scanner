import os, re, textwrap, time
from typing import List, Tuple
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ------------------
# CONFIG
# ------------------
MODEL_NAME = "all-MiniLM-L6-v2"  # small & fast
MAX_TOKENS_PER_CHUNK = 120

seed_urls = [
    "https://en.wikipedia.org/wiki/Global_warming",
    # this BBC link 404s in the demo, we can keep it or drop it
    # "https://www.bbc.com/news/science-environment-1234",
]

# This is where you paste the essay you want to test
essay_text = """
it out at interest, and it fetched us a dollar a day apiece all the year
round—more than a body could tell what to do with. The Widow
Douglas she took me for her son, and allowed she would sivilize me;
but it was rough living in the house all the time, considering how dismal regular and decent the widow was in all her ways; and so when I
couldn’t stand it no longer I lit out. I got into my old rags and my
sugar-hogshead again, and was free and satisfied. But Tom Sawyer he
hunted me up and said he was going to start a band of robbers, and
I might join if I would go back to the widow and be respectable. So
I went back.
The widow she cried over me, and called me a poor lost lamb, and
she called me a lot of other names, too, but she never meant no harm
by it. She put me in them new clothes again, and I couldn’t do nothing but sweat and sweat, and feel all cramped up. Well, then, the old
thing commenced again. The widow rung a bell for supper, and you
had to come to time. When you got to the table you couldn’t go
right to eating, but you had to wait for the widow to tuck down her
head and grumble a little over the victuals, though there warn’t really
anything the matter with them,—that is, nothing only everything
was cooked by itself. In a barrel of odds and ends it is different;
things get mixed up, and the juice kind of swaps around, and the
things go better.
After supper she got out her book and learned me about Moses and
the Bulrushers, and I was in a sweat to find out all about him; but by
and by she let it out that Moses had been dead a considerable long
time; so then I didn’t care no more about him, because I don’t take
no stock in dead people.
Pretty soon I wanted to smoke, and asked the widow to let me. But
she wouldn’t. She said it was a mean practice and wasn’t clean, and I
must try to not do it any more. That is just the way with some people.
They get down on a thing when they don’t know nothing about it.
Here she was a-bothering about Moses, which was no kin to her, and
no use to anybody, being gone, you see, yet finding a power of fault
with me for doing a thing that had some good in it. And she took
snuff, too; of course that was all right, because she done it herself
"""


# ------------------
# HELPERS
# ------------------

def fetch_text(url: str, timeout=8) -> str:
    """Download and lightly clean visible text from a page."""
    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "AI-Scanner/1.0 (+local)"},
        )
        r.raise_for_status()
    except Exception as e:
        print(f"Fetch failed: {url} {e}")
        return ""

    soup = BeautifulSoup(r.text, "html.parser")

    # Strip scripts/styles/nav/etc
    for s in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        s.decompose()

    blocks = []
    for tag in soup.find_all(["p", "li", "pre", "article", "section"]):
        txt = tag.get_text(separator=" ", strip=True)
        if txt:
            blocks.append(txt)

    return "\n\n".join(blocks)


def chunk_text(text: str, max_tokens=MAX_TOKENS_PER_CHUNK) -> List[str]:
    """
    Naive chunker: split text by sentence-ish boundaries, then group into ~max_tokens-word chunks.
    """
    # split on sentence-like punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    cur_words = []
    cur_len = 0

    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        if cur_len + len(words) > max_tokens and cur_words:
            chunks.append(" ".join(cur_words))
            cur_words = []
            cur_len = 0
        cur_words.extend(words)
        cur_len += len(words)

    if cur_words:
        chunks.append(" ".join(cur_words))

    return chunks


def build_corpus_chunks(urls: List[str]) -> List[dict]:
    """
    Fetch all URLs, break into chunks, return list of {url, chunk}.
    """
    corpus_chunks = []
    for url in urls:
        page_txt = fetch_text(url)
        if not page_txt:
            continue
        chunks = chunk_text(page_txt, max_tokens=MAX_TOKENS_PER_CHUNK)
        for i, ch in enumerate(chunks):
            corpus_chunks.append({
                "url": url,
                "chunk": ch,
            })
    return corpus_chunks


def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Encode a list of strings into normalized embeddings (no multiprocessing).
    We explicitly disable parallel workers to avoid the macOS+Python3.13
    semaphore/multiprocessing crash.
    """
    # sentence-transformers uses model.encode(...). By default it *can*
    # spawn multiple workers. We'll force batch_size small and no multi-process.
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
    Compute cosine similarity between each row of A and each row of B.
    Assumes A and B are already L2-normalized embeddings (which we did).
    Result shape: [len(A), len(B)]
    """
    # cosine similarity of normalized vectors = dot product
    return np.dot(A, B.T)


def search_essay_against_corpus(essay_text: str, corpus_chunks: List[dict], model: SentenceTransformer, top_k=5):
    """
    For each essay chunk, find the most similar corpus chunks using cosine sim.
    Return a flat list of matches {score, essay_chunk, source_chunk, url}.
    """
    essay_chunks = chunk_text(essay_text, max_tokens=MAX_TOKENS_PER_CHUNK)

    if not essay_chunks:
        print("No essay chunks found. Did you paste text?")
        return []

    print(f"Essay split into {len(essay_chunks)} chunk(s).")

    # Embed essay + corpus
    corpus_texts = [c["chunk"] for c in corpus_chunks]
    essay_embs = embed_texts(essay_chunks, model)
    corpus_embs = embed_texts(corpus_texts, model)

    # Similarity matrix: [num_essay_chunks, num_corpus_chunks]
    sim_matrix = cosine_sim_matrix(essay_embs, corpus_embs)

    results = []
    for qi, row in enumerate(sim_matrix):
        # argsort descending
        top_idx = np.argsort(-row)[:top_k]
        for idx in top_idx:
            score = float(row[idx])
            results.append({
                "score": score,
                "essay_chunk": essay_chunks[qi],
                "source_chunk": corpus_chunks[idx]["chunk"],
                "url": corpus_chunks[idx]["url"],
            })

    # sort all matches across all chunks
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ------------------
# MAIN
# ------------------

if __name__ == "__main__":
    print("Loading embedding model (this may download weights the first time)...")
    model = SentenceTransformer(MODEL_NAME)

    print("Indexing corpus (this may take a bit)...")
    corpus_chunks = build_corpus_chunks(seed_urls)
    print(f"Indexed {len(corpus_chunks)} chunks.")

    print("Searching essay...")
    matches = search_essay_against_corpus(essay_text, corpus_chunks, model, top_k=5)

    # Pretty-print top matches
    for m in matches[:10]:
        print("\nSCORE {:.3f} - Source: {}".format(m["score"], m["url"]))
        print("EssaySnippet:\n{}\n".format(repr(m["essay_chunk"][:240])))
        print("SourceSnippet:\n{}\n".format(repr(m["source_chunk"][:240])))
