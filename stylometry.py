# stylometry.py — lightweight stylometric features without NLTK

import re
import math

FUNC_WORDS = {
   "the","be","to","of","and","a","in","that","have","i","it","for","not","on","with",
   "he","as","you","do","at","this","but","his","by","from","they","we","say","her",
   "she","or","an","will","my","one","all","would","there","their","what","so","up",
   "out","if","about","who","get","which","go","me"
}

def stylometric_features(text: str, token_logps: list[float]):
    words = re.findall(r"[A-Za-z']+", text.lower())
    total_words = len(words)
    func_count = sum(1 for w in words if w in FUNC_WORDS)
    func_ratio = func_count / total_words if total_words > 0 else 0.0

    sentences = re.split(r"[.!?]+", text)
    sentence_lengths = [len(re.findall(r"[A-Za-z']+", s)) for s in sentences if s.strip()]
    avg_sent_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0

    mean_lp = sum(token_logps) / len(token_logps) if token_logps else 0.0
    burstiness = sum((lp - mean_lp) ** 2 for lp in token_logps) / len(token_logps) if token_logps else 0.0

    return {
        "func_word_ratio": round(func_ratio, 4),
        "avg_sent_len": round(avg_sent_len, 2),
        "burstiness": round(burstiness, 4),
    }
