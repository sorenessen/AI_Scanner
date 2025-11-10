# Changelog

## v0.3.0 — Fingerprint + Semantic Drift (2025-11-10)
- NEW: `semantic_drift` returned in `/scan` (avg_adjacent_sim, std_adjacent_sim, low_drops, risk, score).
- NEW: `llm_fingerprint` returned in `/scan` (nearest_family, similarity, distribution, confidence, human_score).
- NEW: `/fingerprint/reload` (POST) and `/fingerprint/centroids` (GET).
- DATA: Sample centroids in `./model_centroids/` (gpt4, llama, claude, human_baseline).
- OPS: `FPRINT_MIN_TOKENS` env var (default 180) to control min tokens for fingerprinting.
