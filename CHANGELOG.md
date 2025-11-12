## v0.3.4 — Live Verification polish

**UI**
- Live sample HUD shows timer, token %, and rough match % while typing.
- “Start Live Sample (90s)” resets progress bar/text; finalization path is smoother.
- “Finalize” stays disabled until a live sample has started; auto-finalize fires after compute.

**Docs**
- Added a single-page **System Boundary** diagram (`docs/system_architecture.md`).
- Kept the **Algorithm Overview** diagram (`docs/algorithm.md`) and linked both from README.

**Tech**
- `/version` string bumped to `v0.3.4`.

# Changelog

## v0.3.0 — Fingerprint + Semantic Drift (2025-11-10)
- NEW: `semantic_drift` returned in `/scan` (avg_adjacent_sim, std_adjacent_sim, low_drops, risk, score).
- NEW: `llm_fingerprint` returned in `/scan` (nearest_family, similarity, distribution, confidence, human_score).
- NEW: `/fingerprint/reload` (POST) and `/fingerprint/centroids` (GET).
- DATA: Sample centroids in `./model_centroids/` (gpt4, llama, claude, human_baseline).
- OPS: `FPRINT_MIN_TOKENS` env var (default 180) to control min tokens for fingerprinting.
