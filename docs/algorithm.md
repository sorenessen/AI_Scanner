# CopyCat — Algorithm Overview (v0.3.3)

> High-level flow of the AI Text Scanner with PD fingerprint UX and Live Verification.

## System Overview (top-level)

```mermaid
flowchart LR
  A["Client UI"] -->|/scan POST| B["Scan Service"]
  A -->|/demo GET| C["Demo Loader"]
  A -->|/config GET/POST| D["Config Service"]
  A -->|/version GET| E["Version Service"]

  E -->|fingerprint_centroids| A
  D -->|settings| A
  C -->|sample texts| A
  B -->|metrics + explain + verdict| A


```markdown
## Scan Pipeline (compact, vertical)

```mermaid
flowchart TD
  B1["Normalize + Tokenize"]
  B2["Language Heuristics"]
  B3["Nonsense Guard (rhyme, meter CV, invented ratio, lex hits, semantic discontinuity)"]
  B4["Model Inference (top-k ranks, probs, Δlogp)"]
  B5["Stylometry (function words, hapax, sent mean/var, punct entropy)"]
  B6["Calibration & Bands (abstain window, caps)"]
  B7["Explain Builder (headline, why, how-to-fix, notes, teacher report)"]
  B8["Per-token Table + Raw JSON"]

  B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7 --> B8

  classDef step fill:#111,stroke:#bbb,color:#eee,rx:6,ry:6;
  class B1,B2,B3,B4,B5,B6,B7,B8 step;


```markdown
## Live Verification (UI only)

```mermaid
flowchart TD
  L1["User typing sample (90s)"] --> L2["Quick Stylometry (func words, hapax, sent stats, punct entropy)"]
  L2 --> L3["Cosine similarity vs reference (from last scan)"]
  classDef ui fill:#0f1a2b,stroke:#6ea8ff,color:#eaf2ff,rx:6,ry:6;
  class L1,L2,L3 ui;


### Legend / Notes
- **Scan Service** returns: `calibrated_prob`, `verdict`, `explain{band,headline,why,what_to_fix,notes,teacher_report}`, `stylometry`, `nonsense_signals`, `per_token`.
- **/version** should also return `fingerprint_centroids: N` (shows as “PD: N” in UI).
- Live Verification is advisory—doesn’t change the detector’s verdict.


