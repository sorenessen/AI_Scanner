# CopyCat — Algorithm Overview (v0.3.3)

> High-level flow of the AI Text Scanner with PD fingerprint UX and Live Verification.

```mermaid
flowchart TD
    A["Client UI"] -->|"/scan POST"| B["Scan Service"]
    A -->|"/demo GET"| C["Demo Loader"]
    A -->|"/config GET/POST"| D["Config Service"]
    A -->|"/version GET"| E["Version Service"]

    subgraph "Scan Pipeline"
        B1["Normalize + Tokenize"] --> B2["Language Heuristics"]
        B2 --> B3["Nonsense Guard (rhyme, meter CV, invented ratio, lex hits, semantic discontinuity)"]
        B3 --> B4["Model Inference (top-k ranks, probs, logp deltas)"]
        B4 --> B5["Stylometry (function words, hapax, sent mean/var, punct entropy)"]
        B5 --> B6["Calibration + Bands (abstain window, caps)"]
        B6 --> B7["Explain Builder (headline, why, how-to-fix, notes, teacher report)"]
        B7 --> B8["Per-token Table + Raw JSON"]
    end

    E -->|fingerprint_centroids| A
    D -->|settings| A
    B -->|metrics + explain + verdict| A
    C -->|sample texts| A

    subgraph "Live Verification (UI)"
        L1["User typing sample"] --> L2["Quick Stylometry"]
        L2 --> L3["Cosine Similarity vs. Reference (from last scan)"]
    end

