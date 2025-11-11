# CopyCat — Algorithm Overview (v0.3.3)

> High-level flow of the AI Text Scanner with PD fingerprint UX and Live Verification.

```mermaid
flowchart TD
  A[Client UI] -->|/scan POST| B[Scan Service]
  A -->|/demo GET| C[Demo Loader]
  A -->|/config GET/POST| D[Config Service]
  A -->|/version GET| E[Version Service]

  subgraph Bx[Scan Pipeline]
    B1[Normalize + Tokenize] --> B2[Language Heuristics]
    B2 --> B3[Nonsense Guard\n(rhyme, meter CV, invented ratio,\nlex hits, semantic discontinuity)]
    B3 --> B4[Model Inference\n(top-k ranks, probs, logp deltas)]
    B4 --> B5[Stylometry\n(function words, hapax, sent mean/var,\npunct entropy)]
    B5 --> B6[Calibration + Bands\n(abstain window, caps)]
    B6 --> B7[Explain Builder\n(headline, why, how-to-fix, notes,\nteacher_report)]
    B7 --> B8[Per-token Table + Raw JSON]
  end

  E -->|fingerprint_centroids| A
  D -->|settings| A
  B -->|metrics + explain + verdict| A
  C -->|sample texts| A

  subgraph L[Live Verification (UI)]
    L1[User typing sample] --> L2[Quick Stylometry]
    L2 --> L3[Cosine Similarity vs. Reference\n(from last scan)]
  end
