# CopyCat — Algorithm Overview (v0.3.3)

High-level flow of the AI Text Scanner with PD fingerprint UX and Live Verification (single diagram).

```mermaid
flowchart LR
  %% ---- Top-level services ----
  A["Client UI"]
  B["Scan Service"]
  C["Demo Loader"]
  D["Config Service"]
  E["Version Service"]

  A -->|/scan POST| B
  A -->|/demo GET| C
  A -->|/config GET/POST| D
  A -->|/version GET| E

  E -->|fingerprint_centroids| A
  D -->|settings| A
  C -->|sample texts| A
  B -->|metrics + explain + verdict| A

  %% ---- Scan Pipeline (vertical) ----
  subgraph SP["Scan Pipeline"]
    direction TB
    B1["Normalize + Tokenize"]
    B2["Language heuristics"]
    B3["Nonsense guard<br/>rhyme · meter CV · invented ratio · lex hits · semantic discontinuity"]
    B4["Model inference<br/>top-k ranks · probs · Δlogp"]
    B5["Stylometry<br/>function words · hapax · sent mean/var · punct entropy"]
    B6["Calibration & bands<br/>abstain window · caps"]
    B7["Explain builder<br/>headline · why · how-to-fix · notes · teacher report"]
    B8["Per-token table + raw JSON"]
    B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7 --> B8
  end

  %% ---- Live Verification (UI only) ----
  subgraph LV["Live Verification (UI only)"]
    direction TB
    L1["User typing sample (90s)"]
    L2["Quick stylometry<br/>func words · hapax · sent stats · punct entropy"]
    L3["Cosine similarity vs reference<br/>(from last scan)"]
    L1 --> L2 --> L3
  end

  %% ---- Light wiring between UI and LV/SP ----
  A -. "start live sample" .-> L1
  B -->|reference stylometry for compare| L3
