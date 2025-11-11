# CopyCat — Algorithm Overview (v0.3.3)

High-level flow of the AI Text Scanner with PD fingerprint UX and Live Verification.

```mermaid
flowchart LR
  %% === LAYOUT CONTROL ===
  %% set graph direction and spacing
  %% (Mermaid ignores x/y, so we force grouping with subgraphs + invisible links)
  %% main services left → right
  %% pipeline = vertical, live verification = horizontal

  %% === CLIENT & SERVICES ===
  A["Client UI"]
  B["Scan Service"]
  C["Demo Loader"]
  D["Config Service"]
  E["Version Service"]

  %% connections between UI and backend services
  A -->|/scan POST| B
  A -->|/demo GET| C
  A -->|/config GET/POST| D
  A -->|/version GET| E

  %% responses back to UI
  E -->|fingerprint_centroids → PD badge| A
  D -->|settings → UI| A
  C -->|sample texts| A
  B -->|metrics + explain + verdict| A

  %% === SCAN PIPELINE ===
  subgraph PIPE["Scan Pipeline"]
    direction TB
    P1["Normalize + Tokenize"]
    P2["Language heuristics"]
    P3["Nonsense guard<br/>rhyme · meter CV · invented ratio · lex hits · semantic discontinuity"]
    P4["Model inference<br/>top-k ranks · probs · Δlogp"]
    P5["Stylometry<br/>function words · hapax · sent mean/var · punct entropy"]
    P6["Calibration & bands<br/>abstain window · caps"]
    P7["Explain builder<br/>headline · why · how-to-fix · notes · teacher report"]
    P8["Per-token table + raw JSON"]
    P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> P8
  end

  %% === LIVE VERIFICATION ===
  subgraph LIVE["Live Verification (UI only)"]
    direction LR
    L1["User typing sample (90 s)"]
    L2["Quick stylometry<br/>func words · hapax · sent stats · punct entropy"]
    L3["Cosine similarity vs reference<br/>(from last scan)"]
    L1 --> L2 --> L3
  end

  %% === WIRING BETWEEN BLOCKS ===
  A -. "start live sample" .-> L1
  B -->|reference stylometry for compare| L3
  P8 -.-> A
  A --> PIPE
  A -.-> LIVE

  %% === STYLING ===
  classDef pipeline fill:#fef9e7,stroke:#d6bb65,color:#000,rx:6,ry:6;
  classDef live fill:#e8f0ff,stroke:#6ea8ff,color:#000,rx:6,ry:6;
  classDef ui fill:#f3f3f3,stroke:#999,color:#111,rx:6,ry:6;

  class P1,P2,P3,P4,P5,P6,P7,P8 pipeline;
  class L1,L2,L3 live;
  class A,B,C,D,E ui;
