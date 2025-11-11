# CopyCat / AI_Scanner — Boundary Diagram (v0.3.3)

_A single, readable boundary view showing what’s inside the product and the minimal data path._

```mermaid
%% Plain Mermaid for GitHub; no custom colors or classes
flowchart TB

%% Actor outside the boundary
USER[(User)]

%% ---------- System Boundary ----------
subgraph SYS[AI_Scanner — System Boundary]
  direction TB

  %% Local runtime / client machine
  subgraph L[Local Runtime (Laptop / Dev Machine)]
    direction TB
    UI[Browser UI — index.html + JS]
    API[FastAPI Service — app.py]
    MOD[Python Modules — stylometry.py / pd_fingerprint.py]
    CFG[(config.json)]
    LOG[(server.log)]
  end

  %% Server / cloud holdings (artifacts & repo)
  subgraph S[Server / Cloud]
    direction TB
    MODEL[(Model Weights)]
    CENTROIDS[(model_centroids/)]
    REPO[(GitHub repo)]
  end

  %% External/optional services
  subgraph X[External / Optional Services]
    direction TB
    LLM[(LLM APIs)]
  end
end
%% ---------- Minimal flow ----------
USER --> UI
UI   --> API
API  --> MOD
MOD  --> MODEL
MODEL --> CENTROIDS
API  --> CFG
API  --> LOG
API  --> LLM
MODEL <--> REPO
