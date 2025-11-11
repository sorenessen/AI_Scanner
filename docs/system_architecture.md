# CopyCat / AI_Scanner — System Architecture Overview (v0.3.3)

1. High-level relational view showing how user actions propagate through the local client, backend service, and external APIs or runtime environments.

2. Logical boundary view for AI_Scanner showing user interaction, runtime layers, and data stores.

```mermaid
flowchart TB

%% actor outside the boundary
USER(["User"])

%% ---------- system boundary ----------
subgraph SYS["AI_Scanner - System Boundary"]
  direction TB

  %% local runtime / client machine
  subgraph L["Local Runtime (Laptop / Dev Machine)"]
    direction TB
    UI["Browser UI (HTML/JS)"]
    API["FastAPI Service (app.py)"]
    MOD["Python Modules (stylometry.py, pd_fingerprint.py)"]
    CFG[("config.json")]
    LOG[("server.log")]
  end

  %% server / cloud holdings (artifacts & repo)
  subgraph S["Server / Cloud"]
    direction TB
    MODEL[("Model Weights")]
    CENTROIDS[("model_centroids/")]
    REPO[("GitHub repo")]
  end

  %% external / optional services
  subgraph X["External / Optional Services"]
    direction TB
    LLM[("LLM APIs")]
  end
end

%% ---------- minimal flow ----------
USER --> UI
UI   --> API
API  --> MOD
MOD  --> MODEL
MODEL --> CENTROIDS
API  --> CFG
API  --> LOG
API  --> LLM
MODEL <--> REPO
```
