# CopyCat / AI_Scanner — System Architecture Overview (v0.3.3)

High-level relational view showing how user actions propagate through the local client, backend service, and external APIs or runtime environments.

```mermaid
flowchart TB

%% externals (outside the boundary)
USER(["User"])
LLM(["LLM APIs"])
GIT(["GitHub"])

%% system boundary
subgraph SYS["AI_Scanner - System Boundary"]
  direction TB

  %% client
  C_UI["Browser UI (HTML/JS)"]

  %% app/runtime
  C_API["FastAPI Service (app.py)"]
  C_MOD["Python Modules (stylometry.py, pd_fingerprint.py)"]
  C_CFG[("config.json")]
  C_LOG[("server.log")]

  %% model/storage
  C_MODEL[("Model Weights")]
  C_CENT[("model_centroids/")]
end

%% minimal relationships
USER --> C_UI
C_UI --> C_API
C_API --> C_MOD
C_MOD --> C_MODEL
C_MODEL --> C_CENT
C_API --> C_CFG
C_API --> C_LOG
C_API --> LLM
C_MODEL --> GIT
GIT --> C_MODEL

```
