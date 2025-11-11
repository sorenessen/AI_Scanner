# CopyCat / AI_Scanner — System Architecture Overview (v0.3.3)

High-level relational view showing how user actions propagate through the local client, backend service, and external APIs or runtime environments.

```mermaid
flowchart LR

%% ---------- ACTORS ----------
subgraph ACTORS["Actors"]
  U[End user]
  OP[Operator / Admin]
end

%% ---------- CLIENT ----------
subgraph CLIENT["Client (user machine)"]
  UI[Browser UI\nindex.html + JS]
end

%% ---------- ACCESS / EDGE (optional) ----------
subgraph EDGE["Access / Edge"]
  LB[Reverse proxy / load balancer]
end

%% ---------- APPLICATION SERVICE ----------
subgraph APP["Application service"]
  API[FastAPI service\napp.py]
  MODS[Modules\nstylometry.py\npd_fingerprint.py]
  CFG[(config.json)]
  LOG[(server.log)]
end

%% ---------- MODEL / RUNTIME ----------
subgraph MODEL["Model / AI engine"]
  WEIGHTS[(Model weights)]
  PD[(./pd_fingerprints/)]
end

%% ---------- STORAGE ----------
subgraph STORE["Storage"]
  CENT[(./model_centroids/)]
end

%% ---------- EXTERNAL ----------
subgraph EXT["External services"]
  LLM[(HuggingFace / OpenAI)]
  GIT[(GitHub repository\nmain branch)]
end

%% ---------- FLOWS ----------
U  -- "HTTP GET/POST" --> UI
UI -- "/scan, /config, /version" --> LB
LB --> API

API --> MODS
MODS --> WEIGHTS
MODS --> PD
WEIGHTS --> CENT
CENT --> API

API --> LOG
API --> CFG

API --> UI
UI  -- "renders Explain + PD badge" --> U

OP  -- "git push/pull" --> GIT
GIT --> WEIGHTS

API --> LLM


```
