# CopyCat / AI_Scanner — System Architecture Overview (v0.3.3)

High-level relational view showing how user actions propagate through the local client, backend service, and external APIs or runtime environments.

```mermaid
flowchart TB
  %% ---- LAYERS ----
  subgraph U["User Environment"]
    user[🧑 User]
    browser[🌐 Web Browser (index.html + JS)]
  end

  subgraph L["Local Runtime (Laptop / Dev Machine)"]
    apiLocal[⚙️ FastAPI Backend (app.py)]
    pyModules[📦 Python Modules (stylometry.py, pd_fingerprint.py, etc.)]
    pdDir[(./pd_fingerprints/)]
    configFile[(config.json)]
    logs[(server.log)]
  end

  subgraph S["Server / Cloud Environment"]
    ghRepo[💾 GitHub Repository (main branch)]
    modelSrv[🧠 Local Model Weights / AI Engine]
    storage[(Local/Remote Volume: ./model_centroids/)]
  end

  subgraph E["External / Optional Services"]
    extAPI[(3rd-party APIs: HuggingFace, OpenAI, etc.)]
    clientSync[(Git Pull / Push via CLI)]
  end

  %% ---- CONNECTIONS ----
  user --> browser
  browser -->|"HTTP POST /scan"| apiLocal
  browser -->|"GET /config, /version"| apiLocal
  apiLocal --> pyModules
  pyModules --> modelSrv
  pyModules --> pdDir
  apiLocal --> configFile
  apiLocal --> logs
  apiLocal --> ghRepo
  ghRepo <-->|git push/pull| clientSync
  modelSrv --> storage
  modelSrv --> extAPI
  browser -->|"renders Explain Panel + PD Badge"| user

  %% add to bottom of diagram
  classDef user fill:#f4f9ff,stroke:#1e70b8,color:#000,rx:8,ry:8;
  classDef local fill:#fff4e6,stroke:#ffb84d,rx:8,ry:8;
  classDef server fill:#eef8f1,stroke:#37965d,rx:8,ry:8;
  classDef ext fill:#f5f5f5,stroke:#999,rx:8,ry:8;

  class user,browser user;
  class apiLocal,pyModules,pdDir,configFile,logs local;
  class ghRepo,modelSrv,storage server;
  class extAPI,clientSync ext;

