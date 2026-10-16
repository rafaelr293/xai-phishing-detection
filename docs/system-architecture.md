# System Architecture

The platform is deployed as a static dashboard plus three containerized backend
services. The API owns ingestion, orchestration, risk aggregation, and
persistence. Browser traversal and visual analysis run in separate workers so
that URL browsing and image inference remain isolated from the main API.

```mermaid
flowchart TB
    analyst["SOC analyst"]
    dashboard["Static SOC dashboard<br/>HTML + CSS + vanilla JavaScript<br/>localhost:8080"]
    api["FastAPI API<br/>Uvicorn<br/>localhost:8000"]
    db[("PostgreSQL 16<br/>phishing_detection")]
    inbox[("data/inbox<br/>.eml files")]
    artifacts[("data/artifacts<br/>screenshots + heatmaps")]
    models[("Tan Files<br/>bundled into API and visual-worker images<br/>email, URL, and EfficientNet-B0 models")]
    browser["Browser worker<br/>FastAPI + Playwright + Chromium<br/>:8010"]
    visual["Visual worker<br/>FastAPI + PyTorch + Torchvision<br/>:8010"]
    target(("Public URL / website"))

    analyst -->|"reviews findings<br/>triggers scans and traversal"| dashboard
    dashboard -->|"REST / JSON<br/>GET and POST /api/*"| api
    api -->|"SQLAlchemy + psycopg<br/>read/write analyses and settings"| db
    api -->|"reads and hashes email files"| inbox
    api -->|"loads email and URL models"| models
    api -->|"writes and serves artifacts"| artifacts

    api -->|"POST /traverse<br/>manual URL traversal"| browser
    browser -->|"HTTPS navigation,<br/>redirects, blocked requests"| target
    browser -->|"stores screenshot and traversal metadata"| artifacts
    browser -->|"traversal result"| api

    api -->|"POST /predict<br/>completed screenshot"| visual
    visual -->|"loads EfficientNet-B0"| models
    visual -->|"reads screenshot"| artifacts
    visual -->|"writes heatmap and result"| artifacts
    visual -->|"visual detector result"| api

    subgraph compose["Docker Compose"]
        api
        db
        browser
        visual
    end

    subgraph shared["Mounted data and bundled assets"]
        inbox
        artifacts
        models
    end

    classDef client fill:#17202a,stroke:#56d6c9,color:#fff
    classDef service fill:#1f2937,stroke:#8ab4f8,color:#fff
    classDef storage fill:#27272a,stroke:#f0c674,color:#fff
    classDef external fill:#3b1f2b,stroke:#f7768e,color:#fff
    class analyst,dashboard client
    class api,browser,visual service
    class db,inbox,artifacts,models storage
    class target external
```

## Main analysis flow

1. The dashboard sends an ingestion request to the API.
2. The API reads `.eml` files from `data/inbox`, parses them, extracts URLs and
   attachments, and runs the configured email and URL detectors.
3. Detector results, explanations, risk weights, and email metadata are stored
   in PostgreSQL.
4. An analyst can manually request traversal for a selected URL. The API
   delegates browsing to the Playwright browser worker.
5. The browser worker follows redirects, records blocked requests, and captures
   a screenshot under `data/artifacts`.
6. The API sends the completed screenshot to the visual worker. The PyTorch
   worker produces visual analysis and a heatmap, which are stored as
   traversal-specific artifacts.
7. The dashboard polls the API and displays aggregate risk, evidence,
   screenshots, and visual explanations.

## Runtime boundaries

- `db-net` connects the API to PostgreSQL.
- `app-net` connects the API to the browser and visual workers.
- The browser worker is isolated from the public dashboard and is only exposed
  internally to the Compose network.
- PostgreSQL data is persisted in the `phishing-postgres-data` Docker volume.
- Inbox files, screenshots, and heatmaps are shared through the repository's
  mounted `data` directory. Model files from `Tan Files` are copied into the
  API and visual-worker images at build time.
