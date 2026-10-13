# xai-phishing-detection

This repository contains the static SOC dashboard and the FastAPI backend foundation for the explainable phishing detection platform.

---

## Getting Started

### Prerequisites

- Docker Desktop with Docker Compose
- Git
- Python 3.13 only if you want to run the static frontend server or backend tests locally

### 1. Clone and enter the repository

```bash
git clone https://github.com/rafaelr293/xai-phishing-detection.git
cd xai-phishing-detection
```

### 2. Add email files

Place `.eml` files in `data/inbox`. The scanner reads that directory from the API container. The repository includes model files under `Tan Files`; no `.env` file is required for the Docker Compose setup.

### 3. Start the backend stack

From the repository root:

```bash
docker compose up -d --build
```

This starts PostgreSQL, the FastAPI API, the isolated browser worker, and the PyTorch visual worker. The API container applies all Alembic migrations automatically before starting.

Check that the API is ready:

```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/readiness
```

Both endpoints should return `{"status":"ok"}` or `{"status":"ready"}`.

### 4. Start the dashboard

In a second terminal, from the repository root:

```bash
python3 -m http.server 8080
```

Open <http://localhost:8080>. The dashboard uses the API at <http://localhost:8000>.

### 5. Scan the inbox

Trigger an initial scan after placing `.eml` files in `data/inbox`:

```bash
curl -X POST http://localhost:8000/api/ingestion/scan
```

The same email content is not imported twice because the backend stores a content hash.

### 6. Rerun from scratch

The dashboard’s **Rerun analysis** button asks for confirmation. The endpoint permanently deletes stored emails, analyses, detector results, URL records, traversal jobs, ingestion jobs, and files under `data/artifacts/screenshots`, then imports and analyzes the current `.eml` files again:

```bash
curl -X POST http://localhost:8000/api/ingestion/rerun-analysis
```

Use this after changing detector or model behavior when stale database results must not remain.

### 7. Traverse a selected URL

The URL-risk detector marks phishing-like URLs as `selected_for_traversal`. Traversal remains manually triggered from the full analysis menu. The dashboard sends:

```text
POST /api/emails/{email_id}/urls/{url_id}/traverse
```

The browser worker follows the URL, records redirects and blocked requests, and captures a screenshot. Poll the returned traversal ID with:

```text
GET /api/emails/traversals/{traversal_id}
```

When traversal completes, the API automatically sends the screenshot to the visual worker. Visual results are persisted on that specific traversal and shown after the dashboard refreshes.

### Troubleshooting and shutdown

View service logs with:

```bash
docker compose logs -f api browser-worker visual-worker postgres
```

Stop containers while preserving the PostgreSQL volume:

```bash
docker compose down
```

Stop containers and delete the PostgreSQL volume as well:

```bash
docker compose down -v
```

For local backend tests, install the development dependencies with Python 3.13:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[test,ml]'
pytest -q
```

Attachment analysis is static only. It records hashes and metadata, checks risky extensions and filename patterns, compares safe file signatures, and lists ZIP archive members without extracting or executing attachments.

---
