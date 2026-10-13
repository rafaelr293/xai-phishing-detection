# xai-phishing-detection

This repository contains the static SOC dashboard and the FastAPI backend foundation for the explainable phishing detection platform.

---

## Getting Started

### 1. Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone https://github.com/rafaelr293/xai-phishing-detection.git
```

### 2. Start PostgreSQL

Use Python 3.13 for local development. Install it with `pyenv` or the official Python installer, then create the environment with:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[test,ml]'
```

Install Docker Desktop, then run:

```bash
docker compose up -d postgres
```

### 3. Install backend dependencies

From the project root:

```bash
cp .env.example .env
```

### 4. Run the backend

```bash
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
uvicorn app.main:app --app-dir backend --reload
```

The API is available at `http://localhost:8000`. Check `http://localhost:8000/api/health` and trigger an inbox scan with:

```bash
curl -X POST http://localhost:8000/api/ingestion/scan
```

To rerun the configured detector modules against every already-ingested email whose source file is still available:

```bash
curl -X POST http://localhost:8000/api/ingestion/rerun-analysis
```

URL traversal runs in the isolated browser worker. Build and start it with `docker compose up -d --build api browser-worker`. A URL must first be marked `selected_for_traversal` by the URL-risk module before `POST /api/emails/{email_id}/urls/{url_id}/traverse` creates a background job. Poll `GET /api/emails/traversals/{traversal_id}` for completion. Redirects and blocked requests are stored in PostgreSQL; screenshots are stored under `data/artifacts/screenshots` and can be retrieved through the traversal screenshot endpoint.

Visual analysis runs in a separate PyTorch worker. Build it with `docker compose build visual-worker`; its internal health endpoint is `/health` and its prediction endpoint is `/predict`. The API-to-worker job integration is the next step; the worker is isolated now so PyTorch is not installed in the API image.

Put `.eml` files in `data/inbox` before scanning. The backend records content hashes so scanning the same file again does not create a duplicate email.

Attachment analysis is static only. It records hashes and metadata, checks risky extensions and filename patterns, compares safe file signatures, and lists ZIP archive members without extracting or executing attachments. Detector results are persisted with the email analysis record.

The Email/NLP adapter is available behind `ENABLE_EMAIL_NLP`. Enable it only in an environment matching the serialized model dependencies; the current teammate artifact was serialized with scikit-learn `1.9.0` and should be installed with `pip install -e '.[ml]'` under the project’s Python 3.13 environment.

Attachment confidence starts at `0.95` when attachments are present and decreases when verification checks fail, such as an unknown signature, unrecognized extension, or signature mismatch. It is capped at a minimum of `0.50` and is not a calibrated probability.

Unknown attachment types add a low-severity risk indicator and a small `15`-point contribution, while known malicious signatures and executable extensions receive substantially higher scores.

### 5. View the static frontend

You can open `index.html` directly in your browser:

- Double-click `index.html` in your file explorer, or
- Start a simple local server:
  ```bash
  python -m http.server 8080
  ```
  Then visit `http://localhost:8080` in your web browser.

---
