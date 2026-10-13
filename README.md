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

Install Docker Desktop, then run:

```bash
docker compose up -d postgres
```

### 3. Install backend dependencies

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
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

Put `.eml` files in `data/inbox` before scanning. The backend records content hashes so scanning the same file again does not create a duplicate email.

Attachment analysis is static only. It records hashes and metadata, checks risky extensions and filename patterns, compares safe file signatures, and lists ZIP archive members without extracting or executing attachments. Detector results are persisted with the email analysis record.

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
