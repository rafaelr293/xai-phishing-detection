FROM python:3.13-slim-bookworm

WORKDIR /app
COPY pyproject.toml .
COPY backend ./backend
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir '.[visual]'

COPY ["Tan Files", "./Tan Files"]

ENV PYTHONPATH=/app/backend
EXPOSE 8010
CMD ["uvicorn", "visual_worker.server:app", "--host", "0.0.0.0", "--port", "8010"]
