FROM mcr.microsoft.com/playwright/python:v1.50.0-noble

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir '.[ml]'

COPY backend ./backend
COPY Tan Files ./Tan Files
COPY alembic.ini .

ENV PYTHONPATH=/app/backend
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
