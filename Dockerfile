FROM python:3.13-slim-bookworm

WORKDIR /app
COPY pyproject.toml .
COPY backend ./backend
RUN pip install --no-cache-dir '.[ml]' \
	&& playwright install --with-deps chromium

COPY ["Tan Files", "./Tan Files"]
COPY alembic.ini .

ENV PYTHONPATH=/app/backend
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
