# Single image used for both backend processes:
#   - web service:    uvicorn realtime.server:app --host 0.0.0.0 --port $PORT   (default CMD)
#   - worker service: python -m temporal.worker                                  (override the start command)
# Deploy the same image twice (e.g. two Railway services from this repo) with
# different start commands.
#
# State (SQLite, vector store, uploads, audio) lives in /data. Mount ONE persistent
# volume there, or it is lost on every redeploy.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# webrtcvad-wheels builds a C extension.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app /data \
    && chown app:app /app /data
USER app
WORKDIR /app

ENV PATH="/app/.venv/bin:${PATH}" \
    DATA_DIR=/data \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

COPY --chown=app:app pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=app:app . .
RUN uv sync --frozen --no-dev

# Download the embedding model at build time so the container starts without
# depending on Hugging Face (and document search is ready sooner).
RUN python -c "from config.settings import settings; from sentence_transformers import SentenceTransformer; SentenceTransformer(settings.embedding_model)"

VOLUME /data
EXPOSE 8000

# /health answers 503 if the database is unreachable. The long start period covers model load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"

# Railway/Render inject $PORT at runtime; falls back to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn realtime.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
