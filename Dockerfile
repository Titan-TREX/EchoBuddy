# Single image used for both backend processes:
#   - web service:    uvicorn realtime.server:app --host 0.0.0.0 --port $PORT
#   - worker service: python -m temporal.worker
# Deploy the same image twice (e.g. two Railway services from this repo) with
# different start commands, since only one Dockerfile/CMD is needed to
# override per service.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# webrtcvad-wheels builds a C extension.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000
# Railway/Render inject $PORT at runtime; falls back to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn realtime.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
