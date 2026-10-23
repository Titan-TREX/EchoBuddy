# EchoBuddy — Voice-to-Voice AI Chatbot

A voice chatbot with two pipelines behind one UI:

- **Realtime** (WebSocket): live mic streaming, server-side voice-activity detection, streamed text and audio replies, tool calling and document search.
- **Batch** (Temporal.io): record a clip, get a full reply; fault-tolerant, with retries.
- **Text chat**: type a message, get a text and spoken reply.

**Stack:** Deepgram (speech-to-text) · Groq / OpenRouter (LLM) · Cartesia (text-to-speech) · FastAPI · Temporal · SQLite · ChromaDB (RAG) · Langfuse (tracing).

## Running locally

```bash
cp .env.example .env            # then fill in the API keys
uv sync

docker compose up -d            # Terminal 1: Temporal server (batch mode only)
uv run python -m temporal.worker                          # Terminal 2: Temporal worker (batch mode only)
uv run uvicorn realtime.server:app --reload --port 8000   # Terminal 3: API + UI

# open http://localhost:8000
```

The realtime and text-chat modes need only the API server. Temporal is required just for the 🎤 batch mode.

## Configuration — nothing is hardcoded

[`config/settings.py`](config/settings.py) is the single source of truth. Every value below is a field there and can be overridden by an environment variable of the same name in `UPPER_CASE` (see [`.env.example`](.env.example)). Long prompts live in [`config/prompts.py`](config/prompts.py).

| Area | Examples |
|------|----------|
| Credentials | `DEEPGRAM_API_KEY`, `GROQ_API_KEY`, `CARTESIA_API_KEY`, `OPENROUTER_API_KEY`, `LANGFUSE_*` |
| LLM | `GROQ_MODEL`, `OPENROUTER_MODEL`, `SYSTEM_PROMPT`, `HISTORY_MAX_TURNS`, `LLM_*_TEMPERATURE`, `FORCE_TOOL_KEYWORDS` |
| Speech | `DEEPGRAM_MODEL`, `CARTESIA_MODEL`, `CARTESIA_VOICE_ID`, `TTS_SAMPLE_RATE` |
| Voice detection | `VAD_AGGRESSIVENESS`, `VAD_SILENCE_TIMEOUT_MS`, `VAD_MIN_SPEECH_RMS` |
| Tools | `WEB_SEARCH_MAX_RESULTS`, `WEATHER_URL_TEMPLATE`, `CALCULATOR_MAX_EXPONENT` |
| Documents (RAG) | `EMBEDDING_MODEL`, `RAG_CHUNK_SIZE`, `RAG_TOP_K`, `RAG_MAX_DISTANCE`, `MAX_UPLOAD_MB` |
| Server | `LOG_LEVEL`, `CORS_ORIGINS`, `ALLOWED_AUDIO_EXTENSIONS` |
| Access control | `API_TOKEN`, `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`, `TRUST_PROXY_HEADERS` |
| Temporal | `TEMPORAL_ADDRESS`, `TEMPORAL_TASK_QUEUE`, `ACTIVITY_*` retry and timeout tuning |

The old `GEMINI_*` variable names are still accepted as aliases for the Groq settings.

## Architecture

```
config/                      Settings (settings.py), prompts (prompts.py), Langfuse setup (tracing.py)
core/                        Logic shared by BOTH pipelines
├── prompting.py             system prompt + the user's document excerpts + history, tool-forcing heuristic
├── tool_loop.py             the two tool-calling steps (decide, run) against any OpenAI-style client
├── tools.py                 tool registry: web search, weather, calculator, date/time
└── assets.py                `assets/<name>` audio references (no filesystem paths leak into API/DB)
database/
├── connection.py            async SQLite connection + schema bootstrap
├── repository.py            the ONLY place that issues SQL (used by both pipelines)
└── schema.sql               users / sessions / messages
realtime/                    Realtime + text chat
├── server.py                FastAPI app setup (middleware, mounts, startup, background tasks)
├── routes/
│   ├── chat.py              POST /api/voice-chat (Temporal), POST /api/text-chat
│   ├── ws.py                WS /ws (realtime voice)
│   ├── documents.py         /api/upload-doc, /api/docs   (per-user)
│   ├── auth.py · health.py  /api/auth/check, /health
├── security.py              access token, rate limiting, user-id validation
├── session.py               per-connection turn-taking (VAD → STT → LLM → TTS)
├── llm_stream.py            streaming reply (Groq) on top of core/
├── stt.py · tts_stream.py   Deepgram / Cartesia
├── audio.py · cleanup.py    PCM/WAV helpers · old-audio cleanup
rag/                         ingestor.py (PDF/TXT → ChromaDB, per owner), retriever.py (semantic search)
temporal/                    Batch pipeline
├── workflows.py             VoiceChatbotWorkflow (sequencing only, no I/O)
├── worker.py · dto.py
└── activities/              one external dependency per file (stt, llm via OpenRouter, tts, db)
static/                      the UI: index.html, plus build.mjs + vercel.json for the Vercel deploy
tests/                       pytest suite · tests/harness: live end-to-end, load and LLM-quality checks
```

Dependencies point inward: routes / activities → `core` → `config` / `database` / `rag`. Both pipelines share `core/`, so they behave the same (tools, document search, prompts). Nothing outside `database/repository.py` issues SQL, and no tunable is hardcoded.

## Security

- **Access token:** set `API_TOKEN` and every API route (and the WebSocket) requires it. The page itself stays public; the UI asks for the token the first time the server answers 401 and remembers it in that browser. Leave it empty for an open demo.
- **Rate limiting:** always on, per client (30 requests/minute by default; uploads count extra). Behind Railway/Render set `TRUST_PROXY_HEADERS=true` so the limit applies per real client IP, not per proxy. The limiter is in-memory, so it is per server process.
- **Per-user documents:** each browser keeps a long random id (`localStorage`) and sends it as `X-User-Id` (and in the WebSocket start message). Uploaded documents belong to that id: search, listing and deletion only ever see the caller's own files, which are also stored in a per-user folder. Clearing browser storage creates a new identity, so previously uploaded documents are no longer reachable from that browser. Documents ingested before this feature have no owner and are never returned.
- **Audio cleanup:** recordings and spoken replies in `assets/` are deleted after `ASSETS_RETENTION_HOURS` (default 24; 0 keeps them). Only the app's own `usr_*`/`bot_*` audio files are ever touched; chat history text is kept.
- Uploads are restricted to allowed types, size-capped, and saved under sanitised names.

## Observability

With `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` set, every turn is traced in Langfuse: one trace per voice or text turn, with child observations for speech-to-text, RAG retrieval, each tool call, every LLM generation (prompt, reply, latency, tokens) and text-to-speech. Without the keys, tracing is skipped and nothing else changes.

## Testing

```bash
uv run pytest                                   # unit + route tests (no network, ~280 tests)

# Live checks (these call the real Deepgram / Groq / OpenRouter / Cartesia APIs and cost a few requests):
uv run python -m tests.harness.batch_e2e        # Temporal batch pipeline on a local Temporal dev server
uv run python -m tests.harness.load_test http://localhost:8000    # concurrency, rate limiting, real chats
```

## Deployment

Two pieces, hosted separately (Vercel cannot run the backend: it needs WebSockets, a long-lived worker and
a local vector store).

### Backend (any container host: Railway, Render, Fly, a VPS)

Build the [Dockerfile](Dockerfile) (CPU-only PyTorch, runs as a non-root user, embedding model baked in).

1. **Service 1 — API:** default start command. Health check path: `/health`.
2. **Service 2 — worker** (only if you use the 🎤 batch mode): same image, start command `python -m temporal.worker`,
   plus a Temporal server it can reach (`TEMPORAL_ADDRESS`; Temporal Cloud works).
3. **Volume:** mount ONE persistent volume at `/data`. The database, vector store, uploaded documents and audio all
   live there (`DATA_DIR=/data` is already set in the image); without a volume they vanish on every redeploy.
4. **Environment variables:** `DEEPGRAM_API_KEY`, `GROQ_API_KEY`, `CARTESIA_API_KEY`, `OPENROUTER_API_KEY`,
   `LANGFUSE_*` (optional), and for a public site:
   `API_TOKEN` (or leave empty for an open demo), `TRUST_PROXY_HEADERS=true`, `CORS_ORIGINS=https://<your-frontend>`.
5. Run **one** API instance: the SQLite database, vector store and rate limiter are per process.

### Frontend (Vercel)

The page reads the backend URL from `<meta name="api-base">`, which [`static/build.mjs`](static/build.mjs) fills in at
build time from an environment variable, so no URL is hardcoded.

1. Import the repo in Vercel and set **Root Directory = `static`** (so Vercel doesn't mistake the repo for a Python app).
2. Set the environment variable **`API_BASE_URL`** = your backend's public `https://` URL (no path).
   A production build without it fails on purpose.
3. Deploy. Then set the backend's `CORS_ORIGINS` to the Vercel URL.

Or from a terminal: `cd static && npx vercel --prod` (`vercel env add API_BASE_URL production` first).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Connection refused 127.0.0.1:7233` | Start Temporal: `docker compose up -d` |
| Batch mode times out | Make sure the worker is running: `uv run python -m temporal.worker` |
| Reply is the fallback sentence every time | The LLM call is failing. Check the server log; most often `GROQ_MODEL` isn't available to your key |
| First message is slow after a restart | The embedding model loads in the background at startup (~20 s); it is cached afterwards |
| Bot reacts to background noise | Raise `VAD_MIN_SPEECH_RMS` or `VAD_AGGRESSIVENESS` |
