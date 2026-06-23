# Voice-to-Voice AI Chatbot

Dual-mode voice chatbot with **Temporal.io** (batch pipeline) and **WebSocket** (realtime pipeline). Uses **Deepgram** (STT), **Groq LLM**, **Cartesia** (TTS).

## Running the app

```bash
# Terminal 1: Docker up
Docker compose up -d
# Terminal 2: Temporal Worker
uv run python -m temporal.worker  

# Terminal 3: FastAPI Server
uv run uvicorn realtime.server:app --reload --port 8000

# Then open: http://localhost:8000
```

**Requirement:** Temporal server must be running (Docker or `temporal server start-dev`).

## Architecture

```
voice_chatbot_system/
├── config/                  # Environment settings (config/settings.py)
├── database/                # Infrastructure layer
│   ├── connection.py        # async SQLite connection helper
│   ├── schema.sql           # users / sessions / messages (3NF)
│   └── voice_chatbot.db     # SQLite database (auto-created)
├── temporal/                # Batch pipeline - Temporal orchestration
│   ├── workflows.py         # VoiceChatbotWorkflow (sequencing only)
│   ├── dto.py               # Shared dataclasses (workflow ↔ activity boundary)
│   ├── worker.py            # Worker process (registers workflow + activities)
│   └── activities/          # Execution layer - one external dependency per file
│       ├── stt_activity.py      # Deepgram (speech-to-text)
│       ├── llm_activity.py      # Groq LLM (conversational brain)
│       ├── tts_activity.py      # Cartesia (text-to-speech)
│       └── db_activity.py       # SQLite persistence (Temporal-safe)
├── realtime/                # Realtime pipeline - WebSocket streaming
│   ├── server.py            # FastAPI server (both pipelines)
│   ├── session.py           # Per-connection orchestration (VAD, turn-taking)
│   ├── db.py                # Direct SQLite ops (for realtime, non-Temporal)
│   ├── stt.py               # Deepgram (realtime utterance → text)
│   ├── llm_stream.py        # Groq streaming (sentence-by-sentence yields)
│   ├── tts_stream.py        # Cartesia streaming (PCM16 chunks)
│   └── static/
│       └── index.html       # Unified ChatGPT-style UI (both modes)
├── assets/                  # Audio files (user recordings + bot replies)
│                            # Shared by both pipelines
├── requirements.txt
└── .env
```

## Two Modes

| Button | Mode | Behavior |
|--------|------|----------|
| 🎤 Mic | Batch (Temporal) | Click = record, click again = upload → wait for full reply |
| 〜 Waveform | Realtime (WebSocket) | Click = start live mode, click again = stop |

Both modes share the same SQLite database and audio folder.

## Prerequisites

- Python 3.11+
- Temporal server running
- API keys: Deepgram, Groq, Cartesia

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Edit .env with your API keys
cp .env.example .env
```

## Configuration

| Variable | Default |
|----------|---------|
| `DEEPGRAM_API_KEY` | _(required)_ |
| `GROQ_API_KEY` | _(required)_ |
| `CARTESIA_API_KEY` | _(required)_ |
| `DEEPGRAM_MODEL` | `nova-2` |
| `GEMINI_MODEL` | `llama-3.3-70b-versatile` |
| `TEMPORAL_ADDRESS` | `localhost:7233` |
| `TEMPORAL_TASK_QUEUE` | `voice-bot-queue` |

## Database

```
users (id, username, created_at)
  ↓
sessions (id, user_id, started_at, ended_at)
  ↓
messages (id, session_id, role, content, audio_path, created_at)
```

Audio files stored in `assets/` with UUID filenames.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Connection refused 127.0.0.1:7233" | Start Temporal: `temporal server start-dev` |
| "Workflow failed" timeout | Check worker: `python -m temporal.worker` |
| Audio not serving (404) | Create folder: `mkdir -p assets` |

## Files Changed

- `realtime/server.py` — added `/api/voice-chat` endpoint + `/assets` mount
- `realtime/static/index.html` — new dual-mode UI
- All other files unchanged