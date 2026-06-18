# Voice-to-Voice AI Chatbot

A fault-tolerant, low-latency voice chatbot built on **Temporal.io**, with
**Deepgram** for speech-to-text, **Gemini** for the conversational brain,
**Cartesia** for text-to-speech, and a **Streamlit** front end.

# Running the app
You need to ensure your Temporal container is running, and then execute two python processes concurrently (in separate terminals), from the project root:

Bash
# Terminal 1: Temporal Worker process (executes activities)
uv run python -m temporal.worker

# Terminal 2: Streamlit UI
streamlit run ui/app.py
## Architecture

```
voice_chatbot_system/
├── config/                  # Environment-driven settings (config/settings.py)
├── ui/                       # Presentation layer - Streamlit app + Temporal client
├── temporal/                 # Orchestration layer
│   ├── workflows.py          # VoiceChatbotWorkflow - sequencing only, no I/O
│   ├── dto.py                # Shared dataclasses crossing workflow/activity boundaries
│   ├── worker.py             # Worker process (registers workflow + activities)
│   └── activities/           # Execution layer - one external dependency per file
│       ├── stt_activity.py   # Deepgram (speech-to-text)
│       ├── llm_activity.py   # Gemini (conversational brain)
│       ├── tts_activity.py   # Cartesia (text-to-speech)
│       └── db_activity.py    # SQLite persistence
├── database/                 # Infrastructure layer
│   ├── connection.py         # async SQLite connection helper
│   └── schema.sql            # users / sessions / messages (3NF, FK constraints)
├── assets/                    # Recorded + generated audio (.wav/.mp3), UUID filenames
├── requirements.txt
└── .env
```

### Pipeline (one voice turn)

```
Streamlit records audio (.wav)
  -> save to assets/usr_<uuid>.wav
  -> start VoiceChatbotWorkflow on task queue "voice-bot-queue"
       1. transcribe_audio        (Deepgram Nova-2)
       2. ensure_user_and_session (SQLite, idempotent)
       3. log_message (role=user) (SQLite)
       4. fetch_recent_history     (SQLite)
       5. generate_reply           (Gemini)
       6. synthesize_speech        (Cartesia -> assets/bot_<uuid>.mp3)
       7. log_message (role=assistant) (SQLite)
  -> workflow returns transcript + bot reply + bot audio path
  -> Streamlit renders the transcript and autoplays the reply
```

Every activity has an explicit `start_to_close_timeout` and a shared
exponential-backoff `RetryPolicy` (configured in `temporal/workflows.py`), so
transient socket errors, rate limits, or dropped connections to Deepgram,
Gemini, or Cartesia are retried automatically without crashing the workflow.

## Prerequisites

- Python 3.11+
- A running Temporal server. The quickest option for local development is
  the Temporal CLI's dev server:

  ```bash
  temporal server start-dev
  ```

  This starts a local server at `localhost:7233` and a Web UI at
  `http://localhost:8233`.

- API keys for [Deepgram](https://console.deepgram.com/), the
  [Gemini API](https://ai.google.dev/), and [Cartesia](https://play.cartesia.ai/).

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# then edit .env and fill in DEEPGRAM_API_KEY, GEMINI_API_KEY, CARTESIA_API_KEY
```

The SQLite database is created automatically (from `database/schema.sql`) the
first time any activity runs - no manual migration step is needed.

## Running the app

You need **three** processes running concurrently (in separate terminals),
from the project root:

```bash
# Terminal 1: Temporal dev server
temporal server start-dev

# Terminal 2: Temporal worker (executes activities)
python -m temporal.worker

# Terminal 3: Streamlit UI
streamlit run ui/app.py
```

Open the Streamlit URL printed in Terminal 3, click the microphone to record
a message, and wait for the assistant's spoken reply to autoplay.

You can inspect workflow executions, retries, and failures in the Temporal
Web UI at `http://localhost:8233`.

## Configuration reference

All configuration lives in `config/settings.py` and can be overridden via
environment variables or a `.env` file (see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEEPGRAM_API_KEY` | _(required)_ | Deepgram API key |
| `GEMINI_API_KEY` | _(required)_ | Gemini API key |
| `CARTESIA_API_KEY` | _(required)_ | Cartesia API key |
| `DEEPGRAM_MODEL` | `nova-2` | Deepgram STT model |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model used for replies |
| `GEMINI_MAX_HISTORY_TURNS` | `10` | Rolling context window size |
| `CARTESIA_MODEL` | `sonic-2` | Cartesia TTS model |
| `CARTESIA_VOICE_ID` | (default Cartesia voice) | Voice used for replies |
| `TEMPORAL_ADDRESS` | `localhost:7233` | Temporal server address |
| `TEMPORAL_TASK_QUEUE` | `voice-bot-queue` | Task queue shared by worker + client |
| `ACTIVITY_START_TO_CLOSE_TIMEOUT_SECONDS` | `60` | Per-activity timeout |
| `ACTIVITY_RETRY_MAX_ATTEMPTS` | `4` | Max attempts before an activity failure surfaces |

## Notes on third-party SDKs

The Deepgram, Gemini, and Cartesia Python SDKs are under active development
and occasionally change their async client APIs between major versions. The
code in `temporal/activities/` was written against the SDK versions pinned in
`requirements.txt`. If you upgrade a dependency and see `AttributeError`s
from an SDK call, check that provider's current Python SDK docs/changelog for
the matching method names.

## Database schema

`database/schema.sql` defines a 3-tier normalized layout:

- `users (id, username, created_at)`
- `sessions (id, user_id -> users.id, started_at, ended_at)`
- `messages (id, session_id -> sessions.id, role, content, audio_path, created_at)`
  - Indexed on `session_id` (and `(session_id, created_at)`) for fast history lookups.

Audio is never stored in the database - `messages.audio_path` holds a path
relative to the project root (e.g. `assets/bot_f2e4c1.mp3`), and the binary
lives on disk under `assets/`.
