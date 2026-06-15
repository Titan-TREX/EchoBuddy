-- ============================================================================
-- Voice Chatbot System - Relational Schema (SQLite)
--
-- Strict 3-tier normalized layout:
--   users     -> sessions    (1 user has many sessions)
--   sessions  -> messages    (1 session has many messages)
--
-- Audio binaries are NEVER stored here - `messages.audio_path` only ever
-- holds a relative filesystem path under `assets/`.
-- ============================================================================

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,                                   -- UUID4 string
    username    TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,                                   -- UUID4 string
    user_id     TEXT NOT NULL,
    started_at  TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    ended_at    TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    audio_path  TEXT,                                               -- relative path under assets/
    created_at  TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages (session_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages (session_id, created_at);
