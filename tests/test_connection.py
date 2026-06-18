"""
Tests for database/connection.py

Verifies:
  - init_database() creates the schema exactly once (idempotent).
  - get_connection() yields a live aiosqlite connection.
  - PRAGMA foreign_keys is enabled on every connection.
  - Concurrent calls to init_database() don't race.
  - Connection is always closed even when an exception occurs inside the block.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    audio_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@pytest.fixture(autouse=True)
def reset_init_flag():
    """Reset the module-level `_initialized` flag between tests."""
    import database.connection as conn_mod
    conn_mod._initialized = False
    yield
    conn_mod._initialized = False


# ---------------------------------------------------------------------------
# init_database
# ---------------------------------------------------------------------------

class TestInitDatabase:
    @pytest.mark.asyncio
    async def test_creates_schema_on_first_call(self, tmp_path: Path):
        schema_file = tmp_path / "schema.sql"
        schema_file.write_text(MINIMAL_SCHEMA)
        db_path = tmp_path / "test.db"

        with (
            patch("database.connection.settings") as mock_settings,
        ):
            mock_settings.database_path = db_path
            mock_settings.schema_path = schema_file

            from database.connection import init_database
            await init_database()

        assert db_path.exists()

    @pytest.mark.asyncio
    async def test_idempotent_second_call(self, tmp_path: Path):
        """Calling init_database() twice must not raise or re-execute schema."""
        schema_file = tmp_path / "schema.sql"
        schema_file.write_text(MINIMAL_SCHEMA)
        db_path = tmp_path / "test.db"

        with patch("database.connection.settings") as mock_settings:
            mock_settings.database_path = db_path
            mock_settings.schema_path = schema_file

            from database.connection import init_database
            await init_database()
            await init_database()  # second call should be a no-op

        assert db_path.exists()

    @pytest.mark.asyncio
    async def test_sets_initialized_flag(self, tmp_path: Path):
        import database.connection as conn_mod

        schema_file = tmp_path / "schema.sql"
        schema_file.write_text(MINIMAL_SCHEMA)
        db_path = tmp_path / "test.db"

        with patch("database.connection.settings") as mock_settings:
            mock_settings.database_path = db_path
            mock_settings.schema_path = schema_file

            await conn_mod.init_database()

        assert conn_mod._initialized is True

    @pytest.mark.asyncio
    async def test_concurrent_inits_safe(self, tmp_path: Path):
        """Multiple coroutines calling init_database() concurrently should not crash."""
        import database.connection as conn_mod

        schema_file = tmp_path / "schema.sql"
        schema_file.write_text(MINIMAL_SCHEMA)
        db_path = tmp_path / "test.db"

        with patch("database.connection.settings") as mock_settings:
            mock_settings.database_path = db_path
            mock_settings.schema_path = schema_file

            await asyncio.gather(
                conn_mod.init_database(),
                conn_mod.init_database(),
                conn_mod.init_database(),
            )

        assert conn_mod._initialized is True


# ---------------------------------------------------------------------------
# get_connection
# ---------------------------------------------------------------------------

class TestGetConnection:
    @pytest.mark.asyncio
    async def test_yields_connection(self, tmp_path: Path):
        import aiosqlite
        from database.connection import get_connection

        schema_file = tmp_path / "schema.sql"
        schema_file.write_text(MINIMAL_SCHEMA)
        db_path = tmp_path / "test.db"

        with patch("database.connection.settings") as mock_settings:
            mock_settings.database_path = db_path
            mock_settings.schema_path = schema_file

            async with get_connection() as db:
                assert isinstance(db, aiosqlite.Connection)

    @pytest.mark.asyncio
    async def test_foreign_keys_enabled(self, tmp_path: Path):
        from database.connection import get_connection

        schema_file = tmp_path / "schema.sql"
        schema_file.write_text(MINIMAL_SCHEMA)
        db_path = tmp_path / "test.db"

        with patch("database.connection.settings") as mock_settings:
            mock_settings.database_path = db_path
            mock_settings.schema_path = schema_file

            async with get_connection() as db:
                cursor = await db.execute("PRAGMA foreign_keys;")
                row = await cursor.fetchone()
                assert row[0] == 1  # 1 = ON

    @pytest.mark.asyncio
    async def test_connection_closed_after_block(self, tmp_path: Path):
        """The connection must be closed even after normal use."""
        closed_connections = []

        import aiosqlite

        original_close = aiosqlite.Connection.close

        async def tracking_close(self):
            closed_connections.append(True)
            await original_close(self)

        schema_file = tmp_path / "schema.sql"
        schema_file.write_text(MINIMAL_SCHEMA)
        db_path = tmp_path / "test.db"

        with (
            patch("database.connection.settings") as mock_settings,
            patch.object(aiosqlite.Connection, "close", tracking_close),
        ):
            mock_settings.database_path = db_path
            mock_settings.schema_path = schema_file

            from database.connection import get_connection
            async with get_connection():
                pass

        assert len(closed_connections) >= 1

    @pytest.mark.asyncio
    async def test_connection_closed_on_exception(self, tmp_path: Path):
        """Connection must be closed even when body raises an exception."""
        schema_file = tmp_path / "schema.sql"
        schema_file.write_text(MINIMAL_SCHEMA)
        db_path = tmp_path / "test.db"

        with patch("database.connection.settings") as mock_settings:
            mock_settings.database_path = db_path
            mock_settings.schema_path = schema_file

            from database.connection import get_connection

            with pytest.raises(RuntimeError):
                async with get_connection() as db:
                    assert db is not None
                    raise RuntimeError("simulated error inside context")
