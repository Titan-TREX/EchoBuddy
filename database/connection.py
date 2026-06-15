"""
Infrastructure layer: async SQLite connection management.

This module is the *only* place in the codebase that imports `aiosqlite`.
Activities call `get_connection()` and never touch the driver directly,
keeping persistence concerns fully isolated behind a small, testable
interface (Clean Architecture's "Infrastructure" layer).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiosqlite

from config.settings import settings

_init_lock = asyncio.Lock()
_initialized = False


async def init_database() -> None:
    """Create tables/indexes from `schema.sql` if they don't already exist.

    Idempotent and concurrency-safe: multiple activity workers starting up
    at the same time will not race to create the schema twice.
    """
    global _initialized
    if _initialized:
        return

    async with _init_lock:
        if _initialized:
            return

        settings.database_path.parent.mkdir(parents=True, exist_ok=True)
        schema_sql = settings.schema_path.read_text(encoding="utf-8")

        async with aiosqlite.connect(settings.database_path) as db:
            await db.executescript(schema_sql)
            await db.commit()

        _initialized = True


@asynccontextmanager
async def get_connection() -> AsyncIterator[aiosqlite.Connection]:
    """Yield a SQLite connection with foreign-key enforcement enabled.

    Ensures the schema has been applied before handing back the connection,
    and always closes the connection on exit (success or error).
    """
    await init_database()

    db = await aiosqlite.connect(settings.database_path)
    try:
        await db.execute("PRAGMA foreign_keys = ON;")
        yield db
    finally:
        await db.close()
