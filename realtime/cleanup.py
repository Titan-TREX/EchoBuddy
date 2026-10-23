"""
Housekeeping for `settings.assets_dir`.

Every voice turn writes the user's recording (`usr_*`) and the bot's spoken
reply (`bot_*`) there, and nothing else ever removes them. This module deletes
the ones older than `settings.assets_retention_hours` on a schedule
(`settings.assets_cleanup_interval_seconds`). Set the retention to 0 to keep
everything.

Only files this app creates are touched: a known name prefix AND an allowed
audio extension, directly inside the assets directory. Anything else
(`.gitkeep`, subfolders, files you put there yourself) is left alone. Chat
history keeps its text; only the old audio behind it disappears.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

_GENERATED_PREFIXES = ("usr_", "bot_")


def _is_generated_audio(path: Path) -> bool:
    return (
        path.is_file()
        and path.name.startswith(_GENERATED_PREFIXES)
        and path.suffix.lower() in settings.allowed_audio_extensions
    )


def purge_old_files(directory: Path, max_age_seconds: float, now: float | None = None) -> int:
    """Delete generated audio in `directory` last modified more than `max_age_seconds` ago.

    Returns how many files were removed. A file that can't be removed (in use,
    permissions) is logged and skipped; it never stops the sweep.
    """
    cutoff = (time.time() if now is None else now) - max_age_seconds
    removed = 0
    for path in directory.iterdir():
        try:
            if _is_generated_audio(path) and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            logger.warning("Could not remove old audio file %s", path, exc_info=True)
    return removed


async def cleanup_loop() -> None:
    """Sweep forever (until cancelled). Does nothing if retention is disabled."""
    if settings.assets_retention_hours <= 0:
        logger.info("Audio cleanup disabled (ASSETS_RETENTION_HOURS=0).")
        return

    while True:
        try:
            removed = await asyncio.to_thread(
                purge_old_files, settings.assets_dir, settings.assets_retention_hours * 3600
            )
            if removed:
                logger.info("Removed %d audio file(s) older than %sh.", removed, settings.assets_retention_hours)
        except Exception:
            logger.exception("Audio cleanup sweep failed")  # keep the loop alive
        await asyncio.sleep(settings.assets_cleanup_interval_seconds)
