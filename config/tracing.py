"""
Central Langfuse setup, shared by the Temporal worker and the realtime server.

Langfuse reads os.environ, but our config lives in `settings`, so the client is
registered explicitly. Everything else (`@observe`, `langfuse.openai`) picks up
this singleton. If the keys are missing, tracing is silently disabled and the
app behaves exactly as before.
"""

from __future__ import annotations

import logging

from langfuse import Langfuse, get_client, observe, propagate_attributes

from config.settings import settings

logger = logging.getLogger(__name__)

__all__ = ["init_tracing", "flush", "get_client", "observe", "propagate_attributes"]

_enabled = False


def init_tracing() -> bool:
    """Register the Langfuse client once. Returns True if tracing is on."""
    global _enabled
    if _enabled:
        return True
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )
        _enabled = True
    else:
        logger.warning("Langfuse keys not set; calls will not be traced.")
    return _enabled


def flush() -> None:
    """Send any buffered traces (call on shutdown)."""
    if _enabled:
        get_client().flush()


init_tracing()
