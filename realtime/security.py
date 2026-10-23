"""
Access control and rate limiting for the HTTP and WebSocket endpoints.

* **Token (optional):** if `settings.api_token` is set, every guarded endpoint
  needs `Authorization: Bearer <token>`. Browsers can't set headers on a
  WebSocket, so `/ws` also accepts `?token=<token>`. With no token configured
  the API is open - fine for a demo, and the rate limit still applies.
* **Rate limit:** a sliding window per client (IP address, or the
  `X-Forwarded-For` address when `settings.trust_proxy_headers` is on), held in
  memory. It protects your third-party API bills from a runaway script. It is
  per process: run behind a shared store (e.g. Redis) if you scale to several
  instances.
"""

from __future__ import annotations

import re
import secrets
import time
from collections import deque
from collections.abc import Awaitable, Callable

from fastapi import Header, HTTPException, Request, WebSocket

from config.settings import settings

_PRUNE_ABOVE_KEYS = 10_000

# Shape of the id a browser keeps for itself. It doubles as the owner of uploaded
# documents (and names their folder on disk), so it is validated everywhere it enters.
USER_ID_REGEX = r"[A-Za-z0-9_-]{8,64}"
_USER_ID = re.compile(USER_ID_REGEX)


def is_valid_user_id(value: object) -> bool:
    """True only for a string of the right shape (clients can send any JSON type)."""
    return isinstance(value, str) and _USER_ID.fullmatch(value) is not None


def owner_id(x_user_id: str | None = Header(default=None)) -> str:
    """FastAPI dependency: the caller's id from the `X-User-Id` header (400 if absent/invalid)."""
    if not is_valid_user_id(x_user_id):
        raise HTTPException(status_code=400, detail="Missing or invalid X-User-Id header.")
    return x_user_id


class RateLimiter:
    """Sliding-window limiter: at most `limit` cost units per `window` seconds per key."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[tuple[float, int]]] = {}

    def reset(self) -> None:
        self._hits.clear()

    def check(self, key: str, cost: int = 1) -> float:
        """Record `cost` units for `key`. Returns 0 if allowed, else seconds until retry."""
        now = time.monotonic()
        window = settings.rate_limit_window_seconds
        limit = settings.rate_limit_requests

        if len(self._hits) > _PRUNE_ABOVE_KEYS:
            self._prune(now - window)

        hits = self._hits.setdefault(key, deque())
        while hits and hits[0][0] <= now - window:
            hits.popleft()

        used = sum(units for _, units in hits)
        if used + cost > limit:
            return max(hits[0][0] + window - now, 1.0) if hits else float(window)

        hits.append((now, cost))
        return 0.0

    def _prune(self, cutoff: float) -> None:
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1][0] <= cutoff]:
            del self._hits[key]


limiter = RateLimiter()


def client_key(conn: Request | WebSocket) -> str:
    """Who is calling: the forwarded address behind a trusted proxy, else the socket peer."""
    if settings.trust_proxy_headers:
        forwarded = conn.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return conn.client.host if conn.client else "unknown"


def _token_ok(provided: str | None) -> bool:
    if not settings.api_token:
        return True  # auth disabled
    return provided is not None and secrets.compare_digest(provided, settings.api_token)


def _bearer(conn: Request | WebSocket) -> str | None:
    header = conn.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    return value.strip() if scheme.lower() == "bearer" and value else None


def access_guard(cost: int = 1) -> Callable[[Request], Awaitable[None]]:
    """FastAPI dependency: checks the token, then charges `cost` against the rate limit."""

    async def guard(request: Request) -> None:
        if not _token_ok(_bearer(request)):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing access token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        retry_after = limiter.check(client_key(request), cost) if cost else 0.0
        if retry_after:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please slow down.",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

    return guard


async def authorize_websocket(websocket: WebSocket) -> bool:
    """Check token and rate limit before accepting a WebSocket; closes it (1008) if refused."""
    provided = _bearer(websocket) or websocket.query_params.get("token")
    if not _token_ok(provided) or limiter.check(client_key(websocket)):
        await websocket.close(code=1008)
        return False
    return True
