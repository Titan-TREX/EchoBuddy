"""`GET /api/auth/check`: lets the UI find out whether a token is needed and valid."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from realtime.security import access_guard

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/check", dependencies=[Depends(access_guard(cost=0))])  # token check only, not rate-limited
async def check() -> dict:
    return {"ok": True}
