"""``GET /api/health`` — liveness + warmup status.

Surfaces ``app.state.warmup_ok`` (set in ``main.py``'s lifespan) so the
frontend can decide whether to render the cold-start sub-label (DESIGN §6.2).
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/api/health")
async def health(request: Request) -> dict[str, object]:
    return {
        "status": "ok",
        "warmup_ok": bool(getattr(request.app.state, "warmup_ok", False)),
    }
