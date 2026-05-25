"""
POST /api/client-log
Receives browser console errors/warnings and prints them to the server stdout
so they appear in the same terminal as the API server logs.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["client-log"])
logger = logging.getLogger("visomaster.client")


class ClientLogEntry(BaseModel):
    level: str = "error"
    message: str = ""
    source: Optional[str] = None
    stack: Optional[str] = None


@router.post("/client-log", include_in_schema=False)
async def client_log(entry: ClientLogEntry) -> dict[str, bool]:
    """Forward browser console errors to the server log."""
    prefix = f"[browser:{entry.level.upper()}]"
    if entry.source:
        prefix += f" {entry.source}"
    msg = f"{prefix} {entry.message}"
    if entry.stack:
        msg += f"\n{entry.stack}"

    if entry.level == "error":
        logger.error(msg)
    elif entry.level == "warn":
        logger.warning(msg)
    else:
        logger.info(msg)

    return {"ok": True}
