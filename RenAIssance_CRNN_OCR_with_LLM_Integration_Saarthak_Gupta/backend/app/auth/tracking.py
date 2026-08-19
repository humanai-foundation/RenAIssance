"""Push signups to a central Supabase table, using the local DB as the queue.

`tracked_at IS NULL` means "not sent yet". We try once on signup, and a retry
loop sweeps the rest — so an outage (or no config at all) loses nothing, even
across restarts. Never blocks signup, never sends the password hash.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from urllib.parse import quote

import httpx
from sqlalchemy import select

from ..core.config import (
    APP_INSTANCE_ID,
    SUPABASE_PUBLISHABLE_KEY,
    SUPABASE_URL,
    TRACKING_RETRY_INTERVAL,
)
from .db import SessionLocal
from .models import User

logger = logging.getLogger("renaissance.auth.tracking")


def tracking_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY)


def _post_signup(user: User) -> bool:
    """POST one signup. True on success, including "already there".

    Plain insert, not PostgREST upsert — upsert needs read rights and trips the
    insert-only RLS policy. Duplicates just 409 and we treat that as done.
    """
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/users"
    headers = {
        "apikey": SUPABASE_PUBLISHABLE_KEY,
        "Authorization": f"Bearer {SUPABASE_PUBLISHABLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    payload = {
        "username": user.username,
        "name": user.name,
        "email": user.email,
        "institute": user.institute,
        "instance_id": APP_INSTANCE_ID,
    }
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=8.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[tracking] insert failed (will retry): %s", exc)
        return False

    # 409 = already recorded, probably from another instance. Not an error.
    if resp.status_code < 300 or resp.status_code == 409:
        return True
    logger.warning(
        "[tracking] Supabase returned %s (will retry): %s",
        resp.status_code,
        resp.text[:300],
    )
    return False


def _flush_pending_sync() -> int:
    """Push every not-yet-tracked user. Returns how many were pushed."""
    if not tracking_enabled():
        return 0

    pushed = 0
    db = SessionLocal()
    try:
        pending = db.scalars(select(User).where(User.tracked_at.is_(None))).all()
        for user in pending:
            if _post_signup(user):
                user.tracked_at = datetime.now(timezone.utc)
                db.commit()
                pushed += 1
            else:
                db.rollback()  # leave tracked_at NULL -> retried next cycle
    finally:
        db.close()

    if pushed:
        logger.info("[tracking] Pushed %d signup(s) to Supabase", pushed)
    return pushed


def track_user_now(user_id: int) -> None:
    """Try to push one signup right away. On failure the retry loop gets it."""
    if not tracking_enabled():
        logger.info("[tracking] Supabase not configured — skipping (id=%s)", user_id)
        return
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if user is None or user.tracked_at is not None:
            return
        if _post_signup(user):
            user.tracked_at = datetime.now(timezone.utc)
            db.commit()
            logger.info("[tracking] Recorded signup %s centrally", user.email)
    finally:
        db.close()


def update_tracked_profile(
    old_email: str, username: str, name: str, email: str, institute: str | None
) -> None:
    """PATCH the central row after a profile edit. Never raises.

    Matched by the OLD email. If the row was never pushed, this matches nothing
    and the pending insert carries the new values anyway.
    """
    if not tracking_enabled():
        return
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/users?email=eq.{quote(old_email)}"
    headers = {
        "apikey": SUPABASE_PUBLISHABLE_KEY,
        "Authorization": f"Bearer {SUPABASE_PUBLISHABLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    payload = {"username": username, "name": name, "email": email, "institute": institute}
    try:
        resp = httpx.patch(url, headers=headers, json=payload, timeout=8.0)
        if resp.status_code < 300:
            logger.info("[tracking] Synced profile update for %s", email)
        else:
            logger.warning(
                "[tracking] Profile sync returned %s: %s",
                resp.status_code,
                resp.text[:300],
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[tracking] Profile sync failed: %s", exc)


async def retry_loop() -> None:
    """Flush unsent signups on an interval. First pass runs immediately."""
    while True:
        try:
            await asyncio.to_thread(_flush_pending_sync)
        except Exception as exc:  # noqa: BLE001 — loop must never die
            logger.warning("[tracking] retry loop error: %s", exc)
        await asyncio.sleep(TRACKING_RETRY_INTERVAL)
