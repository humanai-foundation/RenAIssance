"""Password hashing and signed-cookie sessions.

bcrypt for passwords, itsdangerous for opaque signed cookies. No JWT.
"""

from __future__ import annotations

import bcrypt
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from ..core.config import SECRET_KEY, SESSION_MAX_AGE

_session_signer = URLSafeTimedSerializer(SECRET_KEY, salt="ren-session")

SESSION_COOKIE = "ren_session"


# ── Passwords ─────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ── Session cookie ──────────────────────────────────────────────────────────

def create_session_token(user_id: int) -> str:
    """Sign the user id into an opaque, tamper-proof token."""
    return _session_signer.dumps({"uid": user_id})


def read_session_token(token: str) -> int | None:
    """Return the user id from a valid, unexpired token, else None."""
    if not token:
        return None
    try:
        data = _session_signer.loads(token, max_age=SESSION_MAX_AGE)
        return int(data["uid"])
    except (BadSignature, SignatureExpired, KeyError, ValueError, TypeError):
        return None
