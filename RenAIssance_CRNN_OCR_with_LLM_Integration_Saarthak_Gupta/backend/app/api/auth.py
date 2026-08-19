"""Signup / login / logout / me routes, plus a token-gated admin user list.

Sessions are a signed httpOnly cookie — no JWT, no refresh tokens.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Cookie, Depends, Header, HTTPException, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..auth.db import get_db
from ..auth.models import User
from ..auth.tracking import track_user_now, update_tracked_profile
from ..auth.schemas import (
    AuthResponse,
    LoginRequest,
    ProfileUpdateRequest,
    SignupRequest,
    UserOut,
)
from ..auth.security import (
    SESSION_COOKIE,
    create_session_token,
    hash_password,
    read_session_token,
    verify_password,
)
from ..core.config import ADMIN_TOKEN, PUBLIC_BASE_URL, SESSION_MAX_AGE

router = APIRouter()

# Secure flag only on https, otherwise the cookie wouldn't stick on localhost.
_COOKIE_SECURE = PUBLIC_BASE_URL.lower().startswith("https")


def _set_session_cookie(response: Response, user_id: int) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=create_session_token(user_id),
        max_age=SESSION_MAX_AGE,
        httponly=True,
        secure=_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def current_user(
    ren_session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
) -> User:
    """Dependency: resolve the logged-in user or raise 401."""
    user_id = read_session_token(ren_session or "")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# ── Signup ────────────────────────────────────────────────────────────────

@router.post("/api/auth/signup", response_model=AuthResponse)
def signup(
    payload: SignupRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    username = payload.username.strip()
    email = str(payload.email).strip().lower()

    existing = db.scalar(
        select(User).where((User.username == username) | (User.email == email))
    )
    if existing is not None:
        field = "username" if existing.username == username else "email"
        raise HTTPException(status_code=409, detail=f"That {field} is already registered")

    user = User(
        username=username,
        email=email,
        name=payload.name.strip(),
        institute=(payload.institute or "").strip() or "personal",
        password_hash=hash_password(payload.password),
        # No email verification — this is usage tracking, not a security gate.
        is_verified=True,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    # Push to central tracking after the response goes out. If it fails,
    # tracked_at stays NULL and the retry loop picks it up later.
    background_tasks.add_task(track_user_now, user.id)

    _set_session_cookie(response, user.id)
    return AuthResponse(user=UserOut(**user.public_dict()))


# ── Login ───────────────────────────────────────────────────────────────────

@router.post("/api/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)):
    ident = payload.username.strip()
    # Allow logging in with either username or email.
    user = db.scalar(
        select(User).where((User.username == ident) | (User.email == ident.lower()))
    )
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    user.last_login = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)

    _set_session_cookie(response, user.id)
    return AuthResponse(user=UserOut(**user.public_dict()))


# ── Logout ──────────────────────────────────────────────────────────────────

@router.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


# ── Current user ─────────────────────────────────────────────────────────────

@router.get("/api/auth/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return UserOut(**user.public_dict())


@router.patch("/api/auth/me", response_model=UserOut)
def update_profile(
    payload: ProfileUpdateRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Update name / email / institute (username is fixed).

    Saved locally, then best-effort synced to Supabase keyed on the old email.
    """
    old_email = user.email
    new_email = str(payload.email).strip().lower()

    if new_email != user.email:
        clash = db.scalar(
            select(User).where(User.email == new_email, User.id != user.id)
        )
        if clash is not None:
            raise HTTPException(status_code=409, detail="That email is already registered")

    user.name = payload.name.strip()
    user.institute = (payload.institute or "").strip() or "personal"
    user.email = new_email
    db.commit()
    db.refresh(user)

    background_tasks.add_task(
        update_tracked_profile,
        old_email,
        user.username,
        user.name,
        user.email,
        user.institute,
    )
    return UserOut(**user.public_dict())


# ── Admin: user tracking view ────────────────────────────────────────────────

@router.get("/api/admin/users")
def admin_users(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    db: Session = Depends(get_db),
):
    # Disabled unless an ADMIN_TOKEN is configured — never left open.
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=404, detail="Not found")
    if not x_admin_token or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    total = db.scalar(select(func.count()).select_from(User)) or 0
    verified = db.scalar(
        select(func.count()).select_from(User).where(User.is_verified.is_(True))
    ) or 0
    users = db.scalars(select(User).order_by(User.created_at.desc())).all()
    return {
        "total_users": total,
        "verified_users": verified,
        "users": [u.public_dict() for u in users],
    }
