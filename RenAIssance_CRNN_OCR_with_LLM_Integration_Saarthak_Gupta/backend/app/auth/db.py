"""Engine + session wiring. SQLite by default, Postgres if DATABASE_URL says so."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from ..core.config import DATABASE_URL

Base = declarative_base()


def _normalize_url(url: str) -> str:
    # SQLAlchemy 2.x wants "postgresql://"; Supabase/Heroku hand out "postgres://".
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


def _make_engine(url: str):
    # check_same_thread=False is needed for SQLite under FastAPI's threadpool,
    # and rejected by Postgres — hence the branch.
    if url.startswith("sqlite"):
        path = url.split("///", 1)[-1]
        if path and path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )
    return create_engine(url, pool_pre_ping=True)


engine = _make_engine(_normalize_url(DATABASE_URL))
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Create missing tables. Safe on every startup."""
    from . import models  # noqa: F401 — registers the models on Base

    Base.metadata.create_all(bind=engine)
    _ensure_columns()


def _ensure_columns() -> None:
    """Hand-rolled migration: create_all never ALTERs, so older users.db files
    are missing `tracked_at`. Existing rows get NULL and the retry loop fills them."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    try:
        columns = {c["name"] for c in inspector.get_columns("users")}
    except Exception:
        return  # table not present yet — create_all will have made it

    if "tracked_at" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN tracked_at TIMESTAMP"))


def get_db():
    """FastAPI dependency: one session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
