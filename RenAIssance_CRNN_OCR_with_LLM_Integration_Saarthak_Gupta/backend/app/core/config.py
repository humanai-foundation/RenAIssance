"""Config constants and env vars, loaded once at import."""

import os

# Local (non-Docker) runs need .env loaded by hand. In Docker compose already
# injected these, and load_dotenv never overrides existing vars, so it's a no-op.
try:
    from dotenv import load_dotenv

    _cfg_dir = os.path.dirname(os.path.abspath(__file__))
    for _env_candidate in (
        os.path.join(_cfg_dir, "..", "..", "..", ".env"),  # repo root
        os.path.join(_cfg_dir, "..", "..", ".env"),         # backend/
    ):
        if os.path.isfile(_env_candidate):
            load_dotenv(_env_candidate)
            break
except ImportError:
    pass


def _default_storage_root() -> str:
    """Resolve storage root for both local and containerized runs."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    repo_root_candidate = os.path.abspath(os.path.join(backend_root, ".."))

    is_repo_layout = (
        os.path.isdir(os.path.join(repo_root_candidate, "backend"))
        and os.path.isdir(os.path.join(repo_root_candidate, "frontend"))
    )

    if is_repo_layout:
        return os.path.join(repo_root_candidate, "storage")
    return os.path.join(backend_root, "storage")

# FastAPI app metadata
APP_TITLE = "RenAIssance OCR API"
APP_VERSION = "2.0.0"

# CORS origins allowed by the backend
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
]

# API-key format validation
MIN_API_KEY_LENGTH = 20

# Batch OCR
MAX_BATCH_SIZE = 4

# Persistent storage root for "My Files"
STORAGE_ROOT = os.getenv("STORAGE_ROOT", _default_storage_root())


# ── Auth (all backend-only; the browser only ever sees a signed cookie) ──

# Local accounts. SQLite on the storage volume by default — zero setup, works
# offline. Point at a postgresql:// URL if you'd rather use Postgres.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(STORAGE_ROOT, 'users.db')}",
)

# Signs session cookies. Set a stable value in production — the random fallback
# means every restart logs everyone out.
SECRET_KEY = os.getenv("SECRET_KEY") or os.urandom(32).hex()

# Guards GET /api/admin/users. Unset = endpoint returns 404 instead of opening up.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# Deployed origin — only used to decide whether the session cookie is Secure.
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:5173")

# Session cookie lifetime (seconds). Default 30 days.
SESSION_MAX_AGE = int(os.getenv("SESSION_MAX_AGE", str(30 * 24 * 60 * 60)))


# ── Central signup tracking (Supabase REST) ──────────────────────────────
# The publishable key is safe to ship: the table has an INSERT-only RLS policy,
# so a leak can add a row and nothing else. Leave empty to disable tracking.
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_PUBLISHABLE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")

# Optional label so you can tell which deployment a signup came from.
APP_INSTANCE_ID = os.getenv("APP_INSTANCE_ID", "") or os.getenv("HOSTNAME", "unknown")

# How often (seconds) to retry signups that haven't reached Supabase yet.
TRACKING_RETRY_INTERVAL = int(os.getenv("TRACKING_RETRY_INTERVAL", "120"))
