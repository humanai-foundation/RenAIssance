#!/usr/bin/env bash
# ============================================================================
# RenAIssance — NATIVE macOS run (uses the Apple GPU via PyTorch MPS)
#
#   ./run-native.sh
#
# WHY native: Docker on macOS has no Metal/GPU passthrough, so the Docker
# images always run on CPU there. Running natively lets the torch-based parts
# (TrOCR, CRNN, local LLM) use the Apple GPU through MPS — the device helper
# (app/utils/torch_device.py) selects "mps" automatically. NOTE: PaddleOCR has
# no Metal backend, so layout/text *detection* still runs on CPU on a Mac.
#
# This sets up a local Python venv (CPU paddle + MPS-capable torch) and runs:
#   backend  →  uvicorn on http://localhost:8000
#   frontend →  vite dev server on http://localhost:5173 (proxies /api → :8000)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

[ "$(uname -s)" = "Darwin" ] || { echo "run-native.sh is for macOS only. On Linux/Windows use ./run.sh"; exit 1; }

green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

# ---- prerequisites --------------------------------------------------------
PY="$(command -v python3.11 || command -v python3 || true)"
[ -n "$PY" ] || { echo "Python 3.11 (or python3) is required: brew install python@3.11"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js is required: brew install node"; exit 1; }

VENV=backend/.venv-native
if [ ! -d "$VENV" ]; then
  bold "Creating Python venv ($VENV)…"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

bold "Installing backend dependencies (first run is slow)…"
pip install --upgrade pip >/dev/null
pip install -r backend/requirements.txt
# torch on macOS-arm64 from PyPI already ships the MPS backend — no special index.
# PaddlePaddle CPU wheel (no GPU on Mac). Best-effort: if the wheel is
# unavailable for this macOS/arch, detection won't work but MPS recognition will.
if ! pip install "paddlepaddle==3.0.0"; then
  echo "WARNING: could not install paddlepaddle (CPU) — PaddleOCR detection will be"
  echo "         unavailable on this Mac. torch (MPS) recognition still works."
fi

# Regenerate the TrOCR processor assets (needs network the first time).
python backend/scripts/normalize_trocr_config.py || \
  echo "NOTE: TrOCR config normalization skipped (weights may be absent locally)."

# ---- env ------------------------------------------------------------------
export STORAGE_ROOT="$(pwd)/storage"
mkdir -p "$STORAGE_ROOT/transcripts" "$STORAGE_ROOT/datasets"
[ -f .env ] && export $(grep -v '^[[:space:]]*#' .env | grep -E '^[A-Za-z_]+=' | xargs -0 2>/dev/null || true)

# ---- run both, clean up on exit ------------------------------------------
PIDS=()
cleanup() { bold "Shutting down…"; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

bold "Starting backend (uvicorn, MPS-aware) on :8000…"
( cd backend && PYTHONPATH=. exec uvicorn app.main:app --host 0.0.0.0 --port 8000 ) &
PIDS+=($!)

bold "Installing frontend deps + starting Vite dev server on :5173…"
( cd frontend && npm install --no-audit --no-fund && exec npm run dev ) &
PIDS+=($!)

green "RenAIssance (native) starting. Open: http://localhost:5173"
echo "  Press Ctrl-C to stop both."
wait
