#!/usr/bin/env bash
# ============================================================================
# RenAIssance backend container entrypoint.
#
# Runs a non-blocking preflight check (RAM + GPU/CUDA availability for the
# image variant) that only LOGS — it never aborts boot, so the server always
# starts and the non-GPU features (Gemini OCR, preprocessing, export) keep
# working even if the GPU image was started without `--gpus all`.
#
# The hard "your machine does not meet the requirements" gate lives in the
# host launcher (run.sh / run.ps1), which checks before pulling the image.
# ============================================================================
set -euo pipefail

# Best-effort: a failing preflight must not prevent the server from starting.
python /app/scripts/preflight.py || true

# Hand off to the CMD (uvicorn ...).
exec "$@"
