"""Startup warnings about missing GPU access or low RAM. Always exits 0.

The hard spec gate lives in run.sh / run.ps1; this is just the in-container net.
"""

from __future__ import annotations

import os
import sys

VARIANT = os.environ.get("RENAISSANCE_VARIANT", "gpu")
MIN_RECOMMENDED_RAM_GB = 8.0


def log(msg: str) -> None:
    print(f"[preflight] {msg}", flush=True)


def check_ram() -> None:
    try:
        import psutil

        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        if total_gb < MIN_RECOMMENDED_RAM_GB:
            log(
                f"WARNING: detected {total_gb:.1f} GB RAM; "
                f">= {MIN_RECOMMENDED_RAM_GB:.0f} GB is recommended. Large pages may "
                "run slowly or hit out-of-memory during detection/recognition."
            )
    except Exception:
        # psutil should always be present, but never let this block boot.
        pass


def check_paddle() -> None:
    """Probe Paddle/CUDA. For the GPU variant the import itself fails when the
    container has no GPU access (paddlepaddle-gpu links libcuda.so.1 at import)."""
    try:
        import paddle  # noqa: PLC0415

        cuda_runtime = False
        try:
            cuda_runtime = paddle.device.cuda.device_count() > 0
        except Exception:
            cuda_runtime = False

        if VARIANT == "gpu":
            if cuda_runtime:
                log("GPU image: CUDA device detected — PaddleOCR detection runs on GPU.")
            else:
                log(
                    "GPU image started WITHOUT a usable CUDA device. PaddleOCR layout "
                    "detection needs a GPU and will be unavailable. Re-run via the "
                    "launcher (./run.sh) or add `--gpus all`, or use the CPU image "
                    "(./run.sh --cpu). Gemini OCR, preprocessing and export still work."
                )
        else:
            log("CPU image: PaddleOCR and torch will run on CPU.")
    except Exception as exc:
        if VARIANT == "gpu":
            log(
                "GPU image: PaddlePaddle could not initialize CUDA "
                f"({exc.__class__.__name__}). This usually means the container was "
                "started without GPU access. PaddleOCR layout detection will be "
                "unavailable; re-run via ./run.sh (or add `--gpus all`), or use the "
                "CPU image (./run.sh --cpu). Other features still work."
            )
        else:
            log(f"Paddle import check skipped ({exc.__class__.__name__}).")


def main() -> int:
    log(f"variant={VARIANT}")
    check_ram()
    check_paddle()
    return 0


if __name__ == "__main__":
    sys.exit(main())
