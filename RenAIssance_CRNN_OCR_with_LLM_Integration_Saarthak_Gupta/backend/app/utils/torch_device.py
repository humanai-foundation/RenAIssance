"""One place to pick the torch device: CUDA > MPS > CPU.

MPS only kicks in on a native macOS run — Docker there has no Metal passthrough.
PaddleOCR ignores this entirely; it has no Metal backend.
"""

from __future__ import annotations

import torch


def select_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # getattr: some torch builds have no mps attribute at all.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"

