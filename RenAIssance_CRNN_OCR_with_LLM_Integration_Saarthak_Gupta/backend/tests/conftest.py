"""Shared pytest fixtures + path setup for the backend test suite.

All tests here are CPU-safe (no GPU required) so they run unchanged in CI on
GPU-less runners. Real-GPU coverage lives in tests/smoke_gpu.py (run manually).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Make `app` and `preprocessing` importable, mirroring app.main's path setup.
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Never touch real user data: point storage at a throwaway dir.
os.environ.setdefault("STORAGE_ROOT", str(BACKEND_DIR / ".pytest-storage"))


@pytest.fixture
def sample_color_image() -> np.ndarray:
    """A deterministic 64x64 BGR uint8 image."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_transcripts() -> dict:
    return {"1": "Hello world", "2": "Second page of text"}
