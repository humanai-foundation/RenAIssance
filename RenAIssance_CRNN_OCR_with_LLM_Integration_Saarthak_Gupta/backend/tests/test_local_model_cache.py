"""base_model_cached must not say "ready" while shards are still downloading.

huggingface_hub writes config.json / tokenizer.* / the shard index up front and
the multi-GB weights last, so a snapshot-dir-exists check reports ready almost
immediately and the UI never shows its one-time-download warning.
"""

import json

from app.services.llm_processing.local_client import (
    FINETUNED_BASE_MODEL,
    base_model_cached,
)

SHARDS = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


def _snapshot(tmp_path, monkeypatch):
    """An HF cache laid out like a real one, with only the small files present."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    snap = (
        tmp_path / "hub"
        / ("models--" + FINETUNED_BASE_MODEL.replace("/", "--"))
        / "snapshots" / "abc123"
    )
    snap.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (snap / name).write_text("{}")
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"layer.{i}": s for i, s in enumerate(SHARDS)}})
    )
    return snap


def test_missing_cache_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert base_model_cached() is False


def test_index_without_weights_is_not_ready(tmp_path, monkeypatch):
    """The exact mid-download state — the bug a dir-exists check misses."""
    _snapshot(tmp_path, monkeypatch)
    assert base_model_cached() is False


def test_partial_shards_are_not_ready(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path, monkeypatch)
    (snap / SHARDS[0]).write_bytes(b"weights")
    assert base_model_cached() is False


def test_all_shards_present_is_ready(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path, monkeypatch)
    for name in SHARDS:
        (snap / name).write_bytes(b"weights")
    assert base_model_cached() is True


def test_unsharded_checkpoint_is_ready(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path, monkeypatch)
    (snap / "model.safetensors.index.json").unlink()
    (snap / "model.safetensors").write_bytes(b"weights")
    assert base_model_cached() is True
