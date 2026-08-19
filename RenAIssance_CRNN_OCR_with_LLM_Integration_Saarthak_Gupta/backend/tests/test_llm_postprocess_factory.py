"""LLM post-processing factory wiring (no network, no model load)."""

import pytest

from app.services.llm_processing import factory
from app.services.llm_processing.local_client import (
    post_process_text_finetuned,
    FINETUNED_BASE_MODEL,
)


def test_local_es_enabled_and_keyless():
    es = next(p for p in factory.LLM_PROVIDERS if p["id"] == "local_es")
    assert es["enabled"] is True
    assert es["requires_key"] is False
    assert es["default_model"] == FINETUNED_BASE_MODEL
    # keyless providers must not demand an API key
    assert factory.provider_requires_key("local_es") is False


def test_local_es_routes_to_finetuned(monkeypatch):
    called = {}

    def fake(text, model, template_name):
        called["args"] = (text, model, template_name)
        return "CLEANED"

    monkeypatch.setattr(factory, "post_process_text_finetuned", fake)
    out = factory.post_process("local_es", None, "raw text", model="x", template_name="t")
    assert out == "CLEANED"
    assert called["args"] == ("raw text", "x", "t")


def test_finetuned_requires_gpu_on_cpu(monkeypatch):
    # On a CPU-only host the fine-tuned provider must fail clearly, never OOM.
    # The function imports select_torch_device at call time, so patch the source.
    import app.utils.torch_device as td
    monkeypatch.setattr(td, "select_torch_device", lambda: "cpu")
    with pytest.raises(ValueError, match="requires a GPU"):
        post_process_text_finetuned("some ocr line")
