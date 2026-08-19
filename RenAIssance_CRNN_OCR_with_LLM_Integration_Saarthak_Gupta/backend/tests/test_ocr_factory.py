"""OCRFactory strategy/registry behavior (no network — only construction)."""

import pytest

from app.services.ocr.base import BaseOCRProvider
from app.services.ocr.factory import OCRFactory


def test_list_providers_contains_all():
    names = set(OCRFactory.list_providers())
    assert {"gemini", "chatgpt", "deepseek", "qwen"} <= names


@pytest.mark.parametrize("name", ["gemini", "chatgpt", "deepseek", "qwen", "GEMINI"])
def test_get_provider_returns_base_instance(name):
    provider = OCRFactory.get_provider(name)
    assert isinstance(provider, BaseOCRProvider)


def test_unknown_provider_raises_valueerror():
    with pytest.raises(ValueError):
        OCRFactory.get_provider("not-a-real-provider")


def test_gemini_default_model_is_gemini_3():
    from app.services.ocr import gemini

    assert gemini.DEFAULT_MODEL == "gemini-3.1-flash-lite"
    assert gemini.DEFAULT_MODEL in gemini.MODEL_IDS


def test_gemini_retries_transient_server_error(monkeypatch):
    """A 503 ServerError should be retried, not surfaced on the first hit."""
    from app.services.ocr import gemini
    from google.genai import errors as genai_errors

    class _FakeResp:  # shape APIError expects for a non-requests response
        body_segments = [{"error": {"message": "busy", "status": "UNAVAILABLE"}}]

    class _Ok:
        text = "hello world"

    calls = {"n": 0}

    class _Client:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                calls["n"] += 1
                if calls["n"] < 2:
                    raise genai_errors.ServerError(503, _FakeResp())
                return _Ok()

    monkeypatch.setattr(gemini, "get_gemini_client", lambda api_key: _Client())
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)  # no real backoff

    out = OCRFactory.get_provider("gemini").transcribe("key", b"img", "gemini-3.1-flash-lite")
    assert out == "hello world"
    assert calls["n"] == 2  # failed once, succeeded on retry


def test_gemini_empty_response_raises(monkeypatch):
    """A thinking model that returns no text must error, not yield ''."""
    from app.services.ocr import gemini

    class _Resp:
        text = None
        prompt_feedback = "BLOCKED"

    class _Client:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                return _Resp()

    monkeypatch.setattr(gemini, "get_gemini_client", lambda api_key: _Client())
    provider = OCRFactory.get_provider("gemini")
    with pytest.raises(ValueError):
        provider.transcribe("key", b"img", "gemini-3.1-flash-lite")
