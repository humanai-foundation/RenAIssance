"""Registry + dispatch for text-cleanup LLM providers.

Gemini uses the google-genai SDK; OpenAI/DeepSeek/Qwen share one
chat-completions client. `local_es` is our GPU-only Spanish fine-tune.
"""

from .gemini_client import post_process_text as _gemini_post_process
from .openai_compat_client import post_process_text_openai_compat
from .local_client import (
    post_process_text_finetuned,
    FINETUNED_BASE_MODEL,
)


# OpenAI-compatible chat-completions endpoints (same URLs as the OCR providers).
_OPENAI_COMPAT_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
}


# Feeds the frontend dropdowns via GET /api/llm/providers. These are TEXT
# models — Qwen uses qwen-plus/turbo/max here, not the qwen-vl-* OCR models.
LLM_PROVIDERS = [
    {
        "id": "gemini",
        "name": "Gemini",
        "enabled": True,
        "requires_key": True,
        "default_model": "gemini-2.5-flash",
        "models": [
            {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
            {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro"},
        ],
    },
    {
        "id": "openai",
        "name": "OpenAI",
        "enabled": True,
        "requires_key": True,
        "default_model": "gpt-5-mini",
        "models": [
            {"id": "gpt-5-mini", "name": "GPT-5 Mini"},
            {"id": "gpt-5.2", "name": "GPT-5.2"},
        ],
    },
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "enabled": True,
        "requires_key": True,
        "default_model": "deepseek-chat",
        "models": [
            {"id": "deepseek-chat", "name": "DeepSeek Chat"},
            {"id": "deepseek-reasoner", "name": "DeepSeek Reasoner"},
        ],
    },
    {
        "id": "qwen",
        "name": "Qwen",
        "enabled": True,
        "requires_key": True,
        "default_model": "qwen-plus",
        "models": [
            {"id": "qwen-plus", "name": "Qwen Plus"},
            {"id": "qwen-turbo", "name": "Qwen Turbo"},
            {"id": "qwen-max", "name": "Qwen Max"},
        ],
    },
    {
        # GPU-only, and the gated base download needs HF_TOKEN. See local_client.
        "id": "local_es",
        "name": "Local fine-tuned (Spanish)",
        "enabled": True,
        "requires_key": False,
        "default_model": FINETUNED_BASE_MODEL,
        "models": [
            {"id": FINETUNED_BASE_MODEL, "name": "Gemma-3-4B Spanish (fine-tuned)"},
        ],
        "note": "Fine-tuned on Spanish historical OCR; runs 4-bit on the server GPU (gated base downloaded on first use — needs HF_TOKEN).",
    },
]

_ENABLED_PROVIDER_IDS = {p["id"] for p in LLM_PROVIDERS if p["enabled"]}
_KEYLESS_PROVIDER_IDS = {
    p["id"] for p in LLM_PROVIDERS if not p.get("requires_key", True)
}


def provider_requires_key(provider: str) -> bool:
    """True unless the provider runs locally (no API key needed)."""
    return (provider or "").lower() not in _KEYLESS_PROVIDER_IDS


def post_process(
    provider: str,
    api_key: str | None,
    text: str,
    model: str,
    template_name: str = "full_cleanup",
) -> str:
    """Route the request to `provider`. Raises ValueError if it's unknown or off."""
    provider = (provider or "gemini").lower()

    if provider not in _ENABLED_PROVIDER_IDS:
        if any(p["id"] == provider for p in LLM_PROVIDERS):
            raise ValueError(
                f"Provider '{provider}' is not available yet."
            )
        valid = ", ".join(sorted(_ENABLED_PROVIDER_IDS))
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. Valid providers: {valid}"
        )

    if provider == "local_es":
        return post_process_text_finetuned(text, model, template_name)

    if provider == "gemini":
        return _gemini_post_process(api_key, text, model, template_name)

    return post_process_text_openai_compat(
        endpoint=_OPENAI_COMPAT_ENDPOINTS[provider],
        api_key=api_key,
        text=text,
        model=model,
        template_name=template_name,
    )
