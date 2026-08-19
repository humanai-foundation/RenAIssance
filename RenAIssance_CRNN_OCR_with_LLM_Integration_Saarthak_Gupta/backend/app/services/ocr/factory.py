"""Look up an OCR provider by name."""

from .base import BaseOCRProvider
from .gemini import GeminiProvider
from .chatgpt import ChatGPTProvider
from .deepseek import DeepSeekProvider
from .qwen import QwenProvider


_PROVIDERS = {
    "gemini": GeminiProvider,
    "chatgpt": ChatGPTProvider,
    "deepseek": DeepSeekProvider,
    "qwen": QwenProvider,
}


class OCRFactory:
    @staticmethod
    def get_provider(name: str) -> BaseOCRProvider:
        """Instantiate a provider by name. ValueError if it isn't one of ours."""
        provider_cls = _PROVIDERS.get(name.lower())
        if provider_cls is None:
            valid = ", ".join(_PROVIDERS.keys())
            raise ValueError(f"Unknown OCR provider: '{name}'. Valid providers: {valid}")
        return provider_cls()

    @staticmethod
    def list_providers() -> list[str]:
        return list(_PROVIDERS.keys())
