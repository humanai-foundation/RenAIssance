"""Interface every OCR provider implements."""

from abc import ABC, abstractmethod
from typing import Optional


class BaseOCRProvider(ABC):
    """Base class for all OCR providers."""

    # Subclasses must fill these in.
    MODELS: list = []
    DEFAULT_MODEL: str = ""
    MODEL_IDS: list = []

    @abstractmethod
    def transcribe(self, api_key: str, image_bytes: bytes, model_name: str, mime_type: str = "image/png", custom_prompt: Optional[str] = None) -> str:
        """OCR the image and return the text.

        custom_prompt replaces utils.prompt.OCR_PROMPT when the user supplies one.
        """
        ...
