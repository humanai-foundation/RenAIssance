"""Gemini OCR via the google-genai SDK."""

import time

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from typing import Optional

from .base import BaseOCRProvider
from ...utils.prompt import OCR_PROMPT

# 5xx from Gemini ("high demand") usually clears in a second. 4xx won't, so
# only ServerError is retried below.
_MAX_RETRIES = 2
_RETRY_BACKOFF_S = (2,)


AVAILABLE_MODELS = [
    {
        "id": "gemini-3.1-flash-lite",
        "name": "Gemini 3.1 Flash Lite",
        "description": "Fast and light, best free-tier quota (recommended)"
    },
    {
        "id": "gemini-3-flash-preview",
        "name": "Gemini 3 Flash",
        "description": "Fast, high quality for most documents"
    },
    {
        "id": "gemini-3.5-flash",
        "name": "Gemini 3.5 Flash",
        "description": "Higher quality, good balance of speed and accuracy"
    },
    {
        "id": "gemini-3.1-pro-preview",
        "name": "Gemini 3.1 Pro",
        "description": "Most capable, best for hard pages (low daily quota)"
    },
    {
        "id": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "description": "Stable fallback when 3.x models are busy"
    },
]

DEFAULT_MODEL = "gemini-3.1-flash-lite"
MODEL_IDS = [m["id"] for m in AVAILABLE_MODELS]

# Without this a thinking model can hang indefinitely, burning free-tier quota.
REQUEST_TIMEOUT_MS = 120_000


def get_gemini_client(api_key: str):

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )


class GeminiProvider(BaseOCRProvider):
    """Gemini OCR provider using Google GenAI SDK."""

    MODELS = AVAILABLE_MODELS
    DEFAULT_MODEL = DEFAULT_MODEL
    MODEL_IDS = MODEL_IDS

    def _generate(self, client, model_name: str, image_bytes: bytes, mime_type: str, prompt: str):
        """One generation, retrying transient 5xx."""
        # Gemini 3 thinks by default and can burn the whole token budget
        # reasoning about a dense page, returning no text at all. Cap it.
        # thinking_level only exists in newer SDKs, hence the field check.
        config = None
        if model_name.startswith("gemini-3") and "thinking_level" in types.ThinkingConfig.model_fields:
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )

        for attempt in range(_MAX_RETRIES):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        prompt
                    ],
                    config=config,
                )
            except genai_errors.ServerError:
                if attempt >= _MAX_RETRIES - 1:
                    raise
                time.sleep(_RETRY_BACKOFF_S[attempt])

    def transcribe(self, api_key: str, image_bytes: bytes, model_name: str, mime_type: str = "image/png", custom_prompt: Optional[str] = None) -> str:
        client = get_gemini_client(api_key)
        prompt = custom_prompt if custom_prompt else OCR_PROMPT

        response = self._generate(client, model_name, image_bytes, mime_type, prompt)

        text = response.text
        if not text:
            # Empty means the budget went to thinking or the output was
            # blocked. Say so — a blank transcript looks like a silent failure.
            reason = getattr(response, "prompt_feedback", None) or getattr(
                response, "candidates", None
            )
            raise ValueError(f"Gemini returned no text (finish info: {reason})")
        return text
