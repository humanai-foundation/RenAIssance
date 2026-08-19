"""DeepSeek OCR over its OpenAI-compatible endpoint."""

import base64
import httpx
from typing import Optional

from .base import BaseOCRProvider
from ...utils.prompt import OCR_PROMPT


DEEPSEEK_MODELS = [
    {
        "id": "deepseek-chat",
        "name": "DeepSeek Chat",
        "description": "DeepSeek general chat model"
    },
    {
        "id": "deepseek-reasoner",
        "name": "DeepSeek Reasoner",
        "description": "DeepSeek reasoning model"
    },
]

DEEPSEEK_MODEL_IDS = [m["id"] for m in DEEPSEEK_MODELS]
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"


class DeepSeekProvider(BaseOCRProvider):
    MODELS = DEEPSEEK_MODELS
    DEFAULT_MODEL = DEEPSEEK_DEFAULT_MODEL
    MODEL_IDS = DEEPSEEK_MODEL_IDS

    def transcribe(self, api_key: str, image_bytes: bytes, model_name: str, mime_type: str = "image/png", custom_prompt: Optional[str] = None) -> str:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{image_b64}"
        prompt = custom_prompt if custom_prompt else OCR_PROMPT

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 4096,
        }

        response = httpx.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
