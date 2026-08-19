"""OpenAI vision OCR over plain httpx."""

import base64
import httpx
from typing import Optional

from .base import BaseOCRProvider
from ...utils.prompt import OCR_PROMPT


CHATGPT_MODELS = [
    {
        "id": "gpt-5.2",
        "name": "GPT-5.2",
        "description": "Latest and most capable multimodal model"
    },
    {
        "id": "gpt-5-mini",
        "name": "GPT-5 Mini",
        "description": "Smaller, faster, and more affordable"
    },
]

CHATGPT_MODEL_IDS = [m["id"] for m in CHATGPT_MODELS]
CHATGPT_DEFAULT_MODEL = "gpt-5.2"


class ChatGPTProvider(BaseOCRProvider):
    MODELS = CHATGPT_MODELS
    DEFAULT_MODEL = CHATGPT_DEFAULT_MODEL
    MODEL_IDS = CHATGPT_MODEL_IDS

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
            "https://api.openai.com/v1/chat/completions",
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
