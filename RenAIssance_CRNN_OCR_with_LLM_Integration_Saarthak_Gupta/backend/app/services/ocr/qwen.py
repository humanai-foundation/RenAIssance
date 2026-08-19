"""Qwen VL OCR over DashScope's OpenAI-compatible endpoint."""

import base64
import httpx
from typing import Optional

from .base import BaseOCRProvider
from ...utils.prompt import OCR_PROMPT


QWEN_MODELS = [
    {
        "id": "qwen-vl-max",
        "name": "Qwen VL Max",
        "description": "Most capable Qwen vision-language model"
    },
    {
        "id": "qwen-vl-ocr",
        "name": "Qwen VL OCR",
        "description": "Qwen model optimized for OCR tasks"
    },
    {
        "id": "qwen2.5-vl-72b-instruct",
        "name": "Qwen2.5 VL 72B",
        "description": "Large Qwen 2.5 vision-language model"
    },
    {
        "id": "qwen2.5-vl-7b-instruct",
        "name": "Qwen2.5 VL 7B",
        "description": "Efficient Qwen 2.5 vision model"
    },
]

QWEN_MODEL_IDS = [m["id"] for m in QWEN_MODELS]
QWEN_DEFAULT_MODEL = "qwen-vl-max"


class QwenProvider(BaseOCRProvider):
    MODELS = QWEN_MODELS
    DEFAULT_MODEL = QWEN_DEFAULT_MODEL
    MODEL_IDS = QWEN_MODEL_IDS

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
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
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
