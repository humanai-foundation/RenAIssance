"""One client for OpenAI, DeepSeek and Qwen.

All three speak the same /chat/completions schema, so only the endpoint differs.
"""

import httpx

from .prompt_templates import get_template


def post_process_text_openai_compat(
    endpoint: str,
    api_key: str,
    text: str,
    model: str,
    template_name: str = "full_cleanup",
) -> str:
    """Clean up OCR text. endpoint is the provider's full chat-completions URL."""
    if not text or not text.strip():
        return text

    prompt = get_template(template_name) + text

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # Output is about as long as the input page, so this is plenty.
        "max_tokens": 4096,
    }

    response = httpx.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()
    result = response.json()["choices"][0]["message"]["content"]

    if not result or not result.strip():
        raise ValueError("LLM returned an empty response")

    return result.strip()
