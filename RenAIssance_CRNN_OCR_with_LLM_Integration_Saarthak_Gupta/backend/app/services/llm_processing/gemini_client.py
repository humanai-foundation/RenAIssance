"""Gemini text post-processing — same google-genai pattern as the OCR provider."""

from google import genai

from .prompt_templates import get_template


DEFAULT_MODEL = "gemini-2.5-flash"


def post_process_text(
    api_key: str,
    text: str,
    model: str = DEFAULT_MODEL,
    template_name: str = "full_cleanup",
) -> str:
    """Clean up OCR text with Gemini. template_name keys into prompt_templates."""
    if not text or not text.strip():
        return text

    client = genai.Client(api_key=api_key)
    prompt = get_template(template_name) + text

    response = client.models.generate_content(
        model=model,
        contents=[prompt],
    )

    result = response.text
    if not result or not result.strip():
        raise ValueError("LLM returned an empty response")

    return result.strip()
