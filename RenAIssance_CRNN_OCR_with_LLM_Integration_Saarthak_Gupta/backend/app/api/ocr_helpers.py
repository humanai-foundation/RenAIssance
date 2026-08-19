"""Shared bits every OCR endpoint uses: base64 parsing, validation,
rate-limit checks, and one wrapper that turns provider calls into OCRResponse."""

import base64
import time
from typing import Optional

from fastapi import HTTPException

try:
    import httpx
except ImportError:  # only Gemini setups can skip it
    httpx = None  # type: ignore

from ..core.rate_limiter import rate_limiter
from ..schemas.ocr import OCRResponse
from ..services.ocr.base import BaseOCRProvider


# ── Base64 parsing ──────────────────────────────────────────────


def parse_base64_image(image_data: str) -> tuple[bytes, str]:
    """Decode raw base64 or a data: URL. Returns (image_bytes, mime_type)."""
    if "," in image_data:
        header, encoded = image_data.split(",", 1)
        mime_type = (
            header.split(":")[1].split(";")[0]
            if ":" in header
            else "image/png"
        )
    else:
        encoded = image_data
        mime_type = "image/png"

    return base64.b64decode(encoded), mime_type


# ── Validation helpers ──────────────────────────────────────────


def validate_model(model: str, valid_ids: list[str]) -> None:
    """400 if the model isn't one we know about."""
    if model not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Available models: {valid_ids}",
        )


def validate_api_key_format(api_key: str, min_length: int = 10) -> None:
    """401 if the key is missing or implausibly short."""
    if not api_key or len(api_key) < min_length:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )


# ── Rate-limit helpers ──────────────────────────────────────────


def check_rate_limit(required_slots: int = 1) -> None:
    """429 with a JSON body when the sliding window has no room."""
    if required_slots <= 1:
        can_proceed, wait_time = rate_limiter.can_proceed()
        if not can_proceed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limited",
                    "message": f"Rate limit exceeded. Please wait {wait_time} seconds.",
                    "wait_seconds": wait_time,
                },
            )
    else:
        available = rate_limiter.get_available_slots()
        if available < required_slots:
            status = rate_limiter.get_status()
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limited",
                    "message": (
                        f"Not enough rate limit slots. "
                        f"Need {required_slots}, have {available}."
                    ),
                    "wait_seconds": status.get("wait_seconds", 60),
                    "available_slots": available,
                },
            )


# ── Unified OCR execution ──────────────────────────────────────


def run_ocr(
    provider: BaseOCRProvider,
    api_key: str,
    image_bytes: bytes,
    model: str,
    mime_type: str,
    custom_prompt: Optional[str] = None,
) -> OCRResponse:
    """Run the provider and normalise whatever it throws into an OCRResponse.

    httpx errors (ChatGPT/DeepSeek/Qwen) and Google SDK errors (Gemini) both
    land here and get mapped onto the same 401/429 shapes.
    """
    start_time = time.time()

    try:
        transcript = provider.transcribe(api_key, image_bytes, model, mime_type, custom_prompt)
        processing_time = int((time.time() - start_time) * 1000)
        return OCRResponse(
            success=True,
            transcript=transcript,
            model_used=model,
            processing_time_ms=processing_time,
        )

    except Exception as exc:
        processing_time = int((time.time() - start_time) * 1000)

        if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
            if exc.response.status_code == 401:
                raise HTTPException(status_code=401, detail="Invalid API key")
            if exc.response.status_code == 429:
                raise HTTPException(
                    status_code=429, detail="API rate limit exceeded"
                )

        error_msg = str(exc)
        if "API_KEY_INVALID" in error_msg or "401" in error_msg:
            raise HTTPException(status_code=401, detail="Invalid API key")
        # A daily quota is not our short sliding-window limit — it won't clear
        # for hours, so tag it separately and let the UI say so.
        if "RESOURCE_EXHAUSTED" in error_msg or "QUOTA_EXCEEDED" in error_msg or "quota" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "quota_exceeded",
                    "message": "Daily Gemini quota reached. Try again tomorrow, pick a different model, or use another API key.",
                },
            )
        if "429" in error_msg:
            raise HTTPException(
                status_code=429,
                detail={"error": "rate_limited", "wait_seconds": 20},
            )

        return OCRResponse(
            success=False,
            error=error_msg,
            model_used=model,
            processing_time_ms=processing_time,
        )
