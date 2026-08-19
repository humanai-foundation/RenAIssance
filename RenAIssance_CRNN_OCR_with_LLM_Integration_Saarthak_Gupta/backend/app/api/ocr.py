"""OCR endpoints. Providers come from OCRFactory, shared work from ocr_helpers.

Only Gemini is rate-limited server-side (free-tier sliding window).
"""

import asyncio
import time

from fastapi import APIRouter, File, UploadFile, HTTPException, Header, Form
from typing import Optional

from ..core.config import MIN_API_KEY_LENGTH, MAX_BATCH_SIZE
from ..core.rate_limiter import rate_limiter
from ..schemas.ocr import (
    OCRResponse,
    GeminiOCRRequest,
    BatchOCRRequest,
    BatchOCRResponse,
    BatchOCRResultItem,
    ChatGPTOCRRequest,
    DeepSeekOCRRequest,
    QwenOCRRequest,
)
from ..services.ocr.factory import OCRFactory
from .ocr_helpers import (
    parse_base64_image,
    validate_model,
    validate_api_key_format,
    check_rate_limit,
    run_ocr,
)


router = APIRouter()


def _provider(name: str):
    return OCRFactory.get_provider(name)


# ── Model listing ───────────────────────────────────────────────

@router.get("/api/models")
async def get_gemini_models():
    """Available Gemini models."""
    p = _provider("gemini")
    return {"models": p.MODELS, "default": p.DEFAULT_MODEL}


@router.get("/api/chatgpt-models")
async def get_chatgpt_models():
    """Available ChatGPT models."""
    p = _provider("chatgpt")
    return {"models": p.MODELS, "default": p.DEFAULT_MODEL}


@router.get("/api/deepseek-models")
async def get_deepseek_models():
    """Available DeepSeek models."""
    p = _provider("deepseek")
    return {"models": p.MODELS, "default": p.DEFAULT_MODEL}


@router.get("/api/qwen-models")
async def get_qwen_models():
    """Available Qwen models."""
    p = _provider("qwen")
    return {"models": p.MODELS, "default": p.DEFAULT_MODEL}


# ── Key check + rate limit (Gemini free-tier) ───────────────────

@router.post("/api/validate-key")
async def validate_api_key(
    x_gemini_api_key: str = Header(..., alias="X-Gemini-API-Key"),
):
    """Format check only — the key is really verified on the first OCR call."""
    validate_api_key_format(x_gemini_api_key, MIN_API_KEY_LENGTH)
    return {
        "valid": True,
        "message": "API key format is valid. It will be verified on first use.",
    }


@router.get("/api/rate-limit-status")
async def get_rate_limit_status():
    """Current rate-limit status (Gemini free-tier sliding window)."""
    return rate_limiter.get_status()


# ── Gemini ──────────────────────────────────────────────────────

@router.post("/api/gemini-ocr-page", response_model=OCRResponse)
async def gemini_ocr_page(
    image: UploadFile = File(...),
    model: str = Form(default="gemini-3.1-flash-lite"),
    custom_prompt: Optional[str] = Form(default=None),
    x_gemini_api_key: str = Header(..., alias="X-Gemini-API-Key"),
):
    """Process a single uploaded image file with Gemini OCR."""
    check_rate_limit()
    provider = _provider("gemini")
    validate_model(model, provider.MODEL_IDS)
    validate_api_key_format(x_gemini_api_key)

    image_bytes = await image.read()
    content_type = image.content_type or "image/png"
    if content_type not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
        content_type = "image/png"

    result = run_ocr(provider, x_gemini_api_key, image_bytes, model, content_type, custom_prompt)
    if result.success:
        rate_limiter.record_request()
    return result


@router.post("/api/gemini-ocr-base64", response_model=OCRResponse)
async def gemini_ocr_base64(
    image_data: str = Form(...),
    model: str = Form(default="gemini-3.1-flash-lite"),
    custom_prompt: Optional[str] = Form(default=None),
    x_gemini_api_key: str = Header(..., alias="X-Gemini-API-Key"),
):
    """Process a base64-encoded image with Gemini OCR (form data)."""
    check_rate_limit()
    provider = _provider("gemini")
    validate_model(model, provider.MODEL_IDS)

    image_bytes, mime_type = parse_base64_image(image_data)
    result = run_ocr(provider, x_gemini_api_key, image_bytes, model, mime_type, custom_prompt)
    if result.success:
        rate_limiter.record_request()
    return result


@router.post("/api/gemini-ocr-json", response_model=OCRResponse)
async def gemini_ocr_json(
    request: GeminiOCRRequest,
    x_gemini_api_key: str = Header(..., alias="X-Gemini-API-Key"),
):
    """Process a base64-encoded image with Gemini OCR (JSON body)."""
    check_rate_limit()
    provider = _provider("gemini")
    validate_model(request.model, provider.MODEL_IDS)

    image_bytes, mime_type = parse_base64_image(request.image_data)
    result = run_ocr(provider, x_gemini_api_key, image_bytes, request.model, mime_type, request.custom_prompt)
    if result.success:
        rate_limiter.record_request()
    return result


@router.post("/api/gemini-ocr-batch", response_model=BatchOCRResponse)
async def gemini_ocr_batch(
    request: BatchOCRRequest,
    x_gemini_api_key: str = Header(..., alias="X-Gemini-API-Key"),
):
    """Process multiple images concurrently with Gemini OCR (max batch size 4)."""
    if not request.items:
        raise HTTPException(status_code=400, detail="Empty batch request")
    if len(request.items) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum of {MAX_BATCH_SIZE}. Got {len(request.items)} items.",
        )

    check_rate_limit(required_slots=len(request.items))
    provider = _provider("gemini")
    validate_model(request.model, provider.MODEL_IDS)

    batch_start = time.time()

    async def _process_item(item) -> BatchOCRResultItem:
        item_start = time.time()
        try:
            image_bytes, mime_type = parse_base64_image(item.image_data)
            transcript = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda ib=image_bytes, mt=mime_type: provider.transcribe(
                    x_gemini_api_key, ib, request.model, mt, request.custom_prompt
                ),
            )
            return BatchOCRResultItem(
                page_index=item.page_index,
                success=True,
                transcript=transcript,
                processing_time_ms=int((time.time() - item_start) * 1000),
            )
        except Exception as exc:
            return BatchOCRResultItem(
                page_index=item.page_index,
                success=False,
                error=str(exc),
                processing_time_ms=int((time.time() - item_start) * 1000),
            )

    results = await asyncio.gather(*[_process_item(i) for i in request.items])
    rate_limiter.record_requests(len(request.items))

    successful = sum(1 for r in results if r.success)
    return BatchOCRResponse(
        results=list(results),
        total_processing_time_ms=int((time.time() - batch_start) * 1000),
        successful_count=successful,
        failed_count=len(results) - successful,
    )


# ── Other providers (no server-side rate limiting) ──────────────

@router.post("/api/chatgpt-ocr-json", response_model=OCRResponse)
async def chatgpt_ocr_json(
    request: ChatGPTOCRRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    """Process a base64-encoded image with ChatGPT OCR."""
    provider = _provider("chatgpt")
    validate_model(request.model, provider.MODEL_IDS)
    image_bytes, mime_type = parse_base64_image(request.image_data)
    return run_ocr(provider, x_api_key, image_bytes, request.model, mime_type, request.custom_prompt)


@router.post("/api/deepseek-ocr-json", response_model=OCRResponse)
async def deepseek_ocr_json(
    request: DeepSeekOCRRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    """Process a base64-encoded image with DeepSeek OCR."""
    provider = _provider("deepseek")
    validate_model(request.model, provider.MODEL_IDS)
    image_bytes, mime_type = parse_base64_image(request.image_data)
    return run_ocr(provider, x_api_key, image_bytes, request.model, mime_type, request.custom_prompt)


@router.post("/api/qwen-ocr-json", response_model=OCRResponse)
async def qwen_ocr_json(
    request: QwenOCRRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    """Process a base64-encoded image with Qwen OCR."""
    provider = _provider("qwen")
    validate_model(request.model, provider.MODEL_IDS)
    image_bytes, mime_type = parse_base64_image(request.image_data)
    return run_ocr(provider, x_api_key, image_bytes, request.model, mime_type, request.custom_prompt)
