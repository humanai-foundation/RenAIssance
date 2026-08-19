"""Optional LLM cleanup pass over OCR text. Providers live in the factory."""

import traceback
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from ..services.llm_processing.factory import (
    post_process,
    provider_requires_key,
    LLM_PROVIDERS,
)
from ..services.llm_processing.local_client import base_model_cached
from ..services.llm_processing.prompt_templates import list_templates


router = APIRouter(prefix="/api/llm", tags=["llm"])


# ── Schemas ─────────────────────────────────────────────────────────

class PostProcessRequest(BaseModel):
    text: str
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    template: str = "full_cleanup"


class PostProcessResponse(BaseModel):
    success: bool
    processed_text: Optional[str] = None
    error: Optional[str] = None
    model_used: Optional[str] = None
    provider_used: Optional[str] = None


# ── Endpoints ───────────────────────────────────────────────────────

@router.get("/providers")
async def get_providers():
    """List post-processing providers + their text models for the UI."""
    return {"providers": LLM_PROVIDERS}


@router.get("/templates")
async def get_templates():
    """List available post-processing prompt templates."""
    return {"templates": list_templates()}


@router.get("/local-status")
async def local_model_status():
    """Are the local corrector's base weights on disk yet?

    Same contract as /api/detect/models-status: the UI checks this first so it
    can warn about the one-time ~8 GB gated download instead of looking hung.
    """
    return {"models_ready": base_model_cached()}


@router.post("/post-process", response_model=PostProcessResponse)
async def post_process_endpoint(
    request: PostProcessRequest,
    # X-Gemini-API-Key stays accepted so the older Gemini flow keeps working.
    x_llm_api_key: Optional[str] = Header(None, alias="X-LLM-API-Key"),
    x_gemini_api_key: Optional[str] = Header(None, alias="X-Gemini-API-Key"),
):
    """Post-process OCR text using the selected LLM provider."""
    if not request.text or not request.text.strip():
        return PostProcessResponse(
            success=True,
            processed_text=request.text,
            model_used=request.model,
            provider_used=request.provider,
        )

    api_key = x_llm_api_key or x_gemini_api_key
    if provider_requires_key(request.provider) and (not api_key or not api_key.strip()):
        raise HTTPException(
            status_code=401,
            detail="Missing API key (X-LLM-API-Key header).",
        )

    try:
        result = post_process(
            provider=request.provider,
            api_key=api_key,
            text=request.text,
            model=request.model,
            template_name=request.template,
        )
        return PostProcessResponse(
            success=True,
            processed_text=result,
            model_used=request.model,
            provider_used=request.provider,
        )
    except ValueError as e:
        # Unknown/disabled provider — the caller's fault, not ours.
        return PostProcessResponse(success=False, error=str(e))
    except Exception as e:
        traceback.print_exc()
        error_msg = str(e)
        error_lower = error_msg.lower()
        if "api key" in error_lower or "authenticate" in error_lower or "401" in error_lower:
            return PostProcessResponse(success=False, error="Invalid API key")
        if "quota" in error_lower or "rate" in error_lower or "429" in error_lower:
            return PostProcessResponse(success=False, error="Rate limited — please try again later")
        return PostProcessResponse(success=False, error=error_msg)
