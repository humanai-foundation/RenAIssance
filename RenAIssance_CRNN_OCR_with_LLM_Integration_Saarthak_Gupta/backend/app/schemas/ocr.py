"""Request/response models for the OCR, preprocess and export endpoints."""

from typing import Optional
from pydantic import BaseModel


class OCRResponse(BaseModel):
    success: bool
    transcript: Optional[str] = None
    error: Optional[str] = None
    model_used: str
    processing_time_ms: int



class ExportRequest(BaseModel):
    transcripts: dict  # {page_number: transcript_text}
    format: str  # "txt", "docx", "pdf"


class PreprocessRequest(BaseModel):
    image_data: str  # Base64 encoded image with data URL prefix
    operations: list  # List of {op, params, enabled} dicts
    preview_mode: bool = False  # Use faster algorithms for preview


# One per provider, each with its own default model.

class GeminiOCRRequest(BaseModel):
    image_data: str
    model: str = "gemini-3.1-flash-lite"
    custom_prompt: Optional[str] = None


class ChatGPTOCRRequest(BaseModel):
    image_data: str
    model: str = "gpt-4o"
    custom_prompt: Optional[str] = None

class DeepSeekOCRRequest(BaseModel):
    image_data: str
    model: str = "deepseek-chat"
    custom_prompt: Optional[str] = None


class QwenOCRRequest(BaseModel):
    image_data: str
    model: str = "qwen-vl-max"
    custom_prompt: Optional[str] = None


class BatchOCRItem(BaseModel):
    page_index: int
    image_data: str


class BatchOCRRequest(BaseModel):
    items: list[BatchOCRItem]
    model: str = "gemini-3.1-flash-lite"
    custom_prompt: Optional[str] = None


class BatchOCRResultItem(BaseModel):
    page_index: int
    success: bool
    transcript: Optional[str] = None
    error: Optional[str] = None
    processing_time_ms: int = 0


class BatchOCRResponse(BaseModel):
    results: list[BatchOCRResultItem]
    total_processing_time_ms: int
    successful_count: int
    failed_count: int
