"""Pydantic schemas for local (CRNN / TrOCR) recognition APIs."""

from typing import Optional
from pydantic import BaseModel


class LocalModelInfo(BaseModel):
    """Single available local OCR model."""
    id: str
    name: str
    model_type: str
    path: str


class LocalModelsResponse(BaseModel):
    """Response from /api/local-recognition-models."""
    models: list[LocalModelInfo]


class LocalRecognizeRequest(BaseModel):
    """Request body for /api/local-recognize."""
    image_data: str                        # base64-encoded page image
    boxes: list[list[list[float]]]         # list of polygons, each polygon = list of [x,y] points
    model_id: str                          # model id (e.g. "crnn:best_crnn")


class LocalRecognizeResult(BaseModel):
    """Single box recognition result."""
    box_index: int
    text: str


class LocalRecognizeResponse(BaseModel):
    """Response from /api/local-recognize."""
    results: list[LocalRecognizeResult]
    processing_time_ms: int
    model_used: str
    model_type: str
    device: str
    error: Optional[str] = None
