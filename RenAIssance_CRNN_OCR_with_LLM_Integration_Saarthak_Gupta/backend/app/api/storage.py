"""Persistent storage API for transcripts and datasets."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..storage.storage_manager import (
    delete_entry,
    ensure_storage_layout,
    get_transcript_detail,
    get_dataset_detail,
    list_entries,
    resolve_storage_root,
    save_detection_dataset,
    save_recognition_dataset,
    save_transcript_session,
    zip_entry,
)

router = APIRouter(prefix="/api/storage", tags=["storage"])


class SaveTranscriptRequest(BaseModel):
    transcripts: dict[str, str] = Field(default_factory=dict)
    transcript_images: dict[str, str] = Field(default_factory=dict)
    source: str = "ocr upload"
    mode: str = "recognition"
    book_name: str = "transcript"
    model_info: dict[str, Any] = Field(default_factory=dict)


class SaveDatasetPageItem(BaseModel):
    page_key: str
    image_data: str
    boxes: list = Field(default_factory=list)
    lines: list = Field(default_factory=list)


class SaveDatasetRequest(BaseModel):
    pages: list[SaveDatasetPageItem] = Field(default_factory=list)
    source: str = "dataset generation"
    book_name: str = "dataset"
    bbox_format: str = "txt"
    mode: str = "recognition"
    model_info: dict[str, Any] = Field(default_factory=dict)


@router.get("/health")
async def storage_health():
    paths = ensure_storage_layout()
    return {
        "ok": True,
        "root": resolve_storage_root(),
        "paths": paths,
    }


@router.get("/overview")
async def storage_overview():
    return {
        "transcripts": list_entries("transcripts"),
        "datasets": list_entries("datasets"),
    }


@router.get("/transcripts")
async def list_transcripts():
    return {"items": list_entries("transcripts")}


@router.get("/datasets")
async def list_datasets():
    return {"items": list_entries("datasets")}


@router.get("/transcripts/{session_id}")
async def transcript_detail(session_id: str):
    try:
        return get_transcript_detail(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Transcript session not found")


@router.get("/datasets/{dataset_id}")
async def dataset_detail(dataset_id: str):
    try:
        return get_dataset_detail(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")


@router.post("/transcripts")
async def save_transcript(request: SaveTranscriptRequest):
    non_empty = {
        k: v
        for k, v in request.transcripts.items()
        if isinstance(v, str) and v.strip()
    }
    if not non_empty:
        raise HTTPException(status_code=400, detail="No transcript pages to save")

    metadata = save_transcript_session(
        transcripts=non_empty,
        source=request.source,
        mode=request.mode,
        transcript_images=request.transcript_images,
        book_name=request.book_name,
        model_info=request.model_info,
    )
    return {"success": True, "item": metadata}


@router.post("/datasets")
async def save_dataset(request: SaveDatasetRequest):
    if not request.pages:
        raise HTTPException(status_code=400, detail="No pages provided")

    pages_data = [page.model_dump() for page in request.pages]

    if request.mode == "detection":
        metadata = save_detection_dataset(
            pages_data=pages_data,
            source=request.source,
            book_name=request.book_name,
            bbox_format=request.bbox_format,
            model_info=request.model_info,
        )
    else:
        metadata = save_recognition_dataset(
            pages_data=pages_data,
            source=request.source,
            book_name=request.book_name,
            model_info=request.model_info,
        )

    return {"success": True, "item": metadata}


@router.get("/download/{kind}/{entry_id}")
async def download_entry(kind: str, entry_id: str):
    if kind not in {"transcripts", "datasets"}:
        raise HTTPException(status_code=400, detail="Unsupported storage kind")

    try:
        buffer, filename = zip_entry(kind, entry_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Stored item not found")

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/{kind}/{entry_id}")
async def remove_entry(kind: str, entry_id: str):
    if kind not in {"transcripts", "datasets"}:
        raise HTTPException(status_code=400, detail="Unsupported storage kind")

    deleted = delete_entry(kind, entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Stored item not found")

    return {"success": True, "deleted": entry_id}
