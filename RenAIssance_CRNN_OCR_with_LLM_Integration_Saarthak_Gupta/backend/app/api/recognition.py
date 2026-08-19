"""Recognition API Router — local (CRNN / TrOCR) model discovery and line OCR."""

import base64
import os
import time

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from ..schemas.recognition import (
    LocalModelInfo,
    LocalModelsResponse,
    LocalRecognizeRequest,
    LocalRecognizeResponse,
    LocalRecognizeResult,
)
from ..services.recognition.crnn_inference import (
    crop_polygon_gray,
    discover_models as discover_crnn_models,
    get_recognizer as get_crnn_recognizer,
)
from ..services.recognition.trocr_inference import (
    crop_polygon_rgb,
    discover_models as discover_trocr_models,
    get_recognizer as get_trocr_recognizer,
)

router = APIRouter()

# Weights live in a different place in dev vs the container, so take whichever
# of the two candidate dirs actually exists.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BACKEND_ROOT, ".."))

_CANDIDATE_DIRS = [
    os.path.join(_BACKEND_ROOT, "models", "weights"),              # container layout
    os.path.join(_REPO_ROOT, "backend", "models", "weights"),      # dev layout
]
_MODEL_SEARCH_DIR = next(
    (p for p in _CANDIDATE_DIRS if os.path.isdir(p)),
    _CANDIDATE_DIRS[0],
)

# Rescanning the weights dir per request is pointless; refresh clears this.
_model_cache: list[dict] | None = None


def _get_models() -> list[dict]:
    global _model_cache
    if _model_cache is None:
        _model_cache = [
            *discover_crnn_models(os.path.join(_MODEL_SEARCH_DIR, "crnn")),
            *discover_trocr_models(os.path.join(_MODEL_SEARCH_DIR, "trocr")),
        ]
    return _model_cache


@router.get("/api/local-recognition-models", response_model=LocalModelsResponse)
async def list_local_models():
    """Return all available local OCR model checkpoints."""
    models = _get_models()
    return LocalModelsResponse(
        models=[LocalModelInfo(**m) for m in models]
    )


@router.post("/api/local-recognition-models/refresh")
async def refresh_local_models():
    """Force rescan of model directory."""
    global _model_cache
    _model_cache = None
    models = _get_models()
    return {"count": len(models), "models": [m["name"] for m in models]}


def _decode_image(image_data: str) -> np.ndarray:
    """Decode a base64 (optionally data-URL prefixed) image to BGR numpy."""
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    raw = base64.b64decode(image_data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


@router.post("/api/local-recognize", response_model=LocalRecognizeResponse)
async def local_recognize(request: LocalRecognizeRequest):
    """Recognise the text in each line box with a local CRNN or TrOCR model.

    model_id is namespaced: "crnn:best_crnn", "trocr:default".
    """
    start = time.time()

    models = _get_models()
    model_info = next((m for m in models if m["id"] == request.model_id), None)
    if model_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_id}' not found. Available: {[m['id'] for m in models]}",
        )

    try:
        image_bgr = _decode_image(request.image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    model_type = model_info["model_type"]

    try:
        if model_type == "crnn":
            recognizer = get_crnn_recognizer(model_info["path"])
        elif model_type == "trocr":
            recognizer = get_trocr_recognizer(model_info["path"])
        else:
            raise RuntimeError(f"Unsupported model type: {model_type}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model '{request.model_id}': {e}",
        )

    results: list[LocalRecognizeResult] = []

    if len(request.boxes) > 0:
        crops = []
        for i, box in enumerate(request.boxes):
            try:
                if model_type == "trocr":
                    crop = crop_polygon_rgb(image_bgr, box)
                else:
                    crop = crop_polygon_gray(image_bgr, box)
                crops.append((i, crop))
            except Exception:
                results.append(LocalRecognizeResult(box_index=i, text=""))

        if crops:
            try:
                images = [c[1] for c in crops]
                texts = recognizer.predict_batch(images)
                for (idx, _), text in zip(crops, texts):
                    results.append(LocalRecognizeResult(box_index=idx, text=text))
            except Exception:
                # Batch failed — retry one at a time so one bad crop doesn't
                # cost us the whole page.
                for idx, img in crops:
                    try:
                        if hasattr(recognizer, "predict"):
                            text = recognizer.predict(img)
                        else:
                            text = recognizer.predict_batch([img])[0]
                    except Exception:
                        text = ""
                    results.append(LocalRecognizeResult(box_index=idx, text=text))

    results.sort(key=lambda r: r.box_index)

    elapsed_ms = int((time.time() - start) * 1000)

    return LocalRecognizeResponse(
        results=results,
        processing_time_ms=elapsed_ms,
        model_used=request.model_id,
        model_type=model_type,
        device=recognizer.device,
    )
