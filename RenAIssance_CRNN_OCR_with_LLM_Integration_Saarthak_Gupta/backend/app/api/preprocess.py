"""Preprocessing endpoints — run the OpenCV pipeline over a base64 image."""

import os
import sys
import time
import base64
import numpy as np
import cv2

from fastapi import APIRouter, HTTPException

from ..schemas.ocr import PreprocessRequest

# `preprocessing` is a sibling of `app`, not a submodule of it.
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from preprocessing import run_pipeline, OP_REGISTRY, validate_pipeline_config


router = APIRouter()


@router.get("/api/preprocess/operations")
async def get_available_operations():
    """Operation names + one-line descriptions for the UI."""
    return {
        "operations": list(OP_REGISTRY.keys()),
        "descriptions": {
            "normalize": "Normalize image brightness and contrast levels",
            "grayscale": "Convert image to grayscale",
            "deskew": "Automatically correct image rotation/skew",
            "denoise": "Remove noise while preserving text edges",
            "contrast": "Enhance contrast using CLAHE",
            "sharpen": "Sharpen text edges for clearer text",
            "threshold": "Convert to binary (black and white)",
            "morph": "Morphological operations (open, close, dilate, erode, gradient, tophat, blackhat)",
            "remove_blobs": "Remove large ink blobs from scanned documents",
            "remove_noise": "Remove small speckles and scanning dust",
        }
    }


@router.post("/api/preprocess")
async def preprocess_image_endpoint(request: PreprocessRequest):
    """Run the pipeline over one image and hand back a base64 data URL.

    preview_mode swaps in faster, rougher algorithms for live tweaking.
    """
    start_time = time.time()

    try:
        validation = validate_pipeline_config(request.operations)
        if not validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_operations",
                    "message": "Invalid pipeline configuration",
                    "errors": validation["errors"]
                }
            )

        image_data = request.image_data
        if "," in image_data:  # data:image/png;base64,...
            header, encoded = image_data.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
        else:
            encoded = image_data
            mime_type = "image/png"

        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_image", "message": "Could not decode image"}
            )

        result = run_pipeline(
            image=image,
            steps=request.operations,
            continue_on_error=True,
            preview_mode=request.preview_mode,
        )

        if result.image is not None:
            # Answer in whatever format we were handed.
            if "jpeg" in mime_type or "jpg" in mime_type:
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
                _, buffer = cv2.imencode('.jpg', result.image, encode_param)
                output_mime = "image/jpeg"
            else:
                _, buffer = cv2.imencode('.png', result.image)
                output_mime = "image/png"

            encoded_result = base64.b64encode(buffer).decode('utf-8')
            result_data_url = f"data:{output_mime};base64,{encoded_result}"
        else:
            result_data_url = None

        processing_time = int((time.time() - start_time) * 1000)

        return {
            "success": result.success,
            "processed_image": result_data_url,
            "processing_time_ms": processing_time,
            "progress_info": result.progress_info,
            "errors": [
                {"step": e["step"], "error": e["error"]}
                for e in result.errors
            ] if result.errors else [],
        }

    except HTTPException:
        raise
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "processed_image": None,
            "processing_time_ms": processing_time,
            "error": str(e),
        }


@router.post("/api/preprocess/validate")
async def validate_operations(operations: list):
    """Dry-run check of a pipeline config. Returns {valid, errors}."""
    validation = validate_pipeline_config(operations)
    return validation
