"""POST /api/detect/layout-aware-lines — image in, text-line polygons out.

One page at a time per worker: the models are ~1.5 GB each and loaded fresh per
call, so two concurrent pages double peak memory and get the worker OOM-killed.
Budget 8 GB VRAM (GPU) or 8 GB RAM (CPU, much slower).
"""

import asyncio
import time
import traceback

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile

from ..services.layout_detection import (
    models_cached,
    run_layout_aware_detection,
    select_tier,
)

router = APIRouter()


@router.get("/api/detect/models-status")
async def detection_models_status(use_gpu: bool = True):
    """Are the detection models on disk yet?

    The UI checks this first so it can warn about the one-time ~15-20 min
    download. Filesystem check only — never imports paddle.
    """
    return {"models_ready": models_cached(use_gpu)}

# FastAPI runs async handlers concurrently; without this two parallel pages
# both load paddle models and fight over VRAM.
_detection_lock = asyncio.Lock()


@router.post("/api/detect/layout-aware-lines")
async def detect_layout_aware_lines(
    image: UploadFile = File(...),
    use_gpu: bool = Form(False),
    region_padding: int = Form(50),
    layout_expand: int = Form(2),
    score_thresh: float = Form(0.5),
    upscale_min_h: int = Form(60),
    nms_iou_thresh: float = Form(0.3),
    gap_multiplier: float = Form(2.0),
    debug_dir: str = Form(""),
):
    """Detect text lines in an uploaded page.

    -> {lines: [4-point polygons], count, processing_time_ms, tier}.
    """
    start = time.time()

    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        del contents

        if img is None:
            return {
                "error": "Could not decode the uploaded image.",
                "lines": [],
                "count": 0,
                "processing_time_ms": 0,
            }

        # The sync work goes to a thread so /api/health stays responsive
        # while a page is in flight.
        async with _detection_lock:
            gpu_fallback = False
            resource_warnings = []

            # Probe inside the lock so the readings match what we're about to
            # allocate. Keeps 4-6 GB laptop GPUs off the server-class models.
            tier = await asyncio.to_thread(select_tier, use_gpu)
            print(f"[LayoutAPI] tier selected: {tier['tier']} on {tier['device']}"
                  f" — {tier['reason']}")
            effective_use_gpu = tier["device"] == "gpu"

            try:
                lines, resource_warnings = await asyncio.to_thread(
                    run_layout_aware_detection,
                    img,
                    use_gpu=effective_use_gpu,
                    layout_model_name=tier["layout_model"],
                    det_model_name=tier["det_model"],
                    rec_model_name=tier["rec_model"],
                    region_padding=region_padding,
                    layout_expand=layout_expand,
                    score_thresh=score_thresh,
                    upscale_min_h=upscale_min_h,
                    nms_iou_thresh=nms_iou_thresh,
                    gap_multiplier=gap_multiplier,
                    debug_dir=debug_dir,
                )
            except (ValueError, RuntimeError) as gpu_err:
                err_msg = str(gpu_err)
                if "Out-of-memory" in err_msg or "out of memory" in err_msg.lower():
                    elapsed = int((time.time() - start) * 1000)
                    return {
                        "error": err_msg,
                        "lines": [],
                        "count": 0,
                        "processing_time_ms": elapsed,
                        "resource_warnings": resource_warnings,
                        "tier": tier,
                        "resource_requirements": (
                            "Minimum 8 GB GPU VRAM required for GPU mode. "
                            "Minimum 8 GB free RAM required for CPU mode."
                        ),
                    }
                if effective_use_gpu:
                    # Crashed for some non-OOM reason — retry on CPU.
                    print(f"[LayoutAPI] GPU failed, falling back to CPU: {gpu_err}")
                    gpu_fallback = True
                    cpu_tier = await asyncio.to_thread(select_tier, False)
                    tier = cpu_tier
                    lines, resource_warnings = await asyncio.to_thread(
                        run_layout_aware_detection,
                        img,
                        use_gpu=False,
                        layout_model_name=cpu_tier["layout_model"],
                        det_model_name=cpu_tier["det_model"],
                        rec_model_name=cpu_tier["rec_model"],
                        region_padding=region_padding,
                        layout_expand=layout_expand,
                        score_thresh=score_thresh,
                        upscale_min_h=upscale_min_h,
                        nms_iou_thresh=nms_iou_thresh,
                        gap_multiplier=gap_multiplier,
                        debug_dir=debug_dir,
                    )
                else:
                    raise

        del img
        elapsed = int((time.time() - start) * 1000)

        resp = {
            "lines": lines,
            "count": len(lines),
            "processing_time_ms": elapsed,
            "tier": tier,
        }
        if resource_warnings:
            resp["resource_warnings"] = resource_warnings
        if gpu_fallback:
            resp["warning"] = "GPU failed mid-run. Fell back to CPU."
        elif use_gpu and tier["device"] == "cpu":
            resp["warning"] = (
                "Not enough free GPU VRAM for detection. "
                f"Ran on CPU instead. {tier['reason']}"
            )
        elif use_gpu and tier["tier"] == "mobile":
            resp["warning"] = (
                f"Low GPU VRAM — using lighter mobile models. {tier['reason']}"
            )
        if use_gpu and not gpu_fallback:
            resp["resource_requirements"] = (
                "Minimum 8 GB GPU VRAM recommended for GPU mode. "
                "Minimum 8 GB free RAM recommended for CPU mode."
            )
        return resp

    except Exception as exc:
        traceback.print_exc()
        elapsed = int((time.time() - start) * 1000)
        resp = {
            "error": f"Line detection failed: {str(exc)}",
            "lines": [],
            "count": 0,
            "processing_time_ms": elapsed,
        }
        # Warnings gathered before the failure often explain it.
        try:
            if resource_warnings:
                resp["resource_warnings"] = resource_warnings
                resp["resource_requirements"] = (
                    "Minimum 8 GB GPU VRAM required for GPU mode. "
                    "Minimum 8 GB free RAM required for CPU mode."
                )
        except NameError:
            pass
        return resp
