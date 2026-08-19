"""Layout-aware text-line detection with PaddleOCR.

Two stages: detect page regions (PP-DocLayout), then run text detection inside
each text region. Models are built fresh per call and freed after — keeping them
resident blew past 8 GB VRAM. Budget 8 GB VRAM (GPU) or 8 GB RAM (CPU, slow).
"""

import gc
import os
import time
from typing import List, Tuple, Dict, Any

# Paddle's PIR path throws ConvertPirAttribute2RuntimeAttribute on these models.
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"

import cv2
import numpy as np
import psutil

# paddlepaddle-gpu links libcuda.so.1 at import time, so importing it in a
# container without GPU access kills the server at boot. Import on first use
# instead and return a readable error if it fails.
_paddle = None
_create_model = None
_PaddleOCR = None
_PADDLE_IMPORT_ERROR: str = ""


def _ensure_paddle() -> None:
    """Import paddle/paddleocr on first use. Raises RuntimeError if unavailable."""
    global _paddle, _create_model, _PaddleOCR, _PADDLE_IMPORT_ERROR
    if _paddle is not None:
        return
    try:
        import paddle as _p
        from paddleocr import PaddleOCR as _O
        # Single layout model, not PPStructureV3 — that pipeline downloads every
        # sub-model it knows (tables, seals, formulas) even with the flags off.
        from paddlex import create_model as _cm
        _paddle = _p
        _create_model = _cm
        _PaddleOCR = _O
    except Exception as exc:
        _PADDLE_IMPORT_ERROR = str(exc)
        raise RuntimeError(
            f"PaddlePaddle failed to load: {exc}. "
            "Start the container with GPU access: docker run --gpus all ... "
            "and ensure CUDA 12.6 drivers are installed on the host. "
            "GPU paddle install: "
            "pip install paddlepaddle-gpu==3.3.0 "
            "-i https://www.paddlepaddle.org.cn/packages/stable/cu126/"
        ) from exc


# ── Tuning constants ──────────────────────────────────────────────────
TEXT_LABELS    = {"text", "paragraph_title", "doc_title", "header"}
REGION_PADDING = 50
LAYOUT_EXPAND  = 2
SCORE_THRESH   = 0.5
UPSCALE_MIN_H  = 60
NMS_IOU_THRESH = 0.3
GAP_MULTIPLIER = 2.0

# Thresholds for the warnings shown to the user.
MIN_GPU_VRAM_GB  = 8.0
MIN_RAM_GB       = 8.0
LOW_RAM_GB       = 4.0

# Tier ladder (env-overridable): enough free VRAM → server models on GPU, less →
# mobile on GPU, less still → CPU, picking server vs mobile there by free RAM.
TIER_SERVER_MIN_VRAM_GB = float(os.environ.get("TIER_SERVER_MIN_VRAM_GB", 6.0))
TIER_MOBILE_MIN_VRAM_GB = float(os.environ.get("TIER_MOBILE_MIN_VRAM_GB", 2.0))
TIER_CPU_SERVER_MIN_RAM_GB = float(os.environ.get("TIER_CPU_SERVER_MIN_RAM_GB", 8.0))

SERVER_MODELS = {
    "layout": "PP-DocLayout_plus-L",
    "det":    "PP-OCRv5_server_det",
    "rec":    "PP-OCRv5_server_rec",
}
MOBILE_MODELS = {
    "layout": "PP-DocLayout-S",
    "det":    "PP-OCRv5_mobile_det",
    "rec":    "PP-OCRv5_mobile_rec",
}


def _model_cache_root() -> str:
    """Where PaddleX caches downloaded models (<root>/official_models/).

    /paddle_models in Docker via PADDLE_PDX_CACHE_HOME, ~/.paddlex locally.
    """
    root = (
        os.environ.get("PADDLE_PDX_MODEL_CACHE_HOME")
        or os.environ.get("PADDLE_PDX_CACHE_HOME")
    )
    if root:
        return root
    return os.path.join(os.path.expanduser("~"), ".paddlex")


def models_cached(use_gpu: bool = True) -> bool:
    """True when this tier's models are already on disk.

    The UI uses this to warn about the multi-GB first-run download. A directory
    only counts if it exists AND is non-empty (a killed download leaves an empty one).
    """
    root = _model_cache_root()
    names = SERVER_MODELS if use_gpu else MOBILE_MODELS
    bases = [os.path.join(root, "official_models"), root]
    for model_name in names.values():
        present = False
        for base in bases:
            d = os.path.join(base, model_name)
            try:
                if os.path.isdir(d) and any(os.scandir(d)):
                    present = True
                    break
            except OSError:
                continue
        if not present:
            return False
    return True


def select_tier(use_gpu: bool) -> Dict[str, Any]:
    """Pick device + model tier from currently-free VRAM/RAM.

    -> {device, tier, layout_model, det_model, rec_model, reason,
        free_vram_gb, free_ram_gb}. `reason` is shown to the user as-is.
    """
    _ensure_paddle()
    info = check_system_resources(use_gpu)
    free_vram = info["gpu_vram_free_gb"]
    free_ram  = info["available_ram_gb"]

    if use_gpu and info["gpu_available"]:
        if free_vram >= TIER_SERVER_MIN_VRAM_GB:
            return {
                "device": "gpu", "tier": "server",
                "layout_model": SERVER_MODELS["layout"],
                "det_model":    SERVER_MODELS["det"],
                "rec_model":    SERVER_MODELS["rec"],
                "reason": (
                    f"{free_vram:.1f} GB free VRAM — using server models on GPU."
                ),
                "free_vram_gb": free_vram, "free_ram_gb": free_ram,
            }
        if free_vram >= TIER_MOBILE_MIN_VRAM_GB:
            return {
                "device": "gpu", "tier": "mobile",
                "layout_model": MOBILE_MODELS["layout"],
                "det_model":    MOBILE_MODELS["det"],
                "rec_model":    MOBILE_MODELS["rec"],
                "reason": (
                    f"{free_vram:.1f} GB free VRAM (< {TIER_SERVER_MIN_VRAM_GB:.0f} GB) "
                    "— using mobile models on GPU for lower memory footprint."
                ),
                "free_vram_gb": free_vram, "free_ram_gb": free_ram,
            }
        # Not enough VRAM even for mobile — fall through to CPU.
        cpu_reason_prefix = (
            f"{free_vram:.1f} GB free VRAM (< {TIER_MOBILE_MIN_VRAM_GB:.0f} GB) "
            "— falling back to CPU. "
        )
    else:
        cpu_reason_prefix = ""  # CPU asked for, or no GPU at all

    if free_ram >= TIER_CPU_SERVER_MIN_RAM_GB:
        return {
            "device": "cpu", "tier": "server",
            "layout_model": SERVER_MODELS["layout"],
            "det_model":    SERVER_MODELS["det"],
            "rec_model":    SERVER_MODELS["rec"],
            "reason": (
                f"{cpu_reason_prefix}{free_ram:.1f} GB free RAM — "
                "using server models on CPU (slow but accurate)."
            ),
            "free_vram_gb": free_vram, "free_ram_gb": free_ram,
        }
    return {
        "device": "cpu", "tier": "mobile",
        "layout_model": MOBILE_MODELS["layout"],
        "det_model":    MOBILE_MODELS["det"],
        "rec_model":    MOBILE_MODELS["rec"],
        "reason": (
            f"{cpu_reason_prefix}{free_ram:.1f} GB free RAM "
            f"(< {TIER_CPU_SERVER_MIN_RAM_GB:.0f} GB) — "
            "using mobile models on CPU to avoid OOM."
        ),
        "free_vram_gb": free_vram, "free_ram_gb": free_ram,
    }


# ── Resource check ────────────────────────────────────────────────────

def check_system_resources(use_gpu: bool) -> Dict[str, Any]:
    """Inspect GPU VRAM (device 0) and system RAM.

    -> {warnings, gpu_available, gpu_vram_gb, gpu_vram_free_gb,
        available_ram_gb, total_ram_gb}. GPU numbers are 0 when there's no GPU.
    """
    warnings: List[str] = []

    gpu_available    = False
    gpu_vram_gb      = 0.0
    gpu_vram_free_gb = 0.0

    try:
        if _paddle is not None:
            gpu_available = bool(_paddle.device.is_compiled_with_cuda())
        if gpu_available:
            props = _paddle.device.cuda.get_device_properties(0)
            gpu_vram_gb = props.total_memory / (1024 ** 3)
            reserved_bytes = _paddle.device.cuda.memory_reserved(0)
            gpu_vram_free_gb = max(0.0, gpu_vram_gb - reserved_bytes / (1024 ** 3))
    except Exception:
        pass

    if use_gpu:
        if not gpu_available:
            warnings.append(
                "GPU not available in this environment. "
                "Running on CPU instead — this is much slower and requires >= 8 GB RAM."
            )
        elif gpu_vram_gb < MIN_GPU_VRAM_GB:
            warnings.append(
                f"WARNING: GPU VRAM is {gpu_vram_gb:.1f} GB, but a minimum of "
                f"{MIN_GPU_VRAM_GB:.0f} GB is recommended for the server-class models "
                "(PP-DocLayout_plus-L + PP-OCRv5_server_det). "
                "You may experience out-of-memory errors or degraded performance."
            )
        elif gpu_vram_free_gb < 4.0:
            warnings.append(
                f"WARNING: Only {gpu_vram_free_gb:.1f} GB of VRAM is currently free. "
                "Other processes may be consuming GPU memory. "
                "Consider freeing GPU memory before running detection."
            )

    mem = psutil.virtual_memory()
    total_ram_gb     = mem.total     / (1024 ** 3)
    available_ram_gb = mem.available / (1024 ** 3)

    if available_ram_gb < LOW_RAM_GB:
        warnings.append(
            f"CRITICAL: Only {available_ram_gb:.1f} GB of RAM is available "
            f"(total: {total_ram_gb:.1f} GB). "
            "The system may freeze or be killed by the OOM killer during detection. "
            f"At least {MIN_RAM_GB:.0f} GB of free RAM is strongly recommended."
        )
    elif not use_gpu and available_ram_gb < MIN_RAM_GB:
        warnings.append(
            f"WARNING: Running on CPU with only {available_ram_gb:.1f} GB of free RAM. "
            f"At least {MIN_RAM_GB:.0f} GB RAM is recommended for CPU-mode detection "
            "with server-class models. Processing may be very slow."
        )

    return {
        "warnings":          warnings,
        "gpu_available":     gpu_available,
        "gpu_vram_gb":       gpu_vram_gb,
        "gpu_vram_free_gb":  gpu_vram_free_gb,
        "available_ram_gb":  available_ram_gb,
        "total_ram_gb":      total_ram_gb,
    }


def _free_memory():
    """Aggressively free Python + GPU memory."""
    gc.collect()
    try:
        if _paddle is not None and _paddle.device.is_compiled_with_cuda():
            _paddle.device.cuda.empty_cache()
    except Exception:
        pass


# ── Layout helpers ────────────────────────────────────────────────────

def _resize_for_layout(image: np.ndarray,
                       max_side: int = 1500) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    mx = max(h, w)
    if mx <= max_side:
        return image, 1.0
    s = max_side / mx
    return cv2.resize(image, (int(w * s), int(h * s)),
                      interpolation=cv2.INTER_AREA), s


def _box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / float(_box_area(a) + _box_area(b) - inter)


def _suppress_overlapping(layout_boxes, iou_thresh=0.3):
    filtered = []
    for box in layout_boxes:
        keep = True
        for kept in filtered:
            if box["label"] != kept["label"]:
                continue
            if _iou_xyxy(box["bbox"], kept["bbox"]) > iou_thresh:
                if _box_area(box["bbox"]) < _box_area(kept["bbox"]):
                    filtered.remove(kept)
                else:
                    keep = False
                break
        if keep:
            filtered.append(box)
    return filtered


def _remove_margin_boxes(layout_boxes, page_width):
    out = []
    for b in layout_boxes:
        x1, _, x2, _ = b["bbox"]
        cx = (x1 + x2) / 2
        if cx < page_width * 0.12 or cx > page_width * 0.88:
            continue
        out.append(b)
    return out


def _merge_title_blocks(layout_boxes, vertical_thresh=70):
    merged = []
    for label in ("doc_title", "text"):
        boxes = sorted(
            [b for b in layout_boxes if b["label"] == label],
            key=lambda x: x["bbox"][1],
        )
        cur = None
        for b in boxes:
            if cur is None:
                cur = b.copy(); continue
            if b["bbox"][1] - cur["bbox"][3] < vertical_thresh:
                cur["bbox"] = [
                    min(cur["bbox"][0], b["bbox"][0]),
                    min(cur["bbox"][1], b["bbox"][1]),
                    max(cur["bbox"][2], b["bbox"][2]),
                    max(cur["bbox"][3], b["bbox"][3]),
                ]
            else:
                merged.append(cur); cur = b.copy()
        if cur:
            merged.append(cur)
    for b in layout_boxes:
        if b["label"] not in ("doc_title", "text"):
            merged.append(b)
    return merged


# ── Text-line helpers ─────────────────────────────────────────────────

def _poly_bounds(poly):
    pts = np.asarray(poly, dtype=np.float32)
    xmn, ymn = pts[:, 0].min(), pts[:, 1].min()
    xmx, ymx = pts[:, 0].max(), pts[:, 1].max()
    return xmn, ymn, xmx, ymx, (xmn+xmx)*0.5, (ymn+ymx)*0.5, xmx-xmn, ymx-ymn


def _nms(boxes, scores, iou_thresh=NMS_IOU_THRESH):
    if not boxes:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep, suppressed = [], set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        bi = _poly_bounds(boxes[i])[:4]
        for j in order:
            if j in suppressed or j == i:
                continue
            bj = _poly_bounds(boxes[j])[:4]
            if _iou_xyxy(bi, bj) > iou_thresh:
                suppressed.add(j)
    return keep


def _resolve_vertical_overlaps(boxes, thresh=0.30):
    if len(boxes) < 2:
        return boxes
    rects = [
        [float(b[:, 0].min()), float(b[:, 1].min()),
         float(b[:, 0].max()), float(b[:, 1].max())]
        for b in boxes
    ]
    order = sorted(range(len(rects)), key=lambda i: rects[i][1])
    rects = [rects[i] for i in order]
    for i in range(len(rects)):
        yi0, yi1 = rects[i][1], rects[i][3]
        hi = yi1 - yi0
        if hi <= 0:
            continue
        for j in range(i + 1, len(rects)):
            yj0, yj1 = rects[j][1], rects[j][3]
            ov_top, ov_bot = max(yi0, yj0), min(yi1, yj1)
            ov = ov_bot - ov_top
            if ov <= 0:
                break
            hj = yj1 - yj0
            if hj <= 0:
                continue
            if ov / min(hi, hj) > thresh:
                mid = (ov_top + ov_bot) / 2.0
                rects[i][3] = mid
                rects[j][1] = mid
                yi1 = mid
    result = []
    for r in rects:
        xmn, ymn, xmx, ymx = r
        if xmx > xmn and ymx > ymn:
            result.append(np.array(
                [[xmn, ymn], [xmx, ymn], [xmx, ymx], [xmn, ymx]],
                dtype=np.float32,
            ))
    return result


def _merge_into_lines(raw_boxes, img_w, img_h, gap_multiplier=GAP_MULTIPLIER):
    if not raw_boxes:
        return []
    bounds = [_poly_bounds(b) for b in raw_boxes]
    heights = np.array([b[7] for b in bounds], dtype=np.float32)
    widths  = np.array([b[6] for b in bounds], dtype=np.float32)
    med_h = float(np.median(heights))
    med_w = float(np.median(widths))
    h_thresh  = 0.5 * med_h
    gap_limit = gap_multiplier * med_w
    MIN_W, MIN_H, MAX_H = 10.0, 10.0, 6.0 * med_h

    print(f"[_merge_into_lines] {len(raw_boxes)} raw boxes | "
          f"med_h={med_h:.1f} med_w={med_w:.1f} | "
          f"h_thresh={h_thresh:.1f} gap_limit={gap_limit:.1f} MAX_H={MAX_H:.1f} | "
          f"img_w={img_w} img_h={img_h}")
    print(f"[_merge_into_lines] height stats: "
          f"min={heights.min():.1f} p25={float(np.percentile(heights,25)):.1f} "
          f"p50={med_h:.1f} p75={float(np.percentile(heights,75)):.1f} max={heights.max():.1f}")
    print(f"[_merge_into_lines] sample box bounds (first 3): "
          f"{[b[:4] for b in bounds[:3]]}")

    filtered = [
        (b, bnd) for b, bnd in zip(raw_boxes, bounds)
        if bnd[6] >= MIN_W and MIN_H <= bnd[7] <= MAX_H
    ]
    print(f"[_merge_into_lines] after h/w filter: {len(filtered)} boxes remain")
    if not filtered:
        return []
    filtered.sort(key=lambda x: x[1][5])

    lines = []
    for box, bnd in filtered:
        cy = bnd[5]
        assigned = False
        for line in lines:
            line_cy = sum(b[1][5] for b in line) / len(line)
            if abs(cy - line_cy) < h_thresh:
                line.append((box, bnd))
                assigned = True
                break
        if not assigned:
            lines.append([(box, bnd)])

    merged = []
    for line in lines:
        line.sort(key=lambda x: x[1][0])
        groups = [[line[0]]]
        for item in line[1:]:
            if item[1][0] - groups[-1][-1][1][2] <= gap_limit:
                groups[-1].append(item)
            else:
                groups.append([item])
        for grp in groups:
            xmn = max(0, min(b[1][0] for b in grp))
            ymn = max(0, min(b[1][1] for b in grp))
            xmx = min(img_w, max(b[1][2] for b in grp))
            ymx = min(img_h, max(b[1][3] for b in grp))
            if xmx > xmn and ymx > ymn:
                merged.append(np.array(
                    [[xmn, ymn], [xmx, ymn], [xmx, ymx], [xmn, ymx]],
                    dtype=np.float32,
                ))

    merged = _resolve_vertical_overlaps(merged)
    merged.sort(key=lambda b: (float(b[:, 1].min()), float(b[:, 0].min())))
    return merged


def _filter_boxes_by_page_size(boxes, img_h, img_w):
    """Drop boxes too small to be a text line, scaled to the median box height.

    Thresholds used to come from page area, which threw away every real line on
    big scans (6048x4536) because page_area * 0.0001 dwarfed a line box.
    """
    if not boxes:
        return []

    heights = np.array(
        [float(np.asarray(b, dtype=np.float32)[:, 1].max()
               - np.asarray(b, dtype=np.float32)[:, 1].min())
         for b in boxes],
        dtype=np.float32,
    )
    median_height = float(np.median(heights))

    min_area   = max(20.0,  median_height * median_height * 0.5)
    min_width  = max(5.0,   median_height * 0.5)
    min_height = max(5.0,   median_height * 0.5)

    print(f"[_filter_boxes_by_page_size] img_h={img_h} img_w={img_w} "
          f"boxes={len(boxes)} median_height={median_height:.1f} "
          f"min_area={min_area:.1f} min_width={min_width:.1f} "
          f"min_height={min_height:.1f}")

    kept = []
    for i, box in enumerate(boxes):
        pts  = np.asarray(box, dtype=np.float32)
        w    = float(pts[:, 0].max() - pts[:, 0].min())
        h    = float(pts[:, 1].max() - pts[:, 1].min())
        area = w * h
        print(f"  BOX BEFORE FILTER: width={w:.1f} height={h:.1f} area={area:.1f}")
        keep = (w >= min_width) and (h >= min_height) and (area >= min_area)
        print(f"  box[{i}]: x=[{pts[:,0].min():.0f},{pts[:,0].max():.0f}] "
              f"y=[{pts[:,1].min():.0f},{pts[:,1].max():.0f}] "
              f"w={w:.1f} h={h:.1f} area={area:.1f} "
              f"({'KEEP' if keep else 'DROP'})")
        if keep:
            kept.append(box)
    return kept


# ── Debug visualisation ───────────────────────────────────────────────

def _save_debug_image(
    image: np.ndarray,
    boxes,
    path: str,
    color=(0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw boxes on a copy of the image and write it to path.

    Accepts either 4-point polygons or dicts with a 'bbox' [x1,y1,x2,y2] key.
    """
    dir_part = os.path.dirname(path)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)

    vis = image.copy()
    for box in boxes:
        if isinstance(box, dict):
            x1, y1, x2, y2 = [int(c) for c in box["bbox"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        else:
            pts = np.asarray(box, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, color, thickness)
    cv2.imwrite(path, vis)
    print(f"[debug] saved {path} ({len(boxes)} boxes)")


# ── Core detection ────────────────────────────────────────────────────

def detect_layout(image: np.ndarray, device: str = "gpu",
                  max_layout_side: int = 1500,
                  model_name: str = "PP-DocLayout_plus-L") -> list:
    """Run layout detection and return cleaned layout boxes."""
    resized, scale = _resize_for_layout(image, max_layout_side)

    print(f"[detect_layout] image={image.shape}, resized={resized.shape}, "
          f"scale={scale:.3f}, model={model_name}, device={device}")

    _ensure_paddle()
    pipeline = None
    try:
        _free_memory()  # drop stale allocations before loading a multi-GB model
        pipeline = _create_model(model_name=model_name, device=device)
        results = pipeline.predict(resized, batch_size=1)
    except (MemoryError, RuntimeError) as exc:
        # Paddle reports GPU OOM as a plain RuntimeError.
        _free_memory()
        if pipeline is not None:
            try:
                del pipeline
            except Exception:
                pass
        raise RuntimeError(
            f"Out-of-memory during layout detection ({device.upper()}). "
            "Ensure at least 8 GB GPU VRAM is available for GPU mode, "
            "or at least 8 GB free RAM for CPU mode. "
            f"Original error: {exc}"
        ) from exc

    layout_boxes = []
    for res in results:
        for b in res["boxes"]:
            label = b["label"]
            score = float(b["score"])
            x1, y1, x2, y2 = b["coordinate"]
            layout_boxes.append({
                "label": label,
                "bbox": [int(x1 / scale), int(y1 / scale),
                         int(x2 / scale), int(y2 / scale)],
                "score": score,
            })

    print(f"[detect_layout] raw layout boxes: {len(layout_boxes)}")
    for b in layout_boxes:
        print(f"  {b['label']:20s}  score={b['score']:.2f}  bbox={b['bbox']}")

    layout_boxes = _suppress_overlapping(layout_boxes)
    layout_boxes = _remove_margin_boxes(layout_boxes, image.shape[1])
    layout_boxes = _merge_title_blocks(layout_boxes)

    print(f"[detect_layout] after post-processing: {len(layout_boxes)} boxes")

    del pipeline, results
    _free_memory()

    return layout_boxes


def detect_text_lines(image: np.ndarray, layout: list,
                      device: str = "gpu",
                      det_model_name: str = "PP-OCRv5_server_det",
                      rec_model_name: str = "PP-OCRv5_server_rec",
                      region_padding: int = REGION_PADDING,
                      layout_expand: int = LAYOUT_EXPAND,
                      score_thresh: float = SCORE_THRESH,
                      upscale_min_h: int = UPSCALE_MIN_H,
                      nms_iou_thresh: float = NMS_IOU_THRESH,
                      gap_multiplier: float = GAP_MULTIPLIER,
                      debug_dir: str = "") -> list:
    """Detect line-level bounding boxes for all text regions."""

    print(f"[detect_text_lines] image={image.shape}, "
          f"{len(layout)} layout regions, det_model={det_model_name}, "
          f"rec_model={rec_model_name}")

    _ensure_paddle()
    ocr = None
    try:
        _free_memory()
        ocr = _PaddleOCR(
            text_detection_model_name=det_model_name,
            text_recognition_model_name=rec_model_name,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
            lang="en",
        )
    except (MemoryError, RuntimeError) as exc:
        _free_memory()
        raise RuntimeError(
            f"Out-of-memory while loading text-detection model ({device.upper()}). "
            "Ensure at least 8 GB GPU VRAM is available for GPU mode, "
            "or at least 8 GB free RAM for CPU mode. "
            f"Original error: {exc}"
        ) from exc

    img_h, img_w = image.shape[:2]
    all_raw, all_scores = [], []

    for region in layout:
        if region["label"] not in TEXT_LABELS:
            continue
        x1, y1, x2, y2 = [int(c) for c in region["bbox"]]
        x1e = max(0, x1 - layout_expand)
        y1e = max(0, y1 - layout_expand)
        x2e = min(img_w, x2 + layout_expand)
        y2e = min(img_h, y2 + layout_expand)
        if x2e <= x1e or y2e <= y1e:
            continue

        crop = image[y1e:y2e, x1e:x2e]
        if crop.size == 0:
            continue

        sc = 1.0
        if crop.shape[0] < upscale_min_h:
            sc = 2.0
            crop = cv2.resize(crop, None, fx=sc, fy=sc,
                              interpolation=cv2.INTER_CUBIC)

        pad_scaled = int(region_padding * sc)
        padded = cv2.copyMakeBorder(
            crop, pad_scaled, pad_scaled, pad_scaled, pad_scaled,
            borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
        del crop

        ox = x1e - region_padding
        oy = y1e - region_padding

        try:
            crop_results = ocr.predict(padded)
        except (MemoryError, RuntimeError) as exc:
            del padded
            if ocr is not None:
                try:
                    del ocr
                except Exception:
                    pass
            _free_memory()
            raise RuntimeError(
                f"Out-of-memory during text-line detection on region {region['bbox']} "
                f"({device.upper()}). "
                "Ensure at least 8 GB GPU VRAM (GPU mode) or 8 GB free RAM (CPU mode). "
                f"Original error: {exc}"
            ) from exc
        del padded

        n = 0
        _logged_raw = 0
        for res in crop_results:
            polys  = res.get("dt_polys",  [])
            scores = res.get("dt_scores", [1.0] * len(polys))
            for poly, score in zip(polys, scores):
                if score < score_thresh:
                    continue
                arr = np.array(poly, dtype=np.float32)
                if _logged_raw < 2:
                    print(f"    [raw poly {_logged_raw}] shape={arr.shape} "
                          f"raw_coords={arr.tolist()} sc={sc} ox={ox} oy={oy}")
                arr /= sc
                arr[:, 0] += ox
                arr[:, 1] += oy
                if _logged_raw < 2:
                    print(f"    [raw poly {_logged_raw}] after_transform: "
                          f"x=[{arr[:,0].min():.0f},{arr[:,0].max():.0f}] "
                          f"y=[{arr[:,1].min():.0f},{arr[:,1].max():.0f}]")
                    _logged_raw += 1
                if arr[:, 0].max() < x1e or arr[:, 0].min() > x2e:
                    continue
                if arr[:, 1].max() < y1e or arr[:, 1].min() > y2e:
                    continue
                all_raw.append(arr)
                all_scores.append(float(score))
                n += 1
        del crop_results

        print(f"  [{region['label']}]  bbox={region['bbox']}  "
              f"scale={sc:.1f}x  ->  {n} raw boxes")

    print(f"[detect_text_lines] total raw boxes: {len(all_raw)}")

    if debug_dir:
        _save_debug_image(
            image, all_raw,
            os.path.join(debug_dir, "page_raw_text_boxes.png"),
            color=(0, 0, 255),
        )

    keep_idx = _nms(all_raw, all_scores, iou_thresh=nms_iou_thresh)
    all_raw = [all_raw[i] for i in keep_idx]
    print(f"[detect_text_lines] after NMS: {len(all_raw)}")

    if debug_dir:
        _save_debug_image(
            image, all_raw,
            os.path.join(debug_dir, "page_after_nms.png"),
            color=(0, 165, 255),
        )

    if all_raw:
        print(f"[detect_text_lines] coordinate sanity: "
              f"img_w={img_w} img_h={img_h}")
        for idx, b in enumerate(all_raw[:5]):
            pts = np.asarray(b, dtype=np.float32)
            bw = float(pts[:, 0].max() - pts[:, 0].min())
            bh = float(pts[:, 1].max() - pts[:, 1].min())
            print(f"  [pre-merge box {idx}] "
                  f"x=[{pts[:,0].min():.0f},{pts[:,0].max():.0f}] "
                  f"y=[{pts[:,1].min():.0f},{pts[:,1].max():.0f}] "
                  f"w={bw:.1f} h={bh:.1f} area={bw*bh:.1f}")

    merged = _merge_into_lines(all_raw, img_w, img_h, gap_multiplier=gap_multiplier)
    del all_raw

    before = len(merged)
    merged = _filter_boxes_by_page_size(merged, img_h, img_w)
    print(f"[detect_text_lines] after page filter: {len(merged)} "
          f"(removed {before - len(merged)})")

    if debug_dir:
        _save_debug_image(
            image, merged,
            os.path.join(debug_dir, "page_final_lines.png"),
            color=(0, 255, 0),
        )

    del ocr
    _free_memory()

    return merged


# ── Public orchestrator ───────────────────────────────────────────────

def run_layout_aware_detection(
    image: np.ndarray,
    use_gpu: bool = False,
    layout_model_name: str = "PP-DocLayout_plus-L",
    det_model_name: str = "PP-OCRv5_server_det",
    rec_model_name: str = "PP-OCRv5_server_rec",
    region_padding: int = REGION_PADDING,
    layout_expand: int = LAYOUT_EXPAND,
    score_thresh: float = SCORE_THRESH,
    upscale_min_h: int = UPSCALE_MIN_H,
    nms_iou_thresh: float = NMS_IOU_THRESH,
    gap_multiplier: float = GAP_MULTIPLIER,
    debug_dir: str = "",
) -> Tuple[List[List[List[float]]], List[str]]:
    """Layout detection then text-line detection.

    -> (4-point polygons in page coordinates, resource warnings). Show the
    warnings to the user — they explain why a run is slow or about to OOM.
    """
    # Silently drop to CPU rather than fail if paddle can't use the GPU.
    cuda_compiled = False
    try:
        _ensure_paddle()
        cuda_compiled = bool(_paddle.device.is_compiled_with_cuda())
    except RuntimeError as _load_err:
        if use_gpu:
            use_gpu = False
            print(f"[run_layout_aware_detection] Paddle failed to load — "
                  f"forcing CPU (will likely be slow): {_load_err}")

    if use_gpu and not cuda_compiled:
        use_gpu = False
        print(
            "[run_layout_aware_detection] WARNING: use_gpu=True requested but "
            "PaddlePaddle is not compiled with CUDA. "
            "Falling back to CPU automatically."
        )

    device = "gpu" if use_gpu else "cpu"

    res_info = check_system_resources(use_gpu)
    resource_warnings: List[str] = res_info["warnings"]

    # Goes first — it explains every other warning below it.
    if not cuda_compiled and device == "cpu":
        cuda_warning = (
            "GPU was requested but PaddlePaddle in this environment is CPU-only "
            "(not compiled with CUDA). Running on CPU instead. "
            "To enable GPU acceleration, CUDA 12.6 driver on the host is required "
            "and the container must be started with --gpus all. Install GPU paddle: "
            "pip install paddlepaddle-gpu==3.3.0 "
            "-i https://www.paddlepaddle.org.cn/packages/stable/cu126/"
        )
        resource_warnings.insert(0, cuda_warning)

    print(f"\n{'='*60}")
    print(f"[run_layout_aware_detection] START")
    print(f"  image shape : {image.shape}")
    print(f"  device      : {device}  (cuda_compiled={cuda_compiled})")
    print(f"  layout_model: {layout_model_name}")
    print(f"  det_model   : {det_model_name}")
    print(f"  rec_model   : {rec_model_name}")
    print(f"  region_padding : {region_padding}")
    print(f"  layout_expand  : {layout_expand}")
    print(f"  score_thresh   : {score_thresh}")
    print(f"  upscale_min_h  : {upscale_min_h}")
    print(f"  nms_iou_thresh : {nms_iou_thresh}")
    print(f"  gap_multiplier : {gap_multiplier}")
    print(f"  gpu_available  : {res_info['gpu_available']}  "
          f"vram={res_info['gpu_vram_gb']:.1f} GB  free={res_info['gpu_vram_free_gb']:.1f} GB")
    print(f"  ram_available  : {res_info['available_ram_gb']:.1f} GB  "
          f"total={res_info['total_ram_gb']:.1f} GB")
    for w in resource_warnings:
        print(f"  [RESOURCE WARNING] {w}")
    print(f"{'='*60}")

    t0 = time.time()

    layout = detect_layout(image, device=device, max_layout_side=1500,
                           model_name=layout_model_name)

    if not layout:
        print("[run_layout_aware_detection] No layout regions found!")
        return [], resource_warnings

    if debug_dir:
        _save_debug_image(
            image, layout,
            os.path.join(debug_dir, "page_layout_boxes.png"),
            color=(255, 0, 0),
        )

    text_regions = [r for r in layout if r["label"] in TEXT_LABELS]
    print(f"\n[run_layout_aware_detection] "
          f"{len(text_regions)} text regions / {len(layout)} total")

    if not text_regions:
        print("[run_layout_aware_detection] No TEXT regions in layout!")
        return [], resource_warnings

    merged = detect_text_lines(image, layout, device=device,
                               det_model_name=det_model_name,
                               rec_model_name=rec_model_name,
                               region_padding=region_padding,
                               layout_expand=layout_expand,
                               score_thresh=score_thresh,
                               upscale_min_h=upscale_min_h,
                               nms_iou_thresh=nms_iou_thresh,
                               gap_multiplier=gap_multiplier,
                               debug_dir=debug_dir)

    lines = [box.tolist() for box in merged]

    elapsed = time.time() - t0
    print(f"\n[run_layout_aware_detection] DONE — "
          f"{len(lines)} lines in {elapsed:.1f}s")

    _free_memory()

    return lines, resource_warnings

