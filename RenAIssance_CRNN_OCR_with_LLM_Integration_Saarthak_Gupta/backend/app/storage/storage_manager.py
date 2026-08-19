"""Storage manager for persistent transcripts and datasets."""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.config import STORAGE_ROOT
from ..services.dataset_builder import align_boxes_with_transcript, crop_line_image
from .file_indexer import ensure_dir, next_numeric_id, read_metadata, safe_rmtree


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_label(text: str, max_len: int = 64) -> str:
    safe = re.sub(r"[^A-Za-z0-9\s_-]", "", text or "")
    safe = re.sub(r"\s+", "_", safe.strip())
    return safe[:max_len] if safe else "line"


def _safe_name(text: str, fallback: str = "session") -> str:
    safe = re.sub(r"[^A-Za-z0-9\s_-]", "", text or "")
    safe = re.sub(r"\s+", " ", safe).strip()
    return safe[:96] if safe else fallback


def _page_sort_key(page_key: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", str(page_key))
    if match:
        return int(match.group(1)), str(page_key)
    return 10**9, str(page_key)


def _decode_data_url(image_data: str) -> np.ndarray | None:
    payload = image_data.split(",", 1)[1] if "," in image_data else image_data
    raw = base64.b64decode(payload)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _image_ext_from_data_url(image_data: str) -> str:
    match = re.match(r"^data:image/([a-zA-Z0-9.+-]+);base64,", image_data or "")
    if not match:
        return "png"
    subtype = match.group(1).lower()
    if subtype in {"jpg", "jpeg"}:
        return "jpg"
    if subtype in {"png", "webp"}:
        return subtype
    return "png"


def _decode_data_url_bytes(image_data: str) -> bytes | None:
    if not image_data:
        return None
    try:
        payload = image_data.split(",", 1)[1] if "," in image_data else image_data
        return base64.b64decode(payload)
    except Exception:
        return None


def _guess_image_mime_from_suffix(suffix: str) -> str:
    ext = suffix.lower().lstrip(".")
    if ext in {"jpg", "jpeg"}:
        return "image/jpeg"
    if ext == "webp":
        return "image/webp"
    return "image/png"


def get_storage_paths() -> dict[str, Path]:
    root = ensure_dir(Path(STORAGE_ROOT))
    transcripts = ensure_dir(root / "transcripts")
    datasets = ensure_dir(root / "datasets")
    return {
        "root": root,
        "transcripts": transcripts,
        "datasets": datasets,
    }


def _build_tags(
    model_info: dict[str, Any] | None,
    mode: str | None = None,
    fmt: str | None = None,
) -> list[str]:
    """
    Derive short, human-readable chips from the pipeline/model info so the
    My Files UI can show at a glance what produced an entry: which
    preprocessing ran, the detection / layout / OCR models, and whether LLM
    post-processing was applied.
    """
    info = model_info or {}
    tags: list[str] = []

    if mode:
        tags.append(str(mode))

    pre = info.get("preprocessing")
    if isinstance(pre, (list, tuple)) and pre:
        shown = ", ".join(str(p) for p in pre[:4])
        if len(pre) > 4:
            shown += f" +{len(pre) - 4}"
        tags.append(f"Preprocess: {shown}")
    elif pre in (None, [], (), ""):
        tags.append("Preprocess: none")

    if info.get("detection_model"):
        tags.append(f"Detect: {info['detection_model']}")
    if info.get("layout_model"):
        tags.append(f"Layout: {info['layout_model']}")

    if info.get("ocr_model") or info.get("ocr_provider"):
        provider = info.get("ocr_provider") or "ocr"
        ocr_model = info.get("ocr_model") or "model"
        tags.append(f"OCR: {provider}/{ocr_model}")

    llm = info.get("llm_postprocess")
    if isinstance(llm, dict) and llm.get("used"):
        provider = llm.get("provider") or "llm"
        llm_model = llm.get("model") or "model"
        tags.append(f"LLM: {provider}/{llm_model}")
    elif isinstance(llm, dict):
        tags.append("LLM: none")

    if fmt:
        tags.append(f"Format: {fmt}")

    return tags


def save_transcript_session(
    transcripts: dict[str, str],
    source: str = "ocr upload",
    mode: str = "recognition",
    transcript_images: dict[str, str] | None = None,
    book_name: str | None = None,
    model_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = get_storage_paths()
    session_id = next_numeric_id(paths["transcripts"], "session")
    session_dir = ensure_dir(paths["transcripts"] / session_id)

    written_pages = 0
    written_images = 0
    sorted_pages = sorted(transcripts.keys(), key=_page_sort_key)
    image_map = transcript_images or {}

    for page_key in sorted_pages:
        text = (transcripts.get(page_key) or "").strip()
        if not text:
            continue
        match = re.search(r"(\d+)", str(page_key))
        if match:
            page_name = str(int(match.group(1)))
        else:
            page_name = re.sub(r"[^A-Za-z0-9_-]", "_", str(page_key))
        (session_dir / f"page{page_name}.txt").write_text(text, encoding="utf-8")

        image_data = image_map.get(page_key) or image_map.get(str(page_key))
        image_bytes = _decode_data_url_bytes(image_data or "")
        if image_bytes:
            ext = _image_ext_from_data_url(image_data or "")
            (session_dir / f"page{page_name}.{ext}").write_bytes(image_bytes)
            written_images += 1

        written_pages += 1

    display_name = _safe_name(book_name or "", fallback=session_id)

    metadata = {
        "id": session_id,
        "type": "transcript",
        "created_at": _utc_now_iso(),
        "num_files": written_pages,
        "num_pages": written_pages,
        "num_images": written_images,
        "source": source,
        "mode": mode,
        "name": display_name,
        "book_name": display_name,
        "model_info": model_info or {},
        "tags": _build_tags(model_info, mode),
    }
    (session_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata


def save_recognition_dataset(
    pages_data: list[dict[str, Any]],
    source: str,
    book_name: str,
    model_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = get_storage_paths()
    dataset_id = next_numeric_id(paths["datasets"], "dataset")
    dataset_dir = ensure_dir(paths["datasets"] / dataset_id)
    images_dir = ensure_dir(dataset_dir / "images")
    labels_dir = ensure_dir(dataset_dir / "labels")

    csv_rows: list[tuple[str, str]] = []
    total_samples = 0

    for page in pages_data:
        page_key = str(page.get("page_key", "unknown"))
        boxes = page.get("boxes", [])
        lines = page.get("lines", [])
        img = _decode_data_url(page.get("image_data", ""))
        if img is None:
            continue

        aligned, _, _ = align_boxes_with_transcript(boxes, lines)
        seen_labels: dict[str, int] = {}

        for idx, (box, text) in enumerate(aligned, start=1):
            crop = crop_line_image(img, box)
            label = _safe_label(text)
            seen_labels[label] = seen_labels.get(label, 0) + 1
            suffix = seen_labels[label]
            sample_name = f"page_{page_key}_line_{idx:04d}_{label}_{suffix}"
            image_name = f"{sample_name}.png"

            ok, png_data = cv2.imencode(".png", crop)
            if not ok:
                continue

            (images_dir / image_name).write_bytes(png_data.tobytes())
            csv_rows.append((f"images/{image_name}", text))
            total_samples += 1

    labels_csv = io.StringIO()
    writer = csv.writer(labels_csv)
    writer.writerow(["image", "text"])
    for image_path, text in csv_rows:
        writer.writerow([image_path, text])
    (labels_dir / "labels.csv").write_text(labels_csv.getvalue(), encoding="utf-8")

    metadata = {
        "id": dataset_id,
        "type": "dataset",
        "created_at": _utc_now_iso(),
        "num_files": total_samples,
        "num_samples": total_samples,
        "source": source,
        "mode": "recognition",
        "dataset_type": "recognition",
        "format": "png+csv",
        "book_name": book_name,
        "model_info": model_info or {},
        "tags": _build_tags(model_info, "recognition"),
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata


def _polygon_to_xyxy(box: list[list[float]]) -> tuple[int, int, int, int]:
    pts = np.asarray(box, dtype=np.float32)
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )


def save_detection_dataset(
    pages_data: list[dict[str, Any]],
    source: str,
    book_name: str,
    bbox_format: str = "txt",
    model_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fmt = (bbox_format or "txt").lower()
    if fmt not in {"txt", "json", "yolo", "coco"}:
        fmt = "txt"

    paths = get_storage_paths()
    dataset_id = next_numeric_id(paths["datasets"], "dataset")
    dataset_dir = ensure_dir(paths["datasets"] / dataset_id)
    images_dir = ensure_dir(dataset_dir / "images")
    bboxes_dir = ensure_dir(dataset_dir / ("labels" if fmt == "yolo" else "bboxes"))

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    coco_image_id = 0
    coco_ann_id = 0
    total_samples = 0

    for page in pages_data:
        page_key = str(page.get("page_key", "unknown"))
        boxes = page.get("boxes", [])
        img = _decode_data_url(page.get("image_data", ""))
        if img is None:
            continue

        h, w = img.shape[:2]
        stem = f"page_{page_key}"
        image_name = f"{stem}.jpg"
        ok, jpg_data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            (images_dir / image_name).write_bytes(jpg_data.tobytes())

        xyxy_boxes = [_polygon_to_xyxy(b) for b in boxes]
        total_samples += len(xyxy_boxes)

        if fmt == "txt":
            lines = "\n".join(f"{x1} {y1} {x2} {y2}" for (x1, y1, x2, y2) in xyxy_boxes)
            (bboxes_dir / f"{stem}.txt").write_text((lines + "\n") if lines else "", encoding="utf-8")
        elif fmt == "json":
            (bboxes_dir / f"{stem}.json").write_text(
                json.dumps([[x1, y1, x2, y2] for (x1, y1, x2, y2) in xyxy_boxes], indent=2),
                encoding="utf-8",
            )
        elif fmt == "yolo":
            yolo_lines = []
            for (x1, y1, x2, y2) in xyxy_boxes:
                bw = max(0, x2 - x1)
                bh = max(0, y2 - y1)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                yolo_lines.append(f"0 {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            (bboxes_dir / f"{stem}.txt").write_text(("\n".join(yolo_lines) + "\n") if yolo_lines else "", encoding="utf-8")
        elif fmt == "coco":
            coco_image_id += 1
            coco_images.append({
                "id": coco_image_id,
                "file_name": image_name,
                "width": int(w),
                "height": int(h),
            })
            for (x1, y1, x2, y2) in xyxy_boxes:
                coco_ann_id += 1
                bw = max(0, x2 - x1)
                bh = max(0, y2 - y1)
                coco_annotations.append({
                    "id": coco_ann_id,
                    "image_id": coco_image_id,
                    "category_id": 1,
                    "bbox": [int(x1), int(y1), int(bw), int(bh)],
                    "area": int(bw * bh),
                    "iscrowd": 0,
                    "segmentation": [],
                })

    if fmt == "yolo":
        (dataset_dir / "classes.txt").write_text("text\n", encoding="utf-8")
    elif fmt == "coco":
        (bboxes_dir / "annotations.json").write_text(
            json.dumps(
                {
                    "images": coco_images,
                    "annotations": coco_annotations,
                    "categories": [{"id": 1, "name": "text", "supercategory": "text"}],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    metadata = {
        "id": dataset_id,
        "type": "dataset",
        "created_at": _utc_now_iso(),
        "num_files": total_samples,
        "num_samples": total_samples,
        "source": source,
        "mode": "detection",
        "dataset_type": "detection",
        "format": fmt,
        "book_name": book_name,
        "model_info": model_info or {},
        "tags": _build_tags(model_info, "detection", fmt),
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata


def list_entries(kind: str) -> list[dict[str, Any]]:
    paths = get_storage_paths()
    if kind not in {"transcripts", "datasets"}:
        return []

    entries = []
    for child in paths[kind].iterdir():
        if not child.is_dir():
            continue
        metadata = read_metadata(child)
        if not metadata:
            continue
        metadata["name"] = metadata.get("name") or child.name
        metadata["id"] = metadata.get("id") or child.name
        entries.append(metadata)

    entries.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return entries


def get_transcript_detail(session_id: str) -> dict[str, Any]:
    paths = get_storage_paths()
    session_dir = paths["transcripts"] / session_id
    if not session_dir.exists() or not session_dir.is_dir():
        raise FileNotFoundError(session_id)

    metadata = read_metadata(session_dir)
    pages = []
    for page_file in sorted(session_dir.glob("page*.txt")):
        image_data = None
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            image_file = session_dir / f"{page_file.stem}{ext}"
            if image_file.exists() and image_file.is_file():
                mime = _guess_image_mime_from_suffix(image_file.suffix)
                payload = base64.b64encode(image_file.read_bytes()).decode("utf-8")
                image_data = f"data:{mime};base64,{payload}"
                break

        pages.append({
            "name": page_file.name,
            "content": page_file.read_text(encoding="utf-8"),
            "image_data": image_data,
        })

    return {
        "metadata": metadata,
        "pages": pages,
    }


def get_dataset_detail(dataset_id: str, preview_limit: int = 18) -> dict[str, Any]:
    paths = get_storage_paths()
    dataset_dir = paths["datasets"] / dataset_id
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_id)

    metadata = read_metadata(dataset_dir)
    if not metadata:
        metadata = {"id": dataset_id}

    dataset_type = metadata.get("dataset_type") or metadata.get("mode") or "recognition"
    samples: list[dict[str, Any]] = []

    def _to_data_url(image_path: Path) -> str | None:
        if not image_path.exists() or not image_path.is_file():
            return None
        mime = _guess_image_mime_from_suffix(image_path.suffix)
        payload = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{payload}"

    if dataset_type == "recognition":
        labels_csv = dataset_dir / "labels" / "labels.csv"
        if labels_csv.exists() and labels_csv.is_file():
            with labels_csv.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if len(samples) >= preview_limit:
                        break
                    image_rel = (row.get("image") or "").strip()
                    text = (row.get("text") or "").strip()
                    image_path = dataset_dir / image_rel if image_rel else None
                    image_data = _to_data_url(image_path) if image_path else None
                    samples.append(
                        {
                            "name": Path(image_rel).name if image_rel else "sample",
                            "image_data": image_data,
                            "text": text,
                            "annotation": text,
                        }
                    )
    else:
        images_dir = dataset_dir / "images"
        fmt = (metadata.get("format") or "txt").lower()
        annotation_by_stem: dict[str, str] = {}

        if fmt in {"txt", "yolo"}:
            labels_dir = dataset_dir / ("labels" if fmt == "yolo" else "bboxes")
            for label_file in labels_dir.glob("page_*.txt"):
                try:
                    text = label_file.read_text(encoding="utf-8").strip()
                    lines = [line for line in text.splitlines() if line.strip()]
                    if not lines:
                        annotation_by_stem[label_file.stem] = "No boxes"
                    else:
                        annotation_by_stem[label_file.stem] = "\n".join(lines)
                except Exception:
                    annotation_by_stem[label_file.stem] = "boxes available"
        elif fmt == "json":
            labels_dir = dataset_dir / "bboxes"
            for label_file in labels_dir.glob("page_*.json"):
                try:
                    arr = json.loads(label_file.read_text(encoding="utf-8"))
                    if not isinstance(arr, list) or not arr:
                        annotation_by_stem[label_file.stem] = "No boxes"
                    else:
                        annotation_by_stem[label_file.stem] = "\n".join(
                            [f"[{', '.join(map(str, box))}]" for box in arr]
                        )
                except Exception:
                    annotation_by_stem[label_file.stem] = "boxes available"
        elif fmt == "coco":
            ann_file = dataset_dir / "bboxes" / "annotations.json"
            if ann_file.exists() and ann_file.is_file():
                try:
                    coco = json.loads(ann_file.read_text(encoding="utf-8"))
                    image_id_to_name = {
                        item.get("id"): Path(item.get("file_name", "")).stem
                        for item in coco.get("images", [])
                    }
                    values: dict[str, list[str]] = {}
                    for ann in coco.get("annotations", []):
                        stem = image_id_to_name.get(ann.get("image_id"))
                        if not stem:
                            continue
                        bbox = ann.get("bbox", [])
                        if isinstance(bbox, list) and len(bbox) == 4:
                            bbox_line = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
                        else:
                            bbox_line = str(bbox)
                        if stem not in values:
                            values[stem] = []
                        values[stem].append(bbox_line)
                    annotation_by_stem = {
                        stem: ("\n".join(lines) if lines else "No boxes")
                        for stem, lines in values.items()
                    }
                except Exception:
                    annotation_by_stem = {}

        image_files = sorted(images_dir.glob("page_*.*")) if images_dir.exists() else []
        for image_file in image_files:
            if len(samples) >= preview_limit:
                break
            image_data = _to_data_url(image_file)
            samples.append(
                {
                    "name": image_file.name,
                    "image_data": image_data,
                    "text": "",
                    "annotation": annotation_by_stem.get(image_file.stem, "annotation available"),
                }
            )

    return {
        "metadata": metadata,
        "samples": samples,
    }


def zip_entry(kind: str, entry_id: str) -> tuple[io.BytesIO, str]:
    paths = get_storage_paths()
    if kind not in {"transcripts", "datasets"}:
        raise FileNotFoundError(kind)

    root_dir = paths[kind]
    entry_dir = root_dir / entry_id
    if not entry_dir.exists() or not entry_dir.is_dir():
        raise FileNotFoundError(entry_id)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if kind == "transcripts":
            base_prefix = Path(entry_id)
            for file_path in sorted(entry_dir.glob("page*.txt")):
                if file_path.is_file():
                    arcname = base_prefix / "transcripts" / file_path.name
                    zf.write(file_path, arcname=str(arcname))

            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                for image_file in sorted(entry_dir.glob(f"page*{ext[1:]}")):
                    if image_file.is_file():
                        arcname = base_prefix / "images" / image_file.name
                        zf.write(image_file, arcname=str(arcname))

            metadata_file = entry_dir / "metadata.json"
            if metadata_file.exists() and metadata_file.is_file():
                zf.write(metadata_file, arcname=str(base_prefix / "metadata.json"))
        else:
            for file_path in entry_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(root_dir)
                    zf.write(file_path, arcname=str(arcname))

    buf.seek(0)
    return buf, f"{entry_id}.zip"


def delete_entry(kind: str, entry_id: str) -> bool:
    paths = get_storage_paths()
    if kind not in {"transcripts", "datasets"}:
        return False

    entry_dir = paths[kind] / entry_id
    if not entry_dir.exists() or not entry_dir.is_dir():
        return False

    safe_rmtree(entry_dir)
    return True


def ensure_storage_layout() -> dict[str, str]:
    paths = get_storage_paths()
    return {k: str(v) for k, v in paths.items()}


def resolve_storage_root() -> str:
    return os.fspath(Path(STORAGE_ROOT))
