"""Build line-level OCR training datasets as ZIPs.

Boxes get sorted into reading order, paired with transcript lines, cropped, and
zipped. One page in memory at a time — a book's worth of full-res scans won't fit.
"""

import csv
import io
import re
import zipfile
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


# ── Alignment ───────────────────────────────────────────────────────

def _box_centroid(box: List[List[float]]) -> Tuple[float, float]:
    """Compute centroid of a 4-point polygon."""
    pts = np.array(box, dtype=np.float32)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def sort_boxes_reading_order(boxes: List[List[List[float]]]) -> List[int]:
    """Indices of `boxes` in reading order: top to bottom, left to right."""
    if not boxes:
        return []

    centroids = [_box_centroid(b) for b in boxes]

    ys = np.array([c[1] for c in centroids])
    if len(ys) == 0:
        return []

    # Row threshold scales with the typical line height, so it survives both
    # thumbnail-sized and 6000px scans.
    heights = []
    for b in boxes:
        pts = np.array(b, dtype=np.float32)
        heights.append(float(pts[:, 1].max() - pts[:, 1].min()))
    med_h = float(np.median(heights)) if heights else 20.0
    row_thresh = max(med_h * 0.5, 10.0)

    order = sorted(range(len(centroids)), key=lambda i: centroids[i][1])

    rows: List[List[int]] = []
    current_row: List[int] = [order[0]]
    current_y = centroids[order[0]][1]

    for idx in order[1:]:
        cy = centroids[idx][1]
        if abs(cy - current_y) < row_thresh:
            current_row.append(idx)
        else:
            rows.append(current_row)
            current_row = [idx]
            current_y = cy
    rows.append(current_row)

    result = []
    for row in rows:
        row.sort(key=lambda i: centroids[i][0])
        result.extend(row)

    return result


def align_boxes_with_transcript(
    boxes: List[List[List[float]]],
    lines: List[str],
) -> Tuple[List[Tuple[List[List[float]], str]], int, int]:
    """Zip boxes (in reading order) with transcript lines.

    -> ([(box, text), ...], num_boxes, num_lines). Extra of either is dropped,
    and the counts let the caller warn about the mismatch.
    """
    sorted_indices = sort_boxes_reading_order(boxes)
    sorted_boxes = [boxes[i] for i in sorted_indices]

    num_pairs = min(len(sorted_boxes), len(lines))
    pairs = [(sorted_boxes[i], lines[i]) for i in range(num_pairs)]

    return pairs, len(sorted_boxes), len(lines)


# ── Cropping ────────────────────────────────────────────────────────

def crop_line_image(
    image: np.ndarray,
    box: List[List[float]],
    padding: int = 4,
) -> np.ndarray:
    """Crop one line out of the page, with a few pixels of breathing room."""
    pts = np.array(box, dtype=np.float32)
    x_min, y_min = int(pts[:, 0].min()), int(pts[:, 1].min())
    x_max, y_max = int(pts[:, 0].max()), int(pts[:, 1].max())

    h, w = image.shape[:2]
    x1 = max(0, x_min - padding)
    y1 = max(0, y_min - padding)
    x2 = min(w, x_max + padding)
    y2 = min(h, y_max + padding)

    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    return image[y1:y2, x1:x2].copy()


# ── Filename helper ─────────────────────────────────────────────────

def _safe_label(text: str, max_len: int = 30) -> str:
    """Turn transcript text into something safe to use as a filename."""
    safe = re.sub(r"[^\w\s-]", "", text)
    safe = re.sub(r"\s+", "_", safe.strip())
    return safe[:max_len] if safe else "line"


# ── Dataset Export ──────────────────────────────────────────────────

def build_dataset_zip(
    pages_data: List[Dict[str, Any]],
    book_name: str = "dataset",
) -> io.BytesIO:
    """ZIP of line crops plus a labels.csv.

    pages_data entries: {page_key, image_data (base64 data URL), boxes, lines}.
    """
    import base64

    buf = io.BytesIO()
    csv_rows: List[Tuple[str, str]] = []  # (image_path, text)

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for page in pages_data:
            page_key = page["page_key"]
            boxes = page["boxes"]
            lines = page["lines"]

            page_num_match = re.search(r"(\d+)", str(page_key))
            page_dir = f"page_{int(page_num_match.group(1))}" if page_num_match else f"page_{page_key}"

            img_b64 = page["image_data"]
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            pairs, _, _ = align_boxes_with_transcript(boxes, lines)
            # Two lines with the same text would collide on filename.
            seen_labels: Dict[str, int] = {}

            for idx, (box, text) in enumerate(pairs):
                crop = crop_line_image(img, box)
                label = _safe_label(text, max_len=80)
                count = seen_labels.get(label, 0) + 1
                seen_labels[label] = count
                label_with_suffix = label if count == 1 else f"{label}_{count}"

                img_filename = f"{book_name}/{page_dir}/{label_with_suffix}.png"

                _, png_data = cv2.imencode(".png", crop)
                zf.writestr(img_filename, png_data.tobytes())

                csv_rows.append((img_filename, text))

                del crop

            del img, nparr, img_bytes

        # BOM so Excel opens the accented text correctly.
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["image", "text"])
        for path, text in csv_rows:
            writer.writerow([path, text])
        csv_bytes = b'\xef\xbb\xbf' + csv_buf.getvalue().encode('utf-8')
        zf.writestr(f"{book_name}/labels.csv", csv_bytes)

    buf.seek(0)
    return buf


# ── Detection-only Dataset Export ──────────────────────────────────

def _polygon_to_xyxy(box: List[List[float]]) -> Tuple[int, int, int, int]:
    """Axis-aligned integer-pixel rect around a 4-point polygon."""
    pts = np.asarray(box, dtype=np.float32)
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )


def build_detection_dataset_zip(
    pages_data: List[Dict[str, Any]],
    book_name: str = "dataset",
    bbox_format: str = "txt",
) -> io.BytesIO:
    """ZIP of full page images plus their line boxes — no transcript needed.

    bbox_format: "txt" (x1 y1 x2 y2 per line), "json" (array of rects),
    "yolo" (normalised cx cy w h + classes.txt), or "coco" (one annotations.json).
    """
    import base64
    import json

    fmt = (bbox_format or "txt").lower()
    if fmt not in ("txt", "json", "yolo", "coco"):
        raise ValueError(f"Unsupported bbox_format: {bbox_format!r} (use 'txt', 'json', 'yolo', or 'coco')")

    buf = io.BytesIO()

    # COCO wants one file for the whole set, so accumulate across pages.
    coco_images: List[Dict[str, Any]] = []
    coco_annotations: List[Dict[str, Any]] = []
    coco_image_id = 0
    coco_ann_id = 0

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for page in pages_data:
            page_key = page["page_key"]
            boxes = page.get("boxes", [])

            img_b64 = page["image_data"]
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            h, w = img.shape[:2]
            stem = f"page_{page_key}"
            jpg_name = f"{stem}.jpg"
            _, jpg_data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            zf.writestr(f"{book_name}/images/{jpg_name}", jpg_data.tobytes())

            xyxy_boxes = [_polygon_to_xyxy(b) for b in boxes]

            if fmt == "txt":
                lines = "\n".join(
                    f"{x1} {y1} {x2} {y2}" for (x1, y1, x2, y2) in xyxy_boxes
                )
                payload = (lines + "\n").encode("utf-8") if lines else b""
                zf.writestr(f"{book_name}/bboxes/{stem}.txt", payload)
            elif fmt == "json":
                payload = json.dumps(
                    [[x1, y1, x2, y2] for (x1, y1, x2, y2) in xyxy_boxes],
                    indent=2,
                ).encode("utf-8")
                zf.writestr(f"{book_name}/bboxes/{stem}.json", payload)
            elif fmt == "yolo":
                yolo_lines = []
                for (x1, y1, x2, y2) in xyxy_boxes:
                    bw = max(0, x2 - x1)
                    bh = max(0, y2 - y1)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    yolo_lines.append(
                        f"0 {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                    )
                payload = ("\n".join(yolo_lines) + "\n").encode("utf-8") if yolo_lines else b""
                zf.writestr(f"{book_name}/labels/{stem}.txt", payload)
            elif fmt == "coco":
                coco_image_id += 1
                coco_images.append({
                    "id": coco_image_id,
                    "file_name": jpg_name,
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

            del img, nparr, img_bytes

        if fmt == "yolo":
            zf.writestr(f"{book_name}/classes.txt", b"text\n")
        elif fmt == "coco":
            coco = {
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": [{"id": 1, "name": "text", "supercategory": "text"}],
            }
            zf.writestr(
                f"{book_name}/annotations.json",
                json.dumps(coco, indent=2).encode("utf-8"),
            )

    buf.seek(0)
    return buf
