"""OpenCV preprocessing ops for OCR.

Every op is (image, params, progress?) -> image and copes with both grayscale
and colour input. OP_REGISTRY at the bottom is what the pipeline looks up.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Callable


ProgressCallbackType = Optional[Callable[[float, str], None]]


# ── Basic ─────────────────────────────────────────────────────────────

def normalize_image(
    img: np.ndarray,
    params: Dict[str, Any],
    progress: ProgressCallbackType = None
) -> np.ndarray:
    """Histogram-stretch brightness/contrast. params: strength 0-100 (50)."""
    if progress:
        progress(0.1, "Analyzing histogram")

    strength = params.get("strength", 50) / 100.0

    if len(img.shape) == 2:
        normalized = _normalize_channel(img, strength)
    else:
        channels = cv2.split(img)
        normalized_channels = []
        for i, ch in enumerate(channels):
            if progress:
                progress(0.2 + (i * 0.25), f"Normalizing channel {i+1}")
            normalized_channels.append(_normalize_channel(ch, strength))
        normalized = cv2.merge(normalized_channels)

    if progress:
        progress(1.0, "Normalize complete")

    return normalized


def _normalize_channel(channel: np.ndarray, strength: float) -> np.ndarray:
    """Blend the channel toward a full 0-255 stretch by `strength`."""
    min_val = np.min(channel)
    max_val = np.max(channel)

    if max_val == min_val:
        return channel

    full_normalized = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)

    if strength >= 1.0:
        return full_normalized

    return cv2.addWeighted(
        channel.astype(np.float32), 1 - strength,
        full_normalized.astype(np.float32), strength,
        0
    ).astype(np.uint8)


def to_grayscale(
    img: np.ndarray,
    params: Dict[str, Any],
    progress: ProgressCallbackType = None
) -> np.ndarray:
    """Convert to grayscale. No params; already-gray input passes through."""
    if progress:
        progress(0.2, "Converting to grayscale")

    if len(img.shape) == 2:
        result = img
    elif len(img.shape) == 3 and img.shape[2] == 1:
        result = img.squeeze()  # single channel, but still 3D
    else:
        result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if progress:
        progress(1.0, "Grayscale complete")

    return result


# Below this spread (degrees) between the most- and least-skewed band, the page
# is skewed uniformly and a single rotation is cleaner than a piecewise warp.
_PIECEWISE_MIN_SPREAD = 1.0


def deskew_image(
    img: np.ndarray,
    params: Dict[str, Any],
    progress: ProgressCallbackType = None
) -> np.ndarray:
    """Straighten a rotated scan, handling skew that drifts down the page.

    params: maxAngle in degrees (15); mode 'auto'|'global'|'piecewise' ('auto');
    bands 2-12 (4) for the piecewise path.

    A single rotation cannot fix a page whose top is straight but whose lower
    half tilts (common near a book's spine, or when the sheet lifts off the
    platen). 'auto' measures the skew separately in horizontal bands: if it is
    roughly constant it rotates once, otherwise it corrects each band by its own
    angle and blends the bands so there is no visible seam.
    """
    if progress:
        progress(0.1, "Preparing deskew analysis")

    max_angle = params.get("maxAngle", 15)
    mode = str(params.get("mode", "auto")).lower()
    num_bands = max(2, min(12, int(params.get("bands", 4))))

    is_color = len(img.shape) == 3
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if is_color else img.copy()

    if progress:
        progress(0.25, "Detecting skew angle")

    # Whole-page angle: the fallback for uniform skew and for bands with no ink.
    global_angle = _detect_skew_contour(gray)
    if global_angle is None or abs(global_angle) > max_angle:
        global_angle = _detect_skew_hough(gray)
    if global_angle is None:
        global_angle = 0.0
    global_angle = max(-max_angle, min(max_angle, global_angle))

    if mode == "global":
        return _apply_global_rotation(img, global_angle, progress)

    if progress:
        progress(0.45, "Measuring skew per band")

    band_angles = _band_skew_angles(gray, num_bands, max_angle, global_angle)
    spread = max(band_angles) - min(band_angles)

    # Uniform skew (or an explicit request for one rotation): rotate once.
    if mode == "auto" and spread < _PIECEWISE_MIN_SPREAD:
        return _apply_global_rotation(img, float(np.median(band_angles)), progress)

    if progress:
        progress(0.6, "Applying piecewise deskew")

    result = _piecewise_deskew(img, band_angles)

    if progress:
        progress(1.0, "Deskew complete")

    return result


def _apply_global_rotation(
    img: np.ndarray, angle: float, progress: ProgressCallbackType = None
) -> np.ndarray:
    """Rotate the whole image about its centre by `angle` (already clamped)."""
    if abs(angle) < 0.1:
        if progress:
            progress(1.0, "No significant skew detected")
        return img

    if progress:
        progress(0.6, f"Rotating by {angle:.2f}°")

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Replicate the border instead of filling black — cleaner page edges.
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    if progress:
        progress(1.0, "Deskew complete")

    return rotated


def _detect_skew_contour(gray: np.ndarray) -> Optional[float]:
    """Skew angle from the min-area rect around the largest contour."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[-1]

    # minAreaRect reports in (-90, 0]; fold it into (-45, 45].
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    return angle


def _detect_skew_hough(gray: np.ndarray) -> Optional[float]:
    """Skew angle from the median slope of near-horizontal Hough lines."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 4,
        maxLineGap=10
    )

    if lines is None or len(lines) == 0:
        return None

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:  # text lines, not vertical rules
                angles.append(angle)

    if not angles:
        return None

    return np.median(angles)


def _band_skew_angles(
    gray: np.ndarray, num_bands: int, max_angle: float, fallback_angle: float
) -> list[float]:
    """Estimate the correction angle for each horizontal band, top to bottom.

    Runs on a downscaled copy (angles are scale-invariant) so estimating many
    bands stays cheap. Bands with too little ink to trust inherit the median of
    the confident bands (or the whole-page angle), then the series is lightly
    smoothed so one noisy band cannot kink the correction.
    """
    scale = min(1.0, 1000.0 / max(gray.shape[:2]))
    small = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0 else gray
    )
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h = binary.shape[0]
    raw: list[Optional[float]] = []
    for i in range(num_bands):
        y0 = i * h // num_bands
        y1 = (i + 1) * h // num_bands
        raw.append(_projection_profile_angle(binary[y0:y1], max_angle))

    return _fill_band_angles(raw, fallback_angle)


def _projection_profile_angle(
    binary_band: np.ndarray, max_angle: float
) -> Optional[float]:
    """Angle (in [-max_angle, max_angle]) that makes this band's lines horizontal.

    Classic projection-profile deskew: at the correct angle the rows of text
    line up, so the horizontal ink projection has the sharpest peaks and valleys.
    Score each candidate by the summed squared row-to-row change in that
    projection and keep the best, coarse (1°) then refined (0.25°). Returns None
    when the band holds too little ink to give a trustworthy answer.
    """
    if binary_band.size == 0 or int(binary_band.sum()) < 255 * 50:
        return None

    h, w = binary_band.shape

    def score(angle: float) -> float:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rot = cv2.warpAffine(
            binary_band, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        proj = rot.sum(axis=1, dtype=np.float64)
        d = np.diff(proj)
        return float(np.dot(d, d))

    coarse = np.arange(-max_angle, max_angle + 0.5, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 1.0, best + 1.0 + 1e-6, 0.25)
    best = max(fine, key=score)
    return float(max(-max_angle, min(max_angle, best)))


def _fill_band_angles(
    raw: list[Optional[float]], fallback_angle: float
) -> list[float]:
    """Replace unconfident (None) bands with a sensible default, then smooth."""
    confident = [a for a in raw if a is not None]
    default = float(np.median(confident)) if confident else float(fallback_angle)
    filled = [a if a is not None else default for a in raw]

    # 3-tap moving average: a single outlier band bends the correction less.
    smoothed = []
    for i in range(len(filled)):
        window = filled[max(0, i - 1):i + 2]
        smoothed.append(float(np.mean(window)))
    return smoothed


def _piecewise_deskew(img: np.ndarray, band_angles: list[float]) -> np.ndarray:
    """Rotate each horizontal band by its own angle and blend the bands.

    Each band is corrected by rotating the whole image about that band's centre,
    then every output row is a linear blend of the two nearest band rotations.
    The blend weights form a partition of unity (they sum to 1 per row), so the
    transition between bands is gradual and leaves no seam.
    """
    h, w = img.shape[:2]
    n = len(band_angles)
    cx = w / 2.0

    # Band-index coordinate for each output row: band i's centre sits at p = i.
    p = np.clip(np.arange(h) / h * n - 0.5, 0.0, n - 1)

    channel_shape = () if img.ndim == 2 else (img.shape[2],)
    acc = np.zeros((h, w) + channel_shape, dtype=np.float32)

    for i, angle in enumerate(band_angles):
        weight = np.clip(1.0 - np.abs(p - i), 0.0, 1.0)  # triangular
        if not weight.any():
            continue
        cy = (i + 0.5) * h / n
        M = cv2.getRotationMatrix2D((cx, cy), float(angle), 1.0)
        rot = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(np.float32)
        acc += rot * weight.reshape(h, *([1] * len(channel_shape)))

    return np.clip(acc, 0, 255).astype(np.uint8)


# ── Enhancement ───────────────────────────────────────────────────────

def denoise_image(
    img: np.ndarray,
    params: Dict[str, Any],
    progress: ProgressCallbackType = None
) -> np.ndarray:
    """Denoise without smearing text edges.

    params: method 'nlm' (best, slow) | 'bilateral' | 'gaussian' (fastest),
    strength 1-20 (10).
    """
    if progress:
        progress(0.1, "Preparing denoising")

    method = params.get("method", "nlm")
    strength = params.get("strength", 10)
    img = img.astype(np.uint8)

    if progress:
        progress(0.2, f"Applying {method} denoising")

    if method == "bilateral":
        d = max(5, min(15, int(strength)))
        sigma_color = strength * 7.5
        sigma_space = strength * 7.5
        result = cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    elif method == "gaussian":
        ksize = max(3, int(strength) | 1)  # kernel must be odd
        result = cv2.GaussianBlur(img, (ksize, ksize), 0)

    else:  # nlm
        h = max(3, min(30, strength))
        template_window = 7
        search_window = 21

        if len(img.shape) == 3:
            result = cv2.fastNlMeansDenoisingColored(
                img, None, h, h, template_window, search_window
            )
        else:
            result = cv2.fastNlMeansDenoising(
                img, None, h, template_window, search_window
            )

    if progress:
        progress(1.0, "Denoise complete")

    return result


def clahe_contrast(
    img: np.ndarray,
    params: Dict[str, Any],
    progress: ProgressCallbackType = None
) -> np.ndarray:
    """CLAHE — local contrast without blowing up noise.

    params: clipLimit (2.0), tileSize (8).
    """
    if progress:
        progress(0.1, "Preparing CLAHE")

    clip_limit = params.get("clipLimit", 2.0)
    tile_size = params.get("tileSize", 8)
    tile_size = max(2, min(16, int(tile_size)))

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    if progress:
        progress(0.3, "Applying CLAHE")

    if len(img.shape) == 2:
        result = clahe.apply(img)
    else:
        # Colour: only touch luminance, so hues stay put.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        if progress:
            progress(0.5, "Enhancing luminance")

        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if progress:
        progress(1.0, "Contrast enhancement complete")

    return result


def sharpen_image(
    img: np.ndarray,
    params: Dict[str, Any],
    progress: ProgressCallbackType = None
) -> np.ndarray:
    """Unsharp-mask the text edges.

    params: amount 0-100 (50), radius in px 0.5-3 (1).
    """
    if progress:
        progress(0.1, "Preparing sharpening")

    amount = params.get("amount", 50) / 100.0
    radius = params.get("radius", 1.0)

    if amount <= 0:
        if progress:
            progress(1.0, "No sharpening applied")
        return img

    ksize = max(3, int(radius * 2) | 1)  # kernel must be odd

    if progress:
        progress(0.3, "Creating blur mask")

    if len(img.shape) == 3:
        blurred = cv2.GaussianBlur(img, (ksize, ksize), radius)
    else:
        blurred = cv2.GaussianBlur(img, (ksize, ksize), radius)

    if progress:
        progress(0.6, "Applying unsharp mask")

    # original + amount * (original - blurred) — adds the high frequencies back.
    sharpened = cv2.addWeighted(
        img.astype(np.float32), 1.0 + amount,
        blurred.astype(np.float32), -amount,
        0
    )

    result = np.clip(sharpened, 0, 255).astype(np.uint8)

    if progress:
        progress(1.0, "Sharpen complete")

    return result


# ── Binarization ──────────────────────────────────────────────────────

def threshold_image(
    img: np.ndarray,
    params: Dict[str, Any],
    progress: ProgressCallbackType = None
) -> np.ndarray:
    """Binarize to pure black/white.

    params: method 'otsu' | 'adaptive' | 'sauvola' (best on degraded scans),
    blockSize (15) and k (0.5) for the local methods.
    """
    if progress:
        progress(0.1, "Preparing binarization")

    method = params.get("method", "otsu")
    block_size = params.get("blockSize", 15)
    k = params.get("k", 0.5)
    block_size = max(3, int(block_size) | 1)  # must be odd, >= 3

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if progress:
        progress(0.3, f"Applying {method} thresholding")

    if method == "adaptive":
        result = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, 8
        )

    elif method == "sauvola":
        result = _sauvola_threshold(gray, block_size, k, progress)

    else:  # otsu
        _, result = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    if progress:
        progress(1.0, "Binarization complete")

    return result


def _sauvola_threshold(
    gray: np.ndarray,
    window_size: int,
    k: float,
    progress: ProgressCallbackType = None
) -> np.ndarray:
    """Sauvola: T(x,y) = mean * (1 + k * (std / R - 1)), R = 128 for 8-bit.

    Means and variances come from box filters, so it stays O(n) per pixel.
    """
    if progress:
        progress(0.4, "Computing local statistics")

    mean = cv2.blur(gray.astype(np.float64), (window_size, window_size))
    sq_mean = cv2.blur(gray.astype(np.float64) ** 2, (window_size, window_size))

    variance = sq_mean - mean ** 2
    variance = np.maximum(variance, 0)  # float error can push this negative
    std = np.sqrt(variance)

    if progress:
        progress(0.7, "Computing threshold map")

    R = 128.0
    threshold = mean * (1.0 + k * (std / R - 1.0))

    if progress:
        progress(0.9, "Applying threshold")

    result = np.zeros_like(gray)
    result[gray > threshold] = 255

    return result.astype(np.uint8)


# ── Morphology ────────────────────────────────────────────────────────

def morph_operations(
    img: np.ndarray,
    params: Dict[str, Any],
    progress: ProgressCallbackType = None
) -> np.ndarray:
    """Morphological cleanup of text and artifacts.

    params: operation open|close|dilate|erode|gradient|tophat|blackhat ('open'),
    kernelSize 1-9 (2), kernelShape ellipse|rect|cross, iterations 1-10 (1).
    """
    if progress:
        progress(0.1, "Preparing morphological operation")

    operation = params.get("operation", "open")
    k = max(1, min(9, int(params.get("kernelSize", 2))))
    shape_name = params.get("kernelShape", "ellipse")
    iterations = max(1, min(10, int(params.get("iterations", 1))))

    shape_map = {
        "ellipse": cv2.MORPH_ELLIPSE,
        "rect": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
    }
    shape = shape_map.get(shape_name, cv2.MORPH_ELLIPSE)
    kernel = cv2.getStructuringElement(shape, (k, k))

    if progress:
        progress(0.3, f"Applying {operation} (k={k}, iter={iterations})")

    # dilate/erode aren't morphologyEx ops, so they're handled separately below.
    morph_ops = {
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "gradient": cv2.MORPH_GRADIENT,
        "tophat": cv2.MORPH_TOPHAT,
        "blackhat": cv2.MORPH_BLACKHAT,
    }

    if operation in morph_ops:
        result = cv2.morphologyEx(
            img, morph_ops[operation], kernel, iterations=iterations
        )
    elif operation == "dilate":
        result = cv2.dilate(img, kernel, iterations=iterations)
    elif operation == "erode":
        result = cv2.erode(img, kernel, iterations=iterations)
    else:
        result = cv2.morphologyEx(
            img, cv2.MORPH_OPEN, kernel, iterations=iterations
        )

    if progress:
        progress(1.0, "Morphological operation complete")

    return result


# ── Blob & noise removal ──────────────────────────────────────────────

def remove_large_blobs(
    img: np.ndarray,
    params: Dict[str, Any],
    progress: ProgressCallbackType = None
) -> np.ndarray:
    """Punch out big ink blobs, keeping the letters they touch.

    Only the eroded inner core gets painted white — erasing the whole connected
    component would take neighbouring characters with it.

    params: minArea (3000), minSolidity (0.55), maxAspectRatio (4.0),
    erosionRatio (0.35 — higher removes less, safer).
    """
    if progress:
        progress(0.1, "Preparing blob detection")

    min_area = int(params.get("minArea", 3000))
    min_solidity = float(params.get("minSolidity", 0.55))
    max_aspect_ratio = float(params.get("maxAspectRatio", 4.0))
    erosion_ratio = float(params.get("erosionRatio", 0.35))

    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Invert first: connectedComponents wants ink as white foreground.
    inverted = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )

    if progress:
        progress(0.3, f"Analyzing {num_labels - 1} components")

    result = binary.copy()

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]

        if area <= min_area:  # small enough to just be a character
            continue

        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        aspect = max(w, h) / max(min(w, h), 1)

        if aspect > max_aspect_ratio:  # long and thin — a stroke or page border
            continue

        # Ragged outline means text; a real blob is close to its convex hull.
        component_mask = np.uint8(labels == lbl) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
        solidity = float(area) / hull_area if hull_area > 0 else 0.0

        if solidity < min_solidity:
            continue

        k_radius = max(3, int(erosion_ratio * (area ** 0.5)))
        k_size = 2 * k_radius + 1
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_size, k_size)
        )
        core = cv2.erode(component_mask, kern, iterations=1)
        result[core == 255] = 255  # white == background

        if progress:
            pct = 0.3 + 0.6 * (lbl / max(num_labels - 1, 1))
            progress(min(pct, 0.9), f"Processing blob {lbl}")

    if progress:
        progress(1.0, "Blob removal complete")

    return result


def remove_small_noise(
    img: np.ndarray,
    params: Dict[str, Any],
    progress: ProgressCallbackType = None
) -> np.ndarray:
    """Drop speckles and scanner dust. params: maxArea (20)."""
    if progress:
        progress(0.1, "Preparing noise detection")

    max_area = max(1, int(params.get("maxArea", 20)))

    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    inverted = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )

    if progress:
        progress(0.4, f"Filtering {num_labels - 1} components (threshold={max_area})")

    keep_mask = np.zeros_like(inverted)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= max_area:
            keep_mask[labels == lbl] = 255

    result = cv2.bitwise_not(keep_mask)

    if progress:
        progress(1.0, "Small noise removal complete")

    return result


# ── Registry ──────────────────────────────────────────────────────────

OP_REGISTRY = {
    "normalize": normalize_image,
    "grayscale": to_grayscale,
    "deskew": deskew_image,
    "denoise": denoise_image,
    "contrast": clahe_contrast,
    "sharpen": sharpen_image,
    "threshold": threshold_image,
    "morph": morph_operations,
    "remove_blobs": remove_large_blobs,
    "remove_noise": remove_small_noise,
}


def get_operation(name: str):
    return OP_REGISTRY.get(name)


def list_operations() -> list[str]:
    return list(OP_REGISTRY.keys())
