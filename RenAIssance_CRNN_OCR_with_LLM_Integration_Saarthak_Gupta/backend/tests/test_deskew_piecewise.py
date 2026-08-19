"""Piecewise deskew: the estimator finds a tilted band's angle, and a page
whose skew grows toward the bottom is detected as variable (not uniform)."""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.operations import (  # noqa: E402
    _band_skew_angles,
    _projection_profile_angle,
    deskew_image,
)


def _ruled_page(h=480, w=640, line_gap=24):
    """White page with evenly spaced horizontal black 'text' lines."""
    img = np.full((h, w), 255, np.uint8)
    for y in range(line_gap, h - line_gap, line_gap):
        cv2.line(img, (40, y), (w - 40, y), 0, 3)
    return img


def _rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=255)


def test_projection_profile_recovers_tilt():
    # Lines tilted by +6 deg need a -6 deg rotation to flatten, so that is what
    # the estimator should report.
    tilted = _rotate(_ruled_page(), 6.0)
    _, binary = cv2.threshold(tilted, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    angle = _projection_profile_angle(binary, max_angle=15)
    assert angle is not None
    assert abs(angle - (-6.0)) <= 1.0, f"expected ~-6, got {angle}"


def test_variable_skew_detected_as_nonuniform():
    # Straight top, 8-degree-tilted bottom, stitched into one page.
    page = _ruled_page()
    h = page.shape[0]
    page[h // 2:, :] = _rotate(page, 8.0)[h // 2:, :]

    angles = _band_skew_angles(page, num_bands=4, max_angle=15, fallback_angle=0.0)
    spread = max(angles) - min(angles)
    assert spread >= 1.0, f"variable skew should register a spread, got {angles}"
    # Bottom bands must be corrected harder than the (straight) top band.
    assert abs(angles[-1]) > abs(angles[0]) + 1.0, angles


def test_deskew_runs_and_preserves_shape():
    page = cv2.cvtColor(_ruled_page(), cv2.COLOR_GRAY2BGR)
    out = deskew_image(page, {"mode": "auto", "maxAngle": 15, "bands": 4})
    assert out.shape == page.shape
    assert out.dtype == np.uint8


if __name__ == "__main__":
    test_projection_profile_recovers_tilt()
    test_variable_skew_detected_as_nonuniform()
    test_deskew_runs_and_preserves_shape()
    print("ok")
