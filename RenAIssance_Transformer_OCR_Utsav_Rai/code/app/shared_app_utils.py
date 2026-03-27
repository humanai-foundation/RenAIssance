"""Shared OCR and preprocessing helpers used by the app entrypoints."""

import math

import cv2
import numpy as np
from deskew import determine_skew


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0

    new_state_dict = {}
    for key, value in state_dict.items():
        name = ".".join(key.split(".")[start_idx:])
        new_state_dict[name] = value
    return new_state_dict


def rotate(image: np.ndarray, angle: float, background: tuple) -> np.ndarray:
    old_height, old_width = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2

    return cv2.warpAffine(
        image,
        rot_mat,
        (int(round(width)), int(round(height))),
        borderValue=background,
    )


def deskew_image(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    if angle is not None:
        return rotate(image, angle, (0, 0, 0))
    return image


def preprocess_image(image: np.ndarray, noise_removal_area_threshold: int, intensity_threshold: int) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 25)

    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    sizes = stats[1:, -1]
    new_image = np.zeros(labels.shape, np.uint8)

    for i in range(1, num_labels):
        component_mask = labels == i
        component_intensity = np.mean(gray[component_mask])
        if sizes[i - 1] >= noise_removal_area_threshold and component_intensity <= intensity_threshold:
            new_image[component_mask] = 255

    inverted_image = cv2.bitwise_not(new_image)
    return cv2.copyMakeBorder(inverted_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def remove_borders(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    img_inverted = cv2.bitwise_not(gray)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    detected_horizontal = cv2.morphologyEx(img_inverted, cv2.MORPH_OPEN, horizontal_kernel)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    detected_vertical = cv2.morphologyEx(img_inverted, cv2.MORPH_OPEN, vertical_kernel)

    detected_lines = cv2.addWeighted(detected_horizontal, 1.0, detected_vertical, 1.0, 0.0)
    dilated_lines = cv2.dilate(detected_lines, np.ones((1, 1), np.uint8), iterations=2)
    closed_lines = cv2.morphologyEx(dilated_lines, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))

    _, binary_lines = cv2.threshold(closed_lines, 127, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(binary_lines, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    mask = np.zeros_like(gray)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(mask, (x1, y1), (x2, y2), 255, 10)

    dilated_mask = cv2.dilate(mask, np.ones((1, 1), np.uint8), iterations=2)
    img_result = gray.copy()
    img_result[dilated_mask == 255] = 255
    return img_result


def read_contour_points(file_path):
    contour_points = []
    with open(file_path, "r") as file:
        for line in file:
            points = list(map(int, line.strip().split(",")))
            contour_points.append(points)
    return contour_points


def get_bounding_boxes(contours, img_width, img_height, padding=10, min_width=20, margin=0.1):
    bounding_boxes = []
    top_margin = img_height * margin
    bottom_margin = img_height * (1 - margin)

    for contour in contours:
        points = np.array(contour).reshape((-1, 2))
        x, y, w, h = cv2.boundingRect(points)
        if w > min_width and (y > top_margin and y + h < bottom_margin):
            x = max(x - padding, 0)
            w = min(w + 2 * padding, img_width - x)
            bounding_boxes.append((x, y, x + w, y + h))

    centers = [(x1 + (x2 - x1) // 2) for (x1, y1, x2, y2) in bounding_boxes]
    median_center = np.median(centers)

    filtered_boxes = []
    for (x1, y1, x2, y2) in bounding_boxes:
        center = x1 + (x2 - x1) // 2
        if abs(center - median_center) < 800:
            filtered_boxes.append((x1, y1, x2, y2))
    return filtered_boxes


def draw_bounding_boxes(image, bounding_boxes, color=(0, 255, 0), thickness=6):
    for (x1, y1, x2, y2) in bounding_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def split_bounding_boxes(image, bounding_boxes, threshold=0.8):
    heights = [y2 - y1 for (x1, y1, x2, y2) in bounding_boxes]
    median_height = np.median(heights)
    new_bounding_boxes = []
    split_bounding_boxes_list = []

    for (x1, y1, x2, y2) in bounding_boxes:
        height = y2 - y1
        ratio = height / median_height
        if ratio > 1 + threshold:
            split_number = round(ratio)
            split_height = height // split_number
            for i in range(split_number):
                new_y1 = y1 + i * split_height
                new_y2 = new_y1 + split_height if i < split_number - 1 else y2
                split_bounding_boxes_list.append((x1, new_y1, x2, new_y2))
        else:
            new_bounding_boxes.append((x1, y1, x2, y2))

    draw_bounding_boxes(image, split_bounding_boxes_list, color=(255, 0, 0))
    return new_bounding_boxes + split_bounding_boxes_list


def filter_and_adjust_bounding_boxes(bounding_boxes):
    if not bounding_boxes:
        return []

    x1s = [x1 for (x1, y1, x2, y2) in bounding_boxes]
    x2s = [x2 for (x1, y1, x2, y2) in bounding_boxes]
    if not x1s or not x2s:
        return []

    median_x1 = int(np.median(x1s)) - 30
    median_x2 = int(np.median(x2s)) + 20

    adjusted_boxes = []
    for (x1, y1, x2, y2) in bounding_boxes:
        adjusted_boxes.append((median_x1, y1, median_x2, y2))

    non_overlapping_boxes = []
    for box in adjusted_boxes:
        overlap = False
        for other_box in non_overlapping_boxes:
            x1, y1, x2, y2 = box
            ox1, oy1, ox2, oy2 = other_box

            inter_x1 = max(x1, ox1)
            inter_y1 = max(y1, oy1)
            inter_x2 = min(x2, ox2)
            inter_y2 = min(y2, oy2)

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            box_area = (x2 - x1) * (y2 - y1)
            other_box_area = (ox2 - ox1) * (oy2 - oy1)

            if inter_area > 0.9 * min(box_area, other_box_area):
                overlap = True
                if box_area > other_box_area:
                    non_overlapping_boxes.remove(other_box)
                    non_overlapping_boxes.append(box)
                break
        if not overlap:
            non_overlapping_boxes.append(box)

    return non_overlapping_boxes
