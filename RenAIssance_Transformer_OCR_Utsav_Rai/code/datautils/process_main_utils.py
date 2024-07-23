import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
import math
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from deskew import determine_skew
import os
import argparse
from typing import Tuple, Union

def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def deskew_image(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    if angle is not None:
        rotated = rotate(image, angle, (0, 0, 0))
        return rotated
    else:
        return image

def preprocess_image(image: np.ndarray) -> np.ndarray:
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Rescale the image if it is too small
    height, width = gray.shape
    if height < 1000:
        scale_factor = 1000 / height
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    # Apply Sauvola's thresholding
    window_size = 25
    sauvola_threshold = threshold_sauvola(gray, window_size=window_size)
    binary_sauvola = gray > sauvola_threshold
    binary_sauvola = (binary_sauvola * 255).astype(np.uint8)
    # Ensure text is black and background is white
    if np.mean(binary_sauvola) < 127:
        binary_sauvola = cv2.bitwise_not(binary_sauvola)
    
    # Apply morphological operations to remove small noise
    kernel = np.ones((2, 2), np.uint8)  # Use a smaller kernel size
    opened = cv2.morphologyEx(binary_sauvola, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # Apply Non-Local Means Denoising with adjusted parameters
    denoised = cv2.fastNlMeansDenoising(closed, None, h=15, templateWindowSize=7, searchWindowSize=21)
    
    # Add border to the image
    bordered = cv2.copyMakeBorder(denoised, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return bordered

def draw_hough_lines(img, lines, color=(0, 255, 255), thickness=2):
    img_copy = np.copy(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
    return img_copy

def remove_borders(image: np.ndarray) -> np.ndarray:
    # Invert the image (text and borders become white, background becomes black)
    img_inverted = cv2.bitwise_not(image)

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    detected_horizontal = cv2.morphologyEx(img_inverted, cv2.MORPH_OPEN, horizontal_kernel)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    detected_vertical = cv2.morphologyEx(img_inverted, cv2.MORPH_OPEN, vertical_kernel)

    # Combine the detected lines
    detected_lines = cv2.addWeighted(detected_horizontal, 1.0, detected_vertical, 1.0, 0.0)

    # Additional processing to improve line detection
    # Dilate the detected lines to make them thicker and more continuous
    dilated_lines = cv2.dilate(detected_lines, np.ones((1, 1), np.uint8), iterations=2)

    # Apply morphological closing to bridge any gaps in the lines
    closed_lines = cv2.morphologyEx(dilated_lines, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))

    # Ensure the final processed image is binary
    _, binary_lines = cv2.threshold(closed_lines, 127, 255, cv2.THRESH_BINARY)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(binary_lines, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Filtered contours using Hough lines
    mask = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(mask, (x1, y1), (x2, y2), 255, 10)

    # Dilate the mask to increase the line size
    dilated_mask = cv2.dilate(mask, np.ones((1, 1), np.uint8), iterations=2)

    # Replace the dilated lines with white pixels on the result image
    img_result = image.copy()
    img_result[dilated_mask == 255] = 255

    return img_result

def process_pdf(book_path, output_dir, dpi=300, remove_borders_flag=False):
    # Check if the output directory exists, and create it if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Open the PDF
    pdf_document = fitz.open(book_path)
    total_pages = pdf_document.page_count

    print(f"Total pages in PDF: {total_pages}")

    image_index = 0  # Start naming images from 0

    for page_number in range(total_pages):
        page = pdf_document.load_page(page_number)
        # Render the page to an image with specified DPI
        zoom = dpi / 72  # 72 is the default DPI for PDF rendering
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)  # Disable alpha to avoid transparency issues
        
        # Convert pixmap to PIL image
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2)  # Increase sharpness
        
        # Convert PIL image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Check aspect ratio to determine if it is a double-page scan
        width, height = image.shape[1], image.shape[0]
        aspect_ratio = width / height
        
        if aspect_ratio > 1:  # Threshold to determine if it's a double-page scan
            # Split the image in half
            left_half = image[:, :width // 2]
            right_half = image[:, width // 2:]

            # Deskew each half
            left_half_deskewed = deskew_image(left_half)
            right_half_deskewed = deskew_image(right_half)

            # Preprocess each half
            left_half_processed = preprocess_image(left_half_deskewed)
            right_half_processed = preprocess_image(right_half_deskewed)

            if remove_borders_flag:
                left_half_processed = remove_borders(left_half_processed)
                right_half_processed = remove_borders(right_half_processed)
            
            # Save the images in TIFF format to preserve quality
            left_half_path = os.path.join(output_dir, f"{image_index}.jpg")
            cv2.imwrite(left_half_path, left_half_processed)
            image_index += 1
            right_half_path = os.path.join(output_dir, f"{image_index}.jpg")
            cv2.imwrite(right_half_path, right_half_processed)
            image_index += 1
        else:
            # If it's a single page, process and save the entire image
            image_deskewed = deskew_image(image)
            image_processed = preprocess_image(image_deskewed)
            
            if remove_borders_flag:
                image_processed = remove_borders(image_processed)

            image_path = os.path.join(output_dir, f"{image_index}.jpg")
            cv2.imwrite(image_path, image_processed)
            image_index += 1

        print(f"Processed page {page_number + 1}/{total_pages}")

    print("Processing completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a PDF book by splitting, deskewing, and preprocessing each page.")
    parser.add_argument("book_path", type=str, help="Path to the PDF book")
    parser.add_argument("output_dir", type=str, help="Directory to save the processed images")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for rendering PDF pages (default: 300)")
    parser.add_argument("--remove_borders", action="store_true", help="Remove borders from the processed images")

    args = parser.parse_args()
    process_pdf(args.book_path, args.output_dir, args.dpi, args.remove_borders)
