import sys
import os
# Add CRAFT directory to sys.path for craft imports
CRAFT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CRAFT'))
if CRAFT_DIR not in sys.path:
    sys.path.insert(0, CRAFT_DIR)
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import tempfile
# Import CRAFT model and utilities
from craft import CRAFT
import craft_utils
import imgproc

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import os
import math
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import streamlit as st
from deskew import determine_skew

st.set_page_config(layout="wide")

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

@st.cache_resource
def load_craft_model():
    # Define the path to the pre-trained CRAFT model weights
    trained_model_path = '../../weights/craft_mlt_25k.pth'
    
    # Initialize the CRAFT model
    net = CRAFT()     # initialize

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the pre-trained weights
    net.load_state_dict(copyStateDict(torch.load(trained_model_path, map_location=device)))
    
    net.to(device)
    net.eval()
    
    # Load refiner model if needed
    refine_net = None
    refine = True  # Set to True if using refine_net
    if refine:
        from refinenet import RefineNet
        refiner_model_path = '../../weights/craft_refiner_CTW1500.pth'  # Update the path
        refine_net = RefineNet()
        refine_net.load_state_dict(copyStateDict(torch.load(refiner_model_path, map_location=device)))
        refine_net.to(device)
        refine_net.eval()
    return net, device, refine_net

def test_net(net, image, text_threshold, link_threshold, low_text, *, cuda, poly, device, refine_net=None):
    # Preprocess the image
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, square_size=1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
    )
    ratio_h = ratio_w = 1 / target_ratio

    # Normalize and prepare input tensor
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # Convert to [c, h, w]
    x = x.unsqueeze(0)  # Add batch dimension: [b, c, h, w]
    x = x.to(device)

    # Forward pass
    with torch.no_grad():
        y, feature = net(x)

    # Make score and link maps
    score_text = y[0, :, :, 0].cpu().numpy()
    score_link = y[0, :, :, 1].cpu().numpy()

    # Refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # Coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    return boxes, polys

# Set up the OCR model
@st.cache_resource
def load_ocr_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Update path to point to the correct location of the OCR weights
    model_path = "../../models"
    processor_path = "../../models"
    processor = TrOCRProcessor.from_pretrained(processor_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    return processor, model, device

processor, model, device = load_ocr_model()

# Functions

def rotate(image: np.ndarray, angle: float, background: tuple) -> np.ndarray:
    old_height, old_width = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (height - old_height) / 2
    rot_mat[0, 2] += (width - old_width) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(width)), int(round(height))), borderValue=background)

def deskew_image(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    if angle is not None:
        rotated = rotate(image, angle, (0, 0, 0))
        return rotated
    else:
        return image

def preprocess_image(image: np.ndarray, noise_removal_area_threshold: int, intensity_threshold: int) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # If already grayscale

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 25)

    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    sizes = stats[1:, -1]
    new_image = np.zeros((labels.shape), np.uint8)

    for i in range(1, num_labels):
        component_mask = (labels == i)
        component_intensity = np.mean(gray[component_mask])
        if sizes[i - 1] >= noise_removal_area_threshold and component_intensity <= intensity_threshold:
            new_image[component_mask] = 255

    inverted_image = cv2.bitwise_not(new_image)
    bordered = cv2.copyMakeBorder(inverted_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return bordered

def remove_borders(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # If already grayscale

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

# Function to read contour points from a file
def read_contour_points(file_path):
    contour_points = []
    with open(file_path, 'r') as file:
        for line in file:
            points = list(map(int, line.strip().split(',')))
            contour_points.append(points)
    return contour_points

# Function to generate bounding boxes around contours
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

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, bounding_boxes, color=(0, 255, 0), thickness=6):
    for (x1, y1, x2, y2) in bounding_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

# Function to split tall bounding boxes into smaller ones based on a threshold
def split_bounding_boxes(image, bounding_boxes, threshold=0.8):
    heights = [y2 - y1 for (x1, y1, x2, y2) in bounding_boxes]
    median_height = np.median(heights)
    new_bounding_boxes = []
    split_bounding_boxes_list = []

    for (x1, y1, x2, y2) in bounding_boxes:
        height = y2 - y1
        ratio = height / median_height
        if ratio > 1 + threshold:  # Identify tall bounding boxes
            split_number = round(ratio)
            split_height = height // split_number
            for i in range(split_number):
                new_y1 = y1 + i * split_height
                new_y2 = new_y1 + split_height if i < split_number - 1 else y2
                split_bounding_boxes_list.append((x1, new_y1, x2, new_y2))
        else:
            new_bounding_boxes.append((x1, y1, x2, y2))
    
    # Visualize the split bounding boxes in red
    draw_bounding_boxes(image, split_bounding_boxes_list, color=(255, 0, 0))
    
    return new_bounding_boxes + split_bounding_boxes_list

# Function to filter and adjust bounding boxes
def filter_and_adjust_bounding_boxes(bounding_boxes):
    # Return early if bounding_boxes is empty
    if not bounding_boxes:
        return []
    
    x1s = [x1 for (x1, y1, x2, y2) in bounding_boxes]
    x2s = [x2 for (x1, y1, x2, y2) in bounding_boxes]
    
    # Check if lists are empty before calculating median
    if not x1s or not x2s:
        return []
    
    median_x1 = int(np.median(x1s)) - 30  # Calculate median x1
    median_x2 = int(np.median(x2s)) + 20  # Calculate median x2 with a small adjustment

    adjusted_boxes = []
    for (x1, y1, x2, y2) in bounding_boxes:
        adjusted_boxes.append((median_x1, y1, median_x2, y2))  # Adjust bounding boxes to median x1 and x2

    # Remove overlapping bounding boxes, keeping only the one with the greater width
    non_overlapping_boxes = []
    for box in adjusted_boxes:
        overlap = False
        for other_box in non_overlapping_boxes:
            x1, y1, x2, y2 = box
            ox1, oy1, ox2, oy2 = other_box
            
            # Calculate intersection area
            inter_x1 = max(x1, ox1)
            inter_y1 = max(y1, oy1)
            inter_x2 = min(x2, ox2)
            inter_y2 = min(y2, oy2)
            
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            box_area = (x2 - x1) * (y2 - y1)
            other_box_area = (ox2 - ox1) * (oy2 - oy1)
            
            if inter_area > 0.9 * min(box_area, other_box_area):  # Check for 90% overlap
                overlap = True
                if box_area > other_box_area:
                    non_overlapping_boxes.remove(other_box)
                    non_overlapping_boxes.append(box)
                break
        if not overlap:
            non_overlapping_boxes.append(box)
    
    return non_overlapping_boxes

# Modified function to prevent caching of the unhashable `pdf_document` object
@st.cache_data
def process_page(_pdf_document, page_number, dpi, noise_threshold, intensity_threshold,
                 remove_borders_option, deskew_option, show_raw,
                 padding, min_width, margin, threshold, line_segmentation):
    processed_pages = []
    left_page_image = None
    right_page_image = None
    raw_left_half = None
    raw_right_half = None

    try:
        page = _pdf_document.load_page(page_number)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image = ImageEnhance.Sharpness(image).enhance(2)  # Enhance sharpness
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Split if the page appears to be a double-page spread
        height, width = image.shape[:2]
        if width > height:
            left_half = image[:, :width // 2]
            right_half = image[:, width // 2:]

            raw_left_half = left_half.copy()
            raw_right_half = right_half.copy()

            if not show_raw:
                if deskew_option:
                    left_half = deskew_image(left_half)
                    right_half = deskew_image(right_half)

                if remove_borders_option:
                    left_half = remove_borders(left_half)
                    right_half = remove_borders(right_half)

                # Apply noise threshold and intensity (See Through) adjustments
                left_half = preprocess_image(left_half, noise_threshold, intensity_threshold)
                right_half = preprocess_image(right_half, noise_threshold, intensity_threshold)

            left_page_image = left_half if not show_raw else raw_left_half
            right_page_image = right_half if not show_raw else raw_right_half

            processed_pages.append(left_page_image)
            processed_pages.append(right_page_image)
        else:
            raw_left_half = image.copy()

            if not show_raw:
                if deskew_option:
                    image = deskew_image(image)

                if remove_borders_option:
                    image = remove_borders(image)

                # Apply noise threshold and intensity (See Through) adjustments
                image = preprocess_image(image, noise_threshold, intensity_threshold)

            left_page_image = image if not show_raw else raw_left_half

            right_page_image = None

            processed_pages.append(left_page_image)

        # Initialize bounding boxes
        ocr_bounding_boxes_left = []
        ocr_bounding_boxes_right = []

        # If line segmentation is enabled
        if line_segmentation:
            if left_page_image is not None:
                left_image_name = f"page_{page_number + 1}_left.jpg"
                left_page_image, ocr_bounding_boxes_left = apply_line_segmentation(
                    left_page_image, padding, min_width, margin, threshold, left_image_name)
            if right_page_image is not None:
                right_image_name = f"page_{page_number + 1}_right.jpg"
                right_page_image, ocr_bounding_boxes_right = apply_line_segmentation(
                    right_page_image, padding, min_width, margin, threshold, right_image_name)
        else:
            ocr_bounding_boxes_left = []
            ocr_bounding_boxes_right = []

        return left_page_image, right_page_image, ocr_bounding_boxes_left, ocr_bounding_boxes_right

    except Exception as e:
        st.error(f"Error processing page {page_number + 1}: {e}")
        return None, None, [], []

def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
    """ Save text detection result one by one """
    img = np.array(img)
    filename, file_ext = os.path.splitext(os.path.basename(img_file))
    res_file = os.path.join(dirname, "res_" + filename + '.txt')
    res_img_file = os.path.join(dirname, "res_" + filename + '.jpg')

    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    with open(res_file, 'w') as f:
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            strResult = ','.join([str(p) for p in poly]) + '\r\n'
            f.write(strResult)

            poly = poly.reshape(-1, 2)
            cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
            ptColor = (0, 255, 255)
            if verticals is not None:
                if verticals[i]:
                    ptColor = (255, 0, 0)

            if texts is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1),
                            font, font_scale, (0, 0, 0), thickness=1)
                cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]),
                            font, font_scale, (0, 255, 255), thickness=1)

    # Save result image
    cv2.imwrite(res_img_file, img)

def apply_line_segmentation(image, padding, min_width, margin, threshold, page_image_name):
    # Check if CRAFT outputs are already cached
    if page_image_name in st.session_state['craft_outputs']:
        contours = st.session_state['craft_outputs'][page_image_name]['contours']
    else:
        net, device, refine_net = load_craft_model()
        
        # Convert image to RGB if necessary
        if len(image.shape) == 2 or image.shape[2] == 1:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run the CRAFT model
        text_threshold = 0.7
        link_threshold = 0.4
        low_text = 0.4
        poly = True  # Since we're using refine_net
        
        boxes, polys = test_net(
            net, 
            image_rgb, 
            text_threshold, 
            link_threshold, 
            low_text,
            cuda=(device.type == 'cuda'), 
            poly=poly, 
            device=device, 
            refine_net=refine_net
        )
    
        # Use saveResult to save the detection results
        with tempfile.TemporaryDirectory() as temp_dir:
            img_file = os.path.join(temp_dir, page_image_name)
            cv2.imwrite(img_file, image_rgb[:, :, ::-1])  # Save the image in BGR format
        
            # Save results using saveResult
            saveResult(img_file, image_rgb[:, :, ::-1], polys, dirname=temp_dir)
        
            # Read contours from the saved text file
            res_file = os.path.join(temp_dir, "res_" + os.path.splitext(page_image_name)[0] + '.txt')
            contours = read_contour_points(res_file)
        
        # Cache the contours
        st.session_state['craft_outputs'][page_image_name] = {'contours': contours}
    
    # Proceed with line segmentation using the contours
    img_height, img_width = image.shape[:2]
    
    # Get bounding boxes
    bounding_boxes = get_bounding_boxes(
        contours,
        img_width,
        img_height,
        padding=padding,
        min_width=min_width,
        margin=margin
    )
    
    # Split bounding boxes
    new_bounding_boxes = split_bounding_boxes(image, bounding_boxes, threshold=threshold)
    
    # Filter and adjust bounding boxes
    adjusted_bounding_boxes = filter_and_adjust_bounding_boxes(new_bounding_boxes)
    
    # Sort bounding boxes by their top (y1) coordinate to maintain order
    sorted_bounding_boxes = sorted(adjusted_bounding_boxes, key=lambda box: box[1])
    
    # Draw bounding boxes on the image
    draw_bounding_boxes(image, sorted_bounding_boxes, color=(0, 255, 0))
    
    return image, sorted_bounding_boxes

def generate_text_from_image_segment(image_segment):
    # Check if image is valid before processing
    if image_segment is None or image_segment.size == 0:
        return "[Empty image segment]"
    
    try:
        if len(image_segment.shape) == 2 or image_segment.shape[2] == 1:
            image_rgb = cv2.cvtColor(image_segment, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image_segment, cv2.COLOR_BGR2RGB)
            
        image_pil = Image.fromarray(image_rgb)
        pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    except Exception as e:
        st.warning(f"Error processing image segment: {e}")
        return "[Error processing image]"

def transcribe_and_display_ocr(left_image, right_image, ocr_bounding_boxes_left, ocr_bounding_boxes_right):
    texts = []
    # Transcribe lines from the left page if available
    if left_image is not None:
        if ocr_bounding_boxes_left:
            for (x1, y1, x2, y2) in ocr_bounding_boxes_left:
                h, w = left_image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                line_segment = left_image[y1:y2, x1:x2]
                text = generate_text_from_image_segment(line_segment)
                texts.append(text)
        else:
            text = generate_text_from_image_segment(left_image)
            texts.append(text)
    # Transcribe lines from the right page if available
    if right_image is not None:
        if ocr_bounding_boxes_right:
            for (x1, y1, x2, y2) in ocr_bounding_boxes_right:
                h, w = right_image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                line_segment = right_image[y1:y2, x1:x2]
                text = generate_text_from_image_segment(line_segment)
                texts.append(text)
        else:
            text = generate_text_from_image_segment(right_image)
            texts.append(text)
    return texts

# Main code

# Add a logo above the title
logo_url = "https://humanai.foundation/activities/gsoc2024.html"  # Replace with the URL you want to link to
online_logo_url = "https://humanai.foundation/images/CERN-HSF-GSoC-logo.png"  # Replace with the online URL of your logo

# Initialize cache and tracking variables
if 'craft_outputs' not in st.session_state:
    st.session_state['craft_outputs'] = {}

if 'previous_line_segmentation' not in st.session_state:
    st.session_state['previous_line_segmentation'] = False

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 0

if 'previous_page' not in st.session_state:
    st.session_state['previous_page'] = st.session_state['current_page']

if 'uploaded_file_name' not in st.session_state:
    st.session_state['uploaded_file_name'] = None

st.sidebar.header("PDF Upload")
uploaded_file = st.sidebar.file_uploader("Select a PDF file", type=["pdf"])


# Sidebar options
st.sidebar.header("Processing Options")

dpi_var = st.sidebar.slider("DPI:", min_value=50, max_value=300, value=100, step=1)

noise_threshold_var = st.sidebar.slider("Noise Threshold:", min_value=0, max_value=100, value=10, step=1)

intensity_threshold_var = st.sidebar.slider("See Through:", min_value=0, max_value=255, value=128, step=1)

remove_borders_var = st.sidebar.checkbox("Remove Borders", value=False)

deskew_var = st.sidebar.checkbox("Deskew", value=False)

show_raw_var = st.sidebar.checkbox("Show Raw Pages", value=False)

line_segmentation_var = st.sidebar.checkbox("Line Segmentation", value=False)

if line_segmentation_var != st.session_state['previous_line_segmentation']:
    st.session_state['craft_outputs'].clear()
    st.session_state['previous_line_segmentation'] = line_segmentation_var

if line_segmentation_var:
    st.sidebar.header("Line Segmentation Parameters")
    padding_var = st.sidebar.slider("Padding", min_value=0, max_value=500, value=20, step=1)
    min_width_var = st.sidebar.slider("Min Width", min_value=0, max_value=500, value=100, step=1)
    margin_var = st.sidebar.slider("Margin", min_value=0.0, max_value=2.0, value=0.1, step=0.01)
    threshold_var = st.sidebar.slider("Threshold", min_value=0.0, max_value=2.0, value=0.2, step=0.01)
else:
    padding_var = 20
    min_width_var = 100
    margin_var = 0.1
    threshold_var = 0.2

# Navigation
st.sidebar.header("Navigation")

if 'total_pages' in st.session_state:
    prev, page_num, next = st.sidebar.columns([1,2,1])
    with prev:
        if st.sidebar.button('Previous Page'):
            if st.session_state['current_page'] > 0:
                st.session_state['current_page'] -= 1
    with page_num:
        page_number = st.sidebar.number_input(
            'Page Number',
            min_value=1,
            max_value=st.session_state['total_pages'],
            value=st.session_state['current_page'] + 1,
            step=1
        )
        st.session_state['current_page'] = int(page_number) - 1
    with next:
        if st.sidebar.button('Next Page'):
            if st.session_state['current_page'] < st.session_state['total_pages'] - 1:
                st.session_state['current_page'] += 1
else:
    st.sidebar.info("Upload a PDF to enable navigation.")

def get_virtual_page_count(pdf_document, dpi, **kwargs):
    count = 0
    for i in range(pdf_document.page_count):
        page = pdf_document.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        if width > height:
            count += 2
        else:
            count += 1
    return count

def get_virtual_page(pdf_document, virtual_index, dpi, **kwargs):
    page_idx = 0
    v_idx = 0
    for i in range(pdf_document.page_count):
        page = pdf_document.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        if width > height:
            # Double page: left then right
            if v_idx == virtual_index:
                return image[:, :width // 2], f"Page {i+1} Left"
            v_idx += 1
            if v_idx == virtual_index:
                return image[:, width // 2:], f"Page {i+1} Right"
            v_idx += 1
        else:
            # Single page
            if v_idx == virtual_index:
                return image, f"Page {i+1}"
            v_idx += 1
    return None, "No page"

if uploaded_file is not None:
    if 'pdf_document' not in st.session_state or st.session_state['uploaded_file_name'] != uploaded_file.name:
        try:
            st.session_state['pdf_document'] = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
            st.session_state['uploaded_file_name'] = uploaded_file.name
            st.success(f"PDF loaded: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
            st.stop()

    # Calculate total virtual pages
    if 'total_virtual_pages' not in st.session_state:
        st.session_state['total_virtual_pages'] = get_virtual_page_count(
            st.session_state['pdf_document'], dpi_var
        )
        st.session_state['virtual_page_index'] = 0

    # Navigation for virtual pages
    prev, page_num, next = st.sidebar.columns([1,2,1])
    with prev:
        if st.sidebar.button('Previous'):
            if st.session_state['virtual_page_index'] > 0:
                st.session_state['virtual_page_index'] -= 1
    with page_num:
        vpage = st.sidebar.number_input(
            'Virtual Page',
            min_value=1,
            max_value=st.session_state['total_virtual_pages'],
            value=st.session_state['virtual_page_index'] + 1,
            step=1
        )
        st.session_state['virtual_page_index'] = int(vpage) - 1
    with next:
        if st.sidebar.button('Next'):
            if st.session_state['virtual_page_index'] < st.session_state['total_virtual_pages'] - 1:
                st.session_state['virtual_page_index'] += 1

    # Get current virtual page image
    raw_image, page_label = get_virtual_page(
        st.session_state['pdf_document'],
        st.session_state['virtual_page_index'],
        dpi_var
    )

    # Apply processing options
    processed_image = raw_image.copy() if raw_image is not None else None
    if processed_image is not None and not show_raw_var:
        if deskew_var:
            processed_image = deskew_image(processed_image)
        if remove_borders_var:
            processed_image = remove_borders(processed_image)
        processed_image = preprocess_image(processed_image, noise_threshold_var, intensity_threshold_var)

    # Apply line segmentation if enabled
    ocr_bounding_boxes = []
    if processed_image is not None and line_segmentation_var:
        processed_image, ocr_bounding_boxes = apply_line_segmentation(
            processed_image, padding_var, min_width_var, margin_var, threshold_var, f"virtual_{st.session_state['virtual_page_index']}.jpg"
        )

        # Draw bounding boxes just before display
    if processed_image is not None and len(processed_image.shape) == 2:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

    # Draw bounding boxes just before display
    if processed_image is not None and line_segmentation_var and ocr_bounding_boxes:
        draw_bounding_boxes(processed_image, ocr_bounding_boxes, color=(0, 255, 0))

    st.markdown(f"**{page_label}**")
    if processed_image is not None:
        img_col, ocr_col = st.columns([2, 3])
        with img_col:
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption=page_label, use_container_width=True)
        with ocr_col:
            st.markdown("### OCR Output")
            run_ocr = st.button('Run OCR on This Page', key=f'ocr_{st.session_state["virtual_page_index"]}')
            if run_ocr:
                with st.spinner('Performing OCR...'):
                    texts = transcribe_and_display_ocr(
                        processed_image, None, ocr_bounding_boxes, []
                    )
                    for t in texts:
                        st.write(t)
            else:
                st.info("Click the button above to run OCR on this page.")
    else:
        st.write("No image to display.")

else:
    st.info("Please upload a PDF file from the left panel.")