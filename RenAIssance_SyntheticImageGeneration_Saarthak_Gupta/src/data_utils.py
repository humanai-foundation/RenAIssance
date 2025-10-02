# Standard library imports
import gc
import glob
import io
import os
import random
import re
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import os
import random
from pathlib import Path

from tqdm import tqdm
import string
import subprocess
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

# Third-party data processing
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Image processing and computer vision
import cv2
from PIL import Image
import pytesseract
from skimage import restoration, exposure

# Deep learning and AI models
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Document processing
import fitz  # PyMuPDF
from docx import Document
import time

# Visualization
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


#################################################################################################################################
                                        # Splitting PDF Pages into Images
#################################################################################################################################

def display_sample(dir, sample_number=5):

    sample_path = os.path.join(dir, f"{sample_number}.png")

    if os.path.exists(sample_path):
        img = Image.open(sample_path)
        plt.figure(figsize=(4, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Sample Image: {sample_number}.png")
        plt.show()
    else:
        print(f"Image {sample_number}.png not found in '{dir}'. \n")
        
def pdf_to_images(pdf_path, output_dir, page_numbers=None, book_number=1, progress_bar=None):
    # Create book-specific folder
    book_folder = os.path.join(output_dir, f"book{book_number}")
    os.makedirs(book_folder, exist_ok=True)
    
    pdf_doc = fitz.open(pdf_path)
    image_counter = 1 
    processed_pages = 0
    book_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # If page_numbers is provided, convert to 0-indexed and filter
    if page_numbers:
        # Convert 1-indexed page numbers to 0-indexed for fitz
        pages_to_process = [p - 1 for p in page_numbers if 1 <= p <= len(pdf_doc)]
        if progress_bar:
            progress_bar.set_postfix({"Status": f"Processing {len(pages_to_process)} pages from '{book_name}'"})
    else:
        # Process all pages
        pages_to_process = range(len(pdf_doc))
        if progress_bar:
            progress_bar.set_postfix({"Status": f"Processing all {len(pdf_doc)} pages from '{book_name}'"})

    for page_num in pages_to_process:
        page = pdf_doc.load_page(page_num)
        pix = page.get_pixmap()  # Render page as image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # If the page is significantly wider than tall, assume it's a double-page
        if img.width > img.height * 1.19:
            left = img.crop((0, 0, img.width // 2, img.height))  # Left half
            right = img.crop((img.width // 2, 0, img.width, img.height))  # Right half

            # Save with page number in filename in book-specific folder
            left_filename = f"page{image_counter}.png"
            right_filename = f"page{image_counter + 1}.png"
            
            left.save(os.path.join(book_folder, left_filename), format="PNG")
            image_counter += 1
            right.save(os.path.join(book_folder, right_filename), format="PNG")
            image_counter += 1
        else:
            # Save single page with page number in filename in book-specific folder
            filename = f"page{image_counter}.png"
            img.save(os.path.join(book_folder, filename), format="PNG")
            image_counter += 1

        processed_pages += 1

    pdf_doc.close()
    if progress_bar:
        progress_bar.set_postfix({"Status": f"Completed book {book_number}: {image_counter - 1} images from {processed_pages} pages"})
        
def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def extract_text_by_page(docx_file, output_dir, book_number=1, progress_bar=None):
    # Create book-specific folder
    book_folder = os.path.join(output_dir, f"book{book_number}")
    os.makedirs(book_folder, exist_ok=True)
    
    document = Document(docx_file)
    pages = []
    page_numbers = []  # List to store original page numbers only
    current_page = []
    start_reading = False  # Flag to track when to start reading
    consecutive_empty_lines = 0
    found_pdf_page = False
    book_name = os.path.splitext(os.path.basename(docx_file))[0]
    
    if progress_bar:
        progress_bar.set_postfix({"Status": f"Extracting text from '{book_name}'"})
    
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if not text:  # Empty line found
            consecutive_empty_lines += 1
            
            # Check if we have two consecutive empty lines after any PDF page marker
            if found_pdf_page and consecutive_empty_lines >= 2:
                # Create a blank page
                pages.append("")  # Empty page
                found_pdf_page = False  # Reset flag
                consecutive_empty_lines = 0
            
            if start_reading and current_page:
                # Save current page and start a new one due to empty line
                pages.append("\n".join(current_page))
                current_page = []
            continue
        else:
            consecutive_empty_lines = 0  # Reset counter when non-empty line is found

        # Check for page number pattern BEFORE checking start_reading flag
        page_match = re.search(r'PDF\s+p(\d+)', text, re.IGNORECASE)
        if page_match:
            page_num = int(page_match.group(1))
            page_numbers.append(page_num)  # This maintains original page numbering
            
            # Set flag for any PDF page marker
            found_pdf_page = True
            consecutive_empty_lines = 0  # Reset counter

        if (text == "****" or "pdf" in text.lower()) and not start_reading:
            start_reading = True  # Start reading after the first "****"
            continue

        if not start_reading:
            continue  # Skip lines before the first "****"

        if text == "****" or "pdf" in text.lower() or "end of extract" in text.lower() or "--this part intentionally left blank to check after test--" in text.lower():
            # Page separator found
            if current_page:
                pages.append("\n".join(current_page))  # Save current page
                current_page = []  # Start a new page
        else:
            text = remove_punctuation(text)  # Remove punctuation
            current_page.append(text)

    # Append the last page if it's not empty
    if current_page:
        pages.append("\n".join(current_page))

    # Save each page separately in its own text file in book-specific folder
    for idx, page_text in enumerate(pages, start=1):
        page_file = os.path.join(book_folder, f"page{idx}.txt")
        with open(page_file, "w", encoding="utf-8") as f:
            f.write(page_text)
    
    return page_numbers

def process_books_with_transcripts(input_books_folder, input_transcripts_folder, output_books_folder, output_transcripts_folder):
    book_counter = 1
    
    # Create main output directories
    os.makedirs(output_books_folder, exist_ok=True)
    os.makedirs(output_transcripts_folder, exist_ok=True)
    
    # Get file lists
    pdf_files = sorted(os.listdir(input_books_folder))
    transcript_files = sorted(os.listdir(input_transcripts_folder))
    
    # Create a single progress bar
    with tqdm(total=len(pdf_files), desc="Splitting Images and transcripts", unit="book") as pbar:
        for pdf_file, transcript_file in zip(pdf_files, transcript_files):
            pdf_input_path = os.path.join(input_books_folder, pdf_file)
            transcript_input_path = os.path.join(input_transcripts_folder, transcript_file)
            
            # Extract text first to get page numbers
            page_numbers = extract_text_by_page(transcript_input_path, output_transcripts_folder, book_counter, pbar)
            
            # Process PDF with the book number
            pdf_to_images(pdf_input_path, output_books_folder, page_numbers, book_counter, pbar)
            
            book_counter += 1
            
            # Update progress bar
            pbar.set_postfix({"Status": f"Completed book {book_counter-1}"})
            pbar.update(1)
#################################################################################################################################
                                        # Image Preprocessing
#################################################################################################################################

def load_image(image_path):
    # Convert Path object to string for cv2.imread
    image = cv2.imread(str(image_path))  # Load image (default is BGR)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return image

def convert_to_grayscale(image):
    """Converts an image to grayscale if it's not already."""
    if len(image.shape) == 3:  # Check if the image is in color (BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # Already grayscale

def correct_skew(image):
    """Corrects skew in an image, supporting both grayscale and RGB inputs."""
    # Store original color state
    is_color = len(image.shape) == 3
    
    # Create grayscale copy for skew detection
    if is_color:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No text found, skipping skew correction.")
        return image  # Return original image with 0° skew angle

    # Find the largest contour (assumed to be text block)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the minimum area bounding box
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]

    # Normalize the angle to be between -45° and +45°
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    # Get image dimensions and center
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation with border replication
    deskewed = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return deskewed

def normalize_image(image):
    # Check if the image is RGB
    if len(image.shape) == 3:
        r, g, b = cv2.split(image)
        r_norm = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
        g_norm = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        b_norm = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
        normalized = cv2.merge([r_norm, g_norm, b_norm])
        return normalized
    else:
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
def ensure_300ppi(image, target_dpi=300):
    
    height, width = image.shape[:2]

    # Assume A4 document size in inches (common for scanned books)
    a4_width = 8.27  # inches
    a4_height = 11.69  # inches

    dpi_x = width / a4_width
    dpi_y = height / a4_height
    
    # Convert to PIL image, preserving color if needed
    image_pil = Image.fromarray(image)

    if dpi_x < target_dpi or dpi_y < target_dpi:
        # Calculate upscale factor
        scale_factor = target_dpi / min(dpi_x, dpi_y)  # Scale based on the lower DPI

        # Compute new size
        new_size = (int(image_pil.width * scale_factor), int(image_pil.height * scale_factor))

        # Resize using high-quality Lanczos resampling
        image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)

        # Set the new DPI metadata
        image_pil.info['dpi'] = (target_dpi, target_dpi)

    # Convert back to OpenCV format
    image_upscaled = np.array(image_pil)

    return image_upscaled

def remove_bleed_dual_layer(image):
    """Removes bleed-through, supporting both grayscale and RGB inputs."""
    # Check if image is color
    is_color = len(image.shape) == 3
    
    if is_color:
        # Convert to LAB color space which separates luminance from color
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab_image)
        
        # Apply processing to luminance channel
        kernel = np.ones((51, 51), np.uint8)
        background = cv2.dilate(l, kernel, iterations=2)
        background = cv2.medianBlur(background, 21)
        
        # Subtract background to get foreground
        l_processed = 255 - cv2.absdiff(l, background)
        
        # Normalize to improve contrast
        l_processed = cv2.normalize(l_processed, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # Merge channels back
        result_lab = cv2.merge([l_processed, a, b])
        
        # Convert back to RGB
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
    else:
        # Original grayscale processing
        kernel = np.ones((51, 51), np.uint8)
        background = cv2.dilate(image, kernel, iterations=2)
        background = cv2.medianBlur(background, 21)
        
        # Subtract background to get foreground
        diff = 255 - cv2.absdiff(image, background)
        
        # Normalize to improve contrast
        result = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    return result

def denoise_image(image, method="nlm"):

    # Check if input is valid
    if image is None or image.size == 0:
        print("Error: Empty input image")
        return None
    
    # Ensure image is in uint8 format for OpenCV functions
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Make a copy to avoid modifying the original
    image_copy = image.copy()
    is_color = len(image.shape) == 3
 
    if method == "bilateral":
        # Bilateral filter preserves edges while removing noise
        if is_color:
            return cv2.bilateralFilter(image_copy, 9, 75, 75)
        else:
            return cv2.bilateralFilter(image_copy, 9, 75, 75)
    
    elif method == "nlm":
        # Non-local means denoising
        if is_color:
            return cv2.fastNlMeansDenoisingColored(image_copy, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image_copy, None, 10, 7, 21)
    
    elif method == "wiener":
        # Wiener filter (needs float conversion)
        if is_color:
            # Process each channel separately
            channels = cv2.split(image_copy.astype(np.float32) / 255.0)
            restored_channels = []
            
            for channel in channels:
                restored = restoration.wiener(channel, psf=np.ones((3, 3)) / 9, balance=0.3)
                restored = np.clip(restored * 255, 0, 255).astype(np.uint8)
                restored_channels.append(restored)
            
            return cv2.merge(restored_channels)
        else:
            float_img = image_copy.astype(np.float32) / 255.0
            restored = restoration.wiener(float_img, psf=np.ones((3, 3)) / 9, balance=0.3)
            return np.clip(restored * 255, 0, 255).astype(np.uint8)
        
def sharpen_image(image, method="laplacian"):
    is_color = len(image.shape) == 3
    
    if method == "laplacian":
        if is_color:
            # Process each channel separately
            r, g, b = cv2.split(image)
            r_sharp = cv2.Laplacian(r, cv2.CV_8U)
            g_sharp = cv2.Laplacian(g, cv2.CV_8U)
            b_sharp = cv2.Laplacian(b, cv2.CV_8U)
            
            r_result = cv2.add(r, r_sharp)
            g_result = cv2.add(g, g_sharp)
            b_result = cv2.add(b, b_sharp)
            
            return cv2.merge([r_result, g_result, b_result])
        else:
            laplacian = cv2.Laplacian(image, cv2.CV_8U)
            return cv2.add(image, laplacian)
    
    elif method == "custom":
        # Custom sharpening kernel
        kernel = np.array([[0, -2, 0], 
                          [-2, 9, -2],
                          [0, -2, 0]])
        
        if is_color:
            r, g, b = cv2.split(image)
            r_sharp = cv2.filter2D(r, -1, kernel)
            g_sharp = cv2.filter2D(g, -1, kernel)
            b_sharp = cv2.filter2D(b, -1, kernel)
            return cv2.merge([r_sharp, g_sharp, b_sharp])
        else:
            return cv2.filter2D(image, -1, kernel)
    
    elif method == "unsharp_mask":
        # Unsharp masking - better for color images
        if is_color:
            # Convert to LAB to separate luminance
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b_channel = cv2.split(lab)
            
            # Apply unsharp mask to luminance only
            gaussian = cv2.GaussianBlur(l, (0, 0), 3.0)
            unsharp_mask = cv2.addWeighted(l, 1.5, gaussian, -0.5, 0)
            
            # Merge back
            result = cv2.merge([unsharp_mask, a, b_channel])
            return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale
            gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
            return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

    return image

def enhance_contrast(image, method="clahe"):
    """Enhances contrast in an image, supporting both grayscale and RGB inputs."""
    is_color = len(image.shape) == 3
    
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        if is_color:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel only
            l_enhanced = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale
            return clahe.apply(image)
    
    elif method == "adaptive_eq":
        if is_color:
            # Convert to HSV to separate value from hue
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            
            # Apply adaptive eq to value channel
            v_enhanced = exposure.equalize_adapthist(v, clip_limit=0.03) * 255
            v_enhanced = v_enhanced.astype(np.uint8)
            
            # Merge channels and convert back to RGB
            enhanced_hsv = cv2.merge([h, s, v_enhanced])
            return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
        else:
            # For grayscale
            result = exposure.equalize_adapthist(image, clip_limit=0.03) * 255
            return result.astype(np.uint8)
    
    elif method == "stretch":
        if is_color:
            # Apply to each channel with care to preserve color relationships
            r, g, b = cv2.split(image)
            
            # Get global percentiles for consistent scaling
            low = np.min([np.percentile(r, 2), np.percentile(g, 2), np.percentile(b, 2)])
            high = np.max([np.percentile(r, 98), np.percentile(g, 98), np.percentile(b, 98)])
            
            # Scale each channel with the same limits
            r_stretched = np.clip((r - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
            g_stretched = np.clip((g - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
            b_stretched = np.clip((b - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
            
            return cv2.merge([r_stretched, g_stretched, b_stretched])
        else:
            # For grayscale
            p2, p98 = np.percentile(image, (2, 98))
            result = exposure.rescale_intensity(image, in_range=(p2, p98))
            return result.astype(np.uint8)
    
    return image

def morphological_operations(image, operation, kernel_size=(2, 2), iterations=1):
    is_color = len(image.shape) == 3
    kernel = np.ones(kernel_size, np.uint8)
    
    if is_color:
        r, g, b = cv2.split(image)
        
        if operation == "open":
            r_processed = cv2.morphologyEx(r, cv2.MORPH_OPEN, kernel, iterations=iterations)
            g_processed = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel, iterations=iterations)
            b_processed = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == "close":
            r_processed = cv2.morphologyEx(r, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            g_processed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            b_processed = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif operation == "dilate":
            r_processed = cv2.dilate(r, kernel, iterations=iterations)
            g_processed = cv2.dilate(g, kernel, iterations=iterations)
            b_processed = cv2.dilate(b, kernel, iterations=iterations)
        elif operation == "erode":
            r_processed = cv2.erode(r, kernel, iterations=iterations)
            g_processed = cv2.erode(g, kernel, iterations=iterations)
            b_processed = cv2.erode(b, kernel, iterations=iterations)
        else:
            return image
        
        return cv2.merge([r_processed, g_processed, b_processed])
    else:
        if operation == "open":
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == "close":
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif operation == "dilate":
            return cv2.dilate(image, kernel, iterations=iterations)
        elif operation == "erode":
            return cv2.erode(image, kernel, iterations=iterations)
    
    return image

def binarize_image(image, method="otsu"):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    if method == "otsu":
        # Otsu's method for global thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    elif method == "adaptive":
        # Adaptive thresholding - good for uneven illumination
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 15, 8)
    
    return binary

def apply_binary_mask(color_image, binary_mask):
    # Ensure binary mask is properly formatted
    if len(binary_mask.shape) == 3:
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_RGB2GRAY)
    
    # Create a 3-channel mask for direct multiplication
    mask_3channel = cv2.merge([binary_mask, binary_mask, binary_mask])
    
    # Normalize mask to 0-1 range for multiplication
    mask_normalized = mask_3channel / 255.0
    
    # Apply mask to color image
    masked_image = (color_image * mask_normalized).astype(np.uint8)
    
    return masked_image

def upscale(image, ppi_threshold=150, assumed_width_inches=8.5, 
                             model_path="models/RealESRGAN_x4plus.pth", 
                             scale=4, force_upscale=False):

    # Calculate current PPI
    height, width = image.shape[:2]
    current_ppi = width / assumed_width_inches
    
    # Check if upscaling is needed
    if not force_upscale and current_ppi >= ppi_threshold:
        print(f"Image PPI ({current_ppi:.1f}) is above threshold ({ppi_threshold}). Skipping upscale.")
        return image
    
    try:
        # Initialize upscaler (cache this in real implementation)
        if not hasattr(upscale, '_upsampler'):
            
            # Load model state dict
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            state_dict = torch.load(model_path, map_location=device)
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']
            
            # Initialize model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            
            # Create upsampler
            upscale._upsampler = RealESRGANer(
                scale=scale,
                model_path=model_path,
                model=model,
                tile=0,
                pre_pad=0,
                half=False
            )
            
        output_array, _ = upscale._upsampler.enhance(image, outscale=scale)
        
        return output_array
        
    except Exception as e:
        print(f"Error during upscaling: {e}")
        return image

def process_book_with_transformations(book_directory, output_dir, transformations=None, pbar=None):

    # Default transformation sequence if none provided
    if transformations is None:
        transformations = ['correct_skew', 'ensure_300ppi']
    
    def safe_cuda_cleanup():
        """Safely clear CUDA memory with proper synchronization"""
        try:
            if torch.cuda.is_available():
                # Wait for all CUDA operations to complete
                torch.cuda.synchronize()
                # Small delay to ensure operations are truly finished
                time.sleep(0.1)
                # Clear the cache
                torch.cuda.empty_cache()
        except Exception as e:
            # Don't crash if CUDA cleanup fails
            print(f"Warning: CUDA cleanup failed: {e}")

    def safe_delete_image(image_var):
        """Safely delete image variable with proper checks"""
        try:
            if image_var is not None:
                # If it's a torch tensor, move to CPU first
                if hasattr(image_var, 'cpu'):
                    image_var = image_var.cpu()
                del image_var
        except Exception as e:
            print(f"Warning: Failed to delete image variable: {e}")

    
    # Available transformation functions
    transform_functions = {
        'correct_skew': correct_skew,
        'normalize_image': normalize_image,
        'ensure_300ppi': ensure_300ppi,
        'remove_bleed_dual_layer': remove_bleed_dual_layer,
        'denoise_image': denoise_image,
        'sharpen_image': sharpen_image,
        'enhance_contrast': enhance_contrast,
        'morphological_operations': morphological_operations,
        'binarize_image': binarize_image,
        'apply_binary_mask': apply_binary_mask,
        'upscale': upscale,
    }
    
    # Handle both list and dictionary input formats
    if isinstance(transformations, dict):
        transform_config = transformations
        transform_order = list(transformations.keys())
    else:
        # Convert list to dictionary with empty parameters
        transform_config = {t: {} for t in transformations}
        transform_order = transformations
    
    # Validate transformations
    invalid_transforms = [t for t in transform_order if t not in transform_functions]
    if invalid_transforms:
        raise ValueError(f"Invalid transformations: {invalid_transforms}")
    
    # Extract book number from directory path
    book_name = os.path.basename(book_directory.rstrip('/\\'))
    book_number_match = re.search(r'(\d+)', book_name)
    if book_number_match:
        book_number = book_number_match.group(1)
    else:
        # Fallback: use the entire directory name if no number found
        book_number = book_name
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of page image files and sort them numerically
    image_files = []
    for f in os.listdir(book_directory):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            # Extract page number for sorting
            page_match = re.search(r'page(\d+)', f.lower())
            if page_match:
                page_num = int(page_match.group(1))
                image_files.append((page_num, f))
    
    # Sort by page number
    image_files.sort(key=lambda x: x[0])
    
    if not image_files:
        print(f"No page image files found in {book_directory}.")
        return {'processed': 0, 'errors': 0, 'total': 0, 'book_number': book_number}
    
    # Statistics
    stats = {
        'processed': 0, 
        'errors': 0, 
        'total': len(image_files), 
        'book_number': book_number
    }
    error_log = []
    
    # Process images
    for page_num, image_name in image_files:
        image_path = os.path.join(book_directory, image_name)
        current_image = None
        previous_image = None
        
        try:
            # Load original image
            current_image = load_image(image_path)
            
            # Apply transformations sequentially
            for i, transform_name in enumerate(transform_order):
                transform_func = transform_functions[transform_name]
                transform_params = transform_config[transform_name]
                
                # Store reference to previous image for cleanup
                if i > 0:
                    previous_image = current_image
                
                # Apply transformation with parameters
                try:
                    if transform_params:
                        current_image = transform_func(current_image, **transform_params)
                    else:
                        current_image = transform_func(current_image)
                except Exception as transform_error:
                    print(f"Warning: Transformation {transform_name} failed: {transform_error}")
                    # Continue with the previous image if transformation fails
                    if previous_image is not None:
                        current_image = previous_image
                
                # Safely clean up previous image
                if i > 0 and previous_image is not None:
                    safe_delete_image(previous_image)
                    previous_image = None
                
                # Memory cleanup after intensive operations
                if transform_name in ['upscale', 'denoise_image', 'remove_bleed_dual_layer']:
                    # Wait a moment for operations to complete
                    time.sleep(0.05)
                    gc.collect()
                    safe_cuda_cleanup()
                
                # Update progress bar description if provided
                if pbar:
                    pbar.set_postfix({
                        'book': book_number,
                        'page': f"{page_num}",
                        'step': f"{i+1}/{len(transform_order)}",
                        'transform': transform_name
                    })
            
            # Generate output filename: book_{book_number}_{page_number}.png
            output_filename = f"book_{book_number}_{page_num}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Ensure we have a valid image to save
            if current_image is not None:
                # Save final processed image
                cv2.imwrite(output_path, cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))
                
                # Safely clean up final image
                safe_delete_image(current_image)
                current_image = None
                
                stats['processed'] += 1
            else:
                raise ValueError("No valid image to save after transformations")
            
            # Gentle memory cleanup after each page
            gc.collect()
            safe_cuda_cleanup()
            
        except Exception as e:
            error_msg = f"Error processing {image_name} (page {page_num}): {str(e)}"
            error_log.append(error_msg)
            stats['errors'] += 1
            
            if pbar:
                pbar.set_postfix({'book': book_number, 'page': f"{page_num}", 'status': 'ERROR'})
        
        finally:
            # Always clean up any remaining references
            if current_image is not None:
                safe_delete_image(current_image)
            if previous_image is not None:
                safe_delete_image(previous_image)
            
            # Force cleanup in finally block
            gc.collect()
            safe_cuda_cleanup()
        
        # Update progress bar if provided
        if pbar:
            pbar.update(1)
    
    # Clean up upscaler cache safely
    try:
        if hasattr(upscale, '_upsampler'):
            del upscale._upsampler
    except Exception as e:
        print(f"Warning: Failed to clean upscaler cache: {e}")
    
    # Final comprehensive cleanup
    gc.collect()
    safe_cuda_cleanup()
    
    if error_log:
        print(f"\nErrors encountered in Book {book_number}:")
        for error in error_log:
            print(f"  - {error}")
    
    return stats

def process_multiple_books(books_base_dir, output_dir, book_transformations=None):
    
    if not os.path.exists(books_base_dir):
        raise ValueError(f"Books base directory does not exist: {books_base_dir}")
    
    # Only process books that have transformations specified
    if not book_transformations:
        print("No book transformations specified. Nothing to process.")
        return {'books_processed': 0, 'total_pages': 0, 'total_errors': 0, 'book_results': {}}
    
    # Get all subdirectories and filter only those with transformations
    all_book_dirs = [d for d in os.listdir(books_base_dir) 
                     if os.path.isdir(os.path.join(books_base_dir, d))]
    
    # Filter to only include books that have transformations specified
    books_to_process = [book_dir for book_dir in all_book_dirs 
                       if book_dir in book_transformations]
    
    if not books_to_process:
        print(f"No books found in {books_base_dir} that match the transformation specifications.")
        print(f"Available books: {all_book_dirs}")
        print(f"Books with transformations: {list(book_transformations.keys())}")
        return {'books_processed': 0, 'total_pages': 0, 'total_errors': 0, 'book_results': {}}
    
    # Calculate total pages across all books to be processed
    total_pages = 0
    book_page_counts = {}
    
    for book_dir_name in books_to_process:
        book_path = os.path.join(books_base_dir, book_dir_name)
        # Count image files in this book
        image_files = []
        for f in os.listdir(book_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                page_match = re.search(r'page(\d+)', f.lower())
                if page_match:
                    image_files.append(f)
        
        book_page_counts[book_dir_name] = len(image_files)
        total_pages += len(image_files)
    
    print(f"Processing {len(books_to_process)} books with {total_pages} total pages:")
    for book_name in books_to_process:
        transformations = book_transformations[book_name]
    overall_stats = {
        'books_processed': 0,
        'total_pages': 0,
        'total_errors': 0,
        'book_results': {}
    }
    
    # Create single progress bar for all books
    with tqdm(total=total_pages, desc="Transforming page Images", unit="page") as pbar:
        for book_dir_name in sorted(books_to_process):
            book_path = os.path.join(books_base_dir, book_dir_name)
            transformations = book_transformations[book_dir_name]
            
            try:
                book_stats = process_book_with_transformations(
                    book_path, 
                    output_dir, 
                    transformations,
                    pbar=pbar  # Pass the progress bar
                )
                
                overall_stats['books_processed'] += 1
                overall_stats['total_pages'] += book_stats['processed']
                overall_stats['total_errors'] += book_stats['errors']
                overall_stats['book_results'][book_dir_name] = book_stats
                
            except Exception as e:
                print(f"Failed to process book {book_dir_name}: {str(e)}")
                overall_stats['total_errors'] += 1
                # Update progress bar for skipped pages
                if book_dir_name in book_page_counts:
                    pbar.update(book_page_counts[book_dir_name])
    
    return overall_stats

def copy_all_transcripts():
    
    """
    Copy all transcript files from book folders to destination with Book_{book_number}_{page_number} naming.
    """
    source_base = "data/GAN-DATA/2_splitted/transcripts"
    dest_folder = "data/GAN-DATA/3_processed/transcripts"
    
    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)
    
    base_path = Path(source_base)
    
    if not base_path.exists():
        print(f"Error: Source folder '{source_base}' does not exist.")
        return
    
    # Find all book folders
    book_folders = [f for f in base_path.iterdir() if f.is_dir() and f.name.startswith('book')]
    
    if not book_folders:
        print(f"No book folders found in '{source_base}'")
        return
    
    total_copied = 0
    
    for book_folder in sorted(book_folders):
        # Extract book number from folder name
        book_match = re.search(r'book(\d+)', book_folder.name, re.IGNORECASE)
        
        if not book_match:
            print(f"Warning: Could not extract book number from folder '{book_folder.name}', skipping...")
            continue
            
        book_number = book_match.group(1)
        
        # Get all files in the book folder
        files = [f for f in book_folder.iterdir() if f.is_file()]
        
        if not files:
            print(f"No files found in '{book_folder}'")
            continue
        
        book_copied = 0
        
        for file_path in files:
            filename = file_path.name
            
            # Extract page number from filename
            page_match = re.search(r'(?:page_?|transcript_page_?|^)(\d+)', filename, re.IGNORECASE)
            
            if page_match:
                page_number = page_match.group(1)
            else:
                # Use sequential numbering if no page number found
                page_number = str(book_copied + 1).zfill(3)
                print(f"Warning: Could not extract page number from '{filename}', using {page_number}")
            
            # Create new filename
            file_extension = file_path.suffix
            new_filename = f"Book_{book_number}_{page_number}{file_extension}"
            dest_file_path = Path(dest_folder) / new_filename
            
            try:
                shutil.copy2(file_path, dest_file_path)
                book_copied += 1
                total_copied += 1
            except Exception as e:
                print(f"Error copying {filename}: {str(e)}")
                
def copy_all_images():
    import gc
    
    source_base = "data/GAN-DATA/2_splitted/books"
    dest_folder = "data/GAN-DATA/2_splitted/images"
    
    os.makedirs(dest_folder, exist_ok=True)
    base_path = Path(source_base)
    book_folders = [f for f in base_path.iterdir() if f.is_dir() and f.name.startswith('book')]
    
    for book_folder in sorted(book_folders):
        book_match = re.search(r'book(\d+)', book_folder.name, re.IGNORECASE)
        book_number = book_match.group(1)
        files = [f for f in book_folder.iterdir() if f.is_file()]
        
        book_copied = 0
        for file_path in files:
            filename = file_path.name
            page_match = re.search(r'(?:page_?|image_page_?|^)(\d+)', filename, re.IGNORECASE)
            
            if page_match:
                page_number = page_match.group(1)
            else:
                page_number = str(book_copied + 1).zfill(3)
            
            file_extension = file_path.suffix
            new_filename = f"Book_{book_number}_{page_number}{file_extension}"
            dest_file_path = Path(dest_folder) / new_filename
            ################################################################
            img = load_image(file_path)  
            cv2.imwrite(str(dest_file_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Free memory after each image
            del img
            gc.collect()
            
            # Clear GPU memory if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ###############################################################    
            # shutil.copy2(file_path, dest_file_path)
            book_copied += 1
#################################################################################################################################
                                        # Text Detection and Bounding Boxes
#################################################################################################################################

def plot_bounding_boxes(image_path, text_file_path, save_path=None, figsize=(12, 8), line_width=2,show_image=True):

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert BGR to RGB for matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Read bounding boxes from text file
    bounding_boxes = []
    try:
        with open(text_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    coords = [float(c) for c in line.split(',')]
                    if len(coords) >= 8:  # Ensure we have at least 8 coordinates
                        bounding_boxes.append(coords[:8])  # Take first 8 coordinates
                    else:
                        print(f"Warning: Invalid bounding box format: {line}")
    except FileNotFoundError:
        print(f"Error: Could not read text file from {text_file_path}")
        return
    except Exception as e:
        print(f"Error reading bounding boxes: {e}")
        return
    
    if not bounding_boxes:
        print("No valid bounding boxes found in the text file")
        return
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)
    
    # Plot each bounding box
    for bbox in bounding_boxes:
        # Extract coordinates (x1,y1,x2,y2,x3,y3,x4,y4)
        x_coords = [bbox[0], bbox[2], bbox[4], bbox[6]]
        y_coords = [bbox[1], bbox[3], bbox[5], bbox[7]]
        
        # Create polygon points (connecting the four corners)
        polygon_points = list(zip(x_coords, y_coords))
        
        # Create and add the polygon patch
        polygon = Polygon(polygon_points, 
                         fill=False, 
                         edgecolor='red', 
                         linewidth=line_width,
                         alpha=0.8)
        ax.add_patch(polygon)
    
    # Set title and remove axes
    ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_image:  
        plt.show()

def text_detection(input_root, output_root,model_path):
    print("Generating bounding boxes for each book page...")
    os.makedirs(output_root, exist_ok=True)

    command = (
        f"python CRAFT-pytorch/test.py "
        f"--trained_model={model_path} "
        f"--result_folder={output_root} "
        f"--test_folder={input_root}"
    )

    subprocess.run(command, shell=True, check=True)
    
def plot_random_pages(images_dir="data/3_processed/books", 
                              bbox_dir="data/4_bounding_boxes", 
                              save_dir="assets"):
    # Create assets directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all image files
    image_files = glob.glob(os.path.join(images_dir, "book_*.png"))
    
    # Group files by book number
    books = defaultdict(list)
    for img_path in image_files:
        filename = os.path.basename(img_path)
        # Extract book number from filename (e.g., "book_2_2.png" -> book_number = 2)
        parts = filename.replace('.png', '').split('_')
        if len(parts) >= 2:
            try:
                book_number = int(parts[1])
                books[book_number].append(filename.replace('.png', ''))
            except ValueError:
                continue
    
    # For each book, randomly select one page and plot
    plot_counter = 1
    for book_number, pages in sorted(books.items()):
        # Randomly select one page from this book
        selected_page = random.choice(pages)
        
        image_path = os.path.join(images_dir, f"{selected_page}.png")
        bbox_path = os.path.join(bbox_dir, f"{selected_page}.txt")
        save_path = os.path.join(save_dir, f"plot{plot_counter}.png")
        
        # Check if both image and bbox files exist
        if os.path.exists(image_path) and os.path.exists(bbox_path):
            
            try:
                plot_bounding_boxes(
                    image_path=image_path,
                    text_file_path=bbox_path,
                    save_path=save_path,
                    show_image=False,  # Set to False to avoid showing the image immediately
                )
                plot_counter += 1
            except Exception as e:
                print(f"Error plotting {selected_page}: {e}")
        else:
            print(f"Missing files for {selected_page}")
            if not os.path.exists(image_path):
                print(f"  Missing image: {image_path}")
            if not os.path.exists(bbox_path):
                print(f"  Missing bbox: {bbox_path}")


#################################################################################################################################
                                        # Mapping Bounding Boxes to Transcript
#################################################################################################################################
pytesseract.pytesseract.tesseract_cmd = r".\models\Tesseract-OCR\tesseract.exe"

def similarity_score(a, b):
    """Calculate string similarity between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def map_bounding_boxes_to_transcript(image_path, bbox_path, transcript_path, output_path, tesseract_model=1, similarity_threshold=0.5):
    """Maps bounding boxes to transcript for a single image file."""
    # Read transcript
    with open(transcript_path, 'r') as f:
        transcript = f.read().strip()

    # Split transcript into words
    transcript_lines = transcript.split('\n')
    transcript_words = []
    for line in transcript_lines:
        transcript_words.extend(line.strip().split())

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image from {image_path}")
        return 0, 0, 0

    # Read bounding boxes
    with open(bbox_path, 'r') as f:
        bounding_boxes = [line.strip() for line in f.readlines() if line.strip()]

    # Result mappings
    mappings = []
    total_bbox_count = len(bounding_boxes)

    # Process each bounding box
    for bbox in bounding_boxes:
        try:
            # Parse the bounding box coordinates
            coords = [float(c) for c in bbox.split(',')]

            # Make sure we have enough coordinates
            if len(coords) < 8:
                print(f"Warning: Invalid bounding box format: {bbox}")
                continue

            x_coords = [coords[0], coords[2], coords[4], coords[6]]
            y_coords = [coords[1], coords[3], coords[5], coords[7]]

            # Get the rectangular region (min/max coordinates)
            x_min, y_min = int(min(x_coords)), int(min(y_coords))
            x_max, y_max = int(max(x_coords)), int(max(y_coords))

            # Ensure coordinates are within image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.shape[1], x_max)
            y_max = min(image.shape[0], y_max)

            # Skip if region is too small
            if x_max - x_min < 5 or y_max - y_min < 5:
                continue

            # Extract the region from the image
            roi = image[y_min:y_max, x_min:x_max]

            # Skip if ROI is empty
            if roi.size == 0:
                continue

            # Use pytesseract to get the text in this bounding box
            detected_text = pytesseract.image_to_string(roi,
                                                       config=f'--psm 7 --oem {tesseract_model} -l spa').strip()

            # Skip if no text detected
            if not detected_text:
                continue
            
            detected_text = detected_text.lower() 
            # Find the closest matching word in the transcript
            max_similarity = 0
            best_match = None

            for word in transcript_words:
                sim = similarity_score(detected_text, word)
                if sim > max_similarity:
                    max_similarity = sim
                    best_match = word

            # Only consider it a match if similarity is above threshold
            if max_similarity > similarity_threshold and best_match:
                # Add to mappings and remove matched word from transcript to avoid duplicates
                mappings.append((best_match, bbox, max_similarity))
                transcript_words.remove(best_match)
        except Exception as e:
            print(f"Error processing bounding box {bbox}: {e}")

    # Write results to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for word, bbox, _ in mappings:
            f.write(f"{word}\t{bbox}\n")

    return len(transcript_words) + len(mappings), len(mappings), total_bbox_count

def mapping_bounding_boxes(image_dir, bbox_dir, transcript_dir, output_dir, tesseract_model=1, similarity_threshold=0.5):
    """Processes all files for which transcripts are available and returns summary statistics."""
    from tqdm import tqdm
    total_words = 0
    total_mapped = 0
    total_bb = 0
    files_processed = 0
    no_transcript_files = 0

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if directories exist
    for dir_path in [image_dir, bbox_dir, transcript_dir]:
        if not os.path.exists(dir_path):
            print(f"Error: Directory not found: {dir_path}")
            return

    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    print(f"Processing {len(image_files)} files...")
    
    # Use tqdm for progress bar
    for image_file in tqdm(image_files, desc="Mapping bounding boxes", unit="file"):
        base_name = os.path.splitext(image_file)[0]

        image_path = os.path.join(image_dir, image_file)
        transcript_path = os.path.join(transcript_dir, f"{base_name}.txt")
        bbox_path = os.path.join(bbox_dir, f"{base_name}.txt")
        output_path = os.path.join(output_dir, f"{base_name}_mapped.txt")

        # Check if transcript exists
        if not os.path.exists(transcript_path):
            no_transcript_files += 1
            continue

        # Check if bbox file exists
        if not os.path.exists(bbox_path):
            continue

        # Process the files
        try:
            words_count, mapped_count, bbox_count = map_bounding_boxes_to_transcript(
                image_path, bbox_path, transcript_path, output_path,
                tesseract_model, similarity_threshold
            )

            total_words += words_count
            total_mapped += mapped_count
            total_bb += bbox_count
            files_processed += 1

        except Exception as e:
            print(f"Error processing file {base_name}: {e}")

    # Print summary statistics
    print(f"\n=== MAPPING SUMMARY ===")
    print(f"1. Total number of words in transcript files: {total_words}")
    print(f"2. Total number of bounding boxes detected: {total_bb}")
    print(f"3. Total number of image-word pairs successfully mapped: {total_mapped}")
    
    return total_words, total_bb, total_mapped

#################################################################################################################################
                                        # building dataset from mapped files
#################################################################################################################################

def extract_and_process_all_regions(image_root, aligned_root, output_root, csv_output_path):

    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    df_records = []
    global_index = 0
    
    # Get all alignment files
    alignment_files = [f for f in os.listdir(aligned_root) if f.endswith('_mapped.txt')]
    
    for alignment_file in tqdm(alignment_files, desc="Creating Word Dataset", unit="image"):
        # Construct paths
        image_file = alignment_file.replace("_mapped.txt", ".png")
        image_path = os.path.join(image_root, image_file)
        aligned_path = os.path.join(aligned_root, alignment_file)
        
        # Extract book name from image filename
        # Format: {book_name}_{page_number}.png
        book_name = "_".join(image_file.split("_")[:-1])  # Everything except the last part (page number)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        
        # Check if alignment file exists
        if not os.path.exists(aligned_path):
            print(f"Warning: Alignment file not found {aligned_path}")
            continue
        
        # Process alignment file
        with open(aligned_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split("\t") for line in f if "\t" in line]
        
        # Process each region in the current image
        for text, bbox_str in lines:
            output_filename = f"image{global_index}.png"
            output_path = os.path.join(output_root, output_filename)
            
            # Parse bounding box
            try:
                bbox = list(map(float, bbox_str.split(',')))
                pts = np.array(bbox, dtype=np.int32).reshape((4, 2))
                
                x_min, y_min = np.min(pts, axis=0)
                x_max, y_max = np.max(pts, axis=0)
            except (ValueError, IndexError):
                print(f"Warning: Invalid bounding box format: {bbox_str} - skipping")
                continue
            
            # Check if bounding box has valid dimensions
            if x_min >= x_max or y_min >= y_max:
                print(f"Warning: Invalid bounding box dimensions: {x_min},{y_min},{x_max},{y_max} - skipping")
                continue
            
            # Ensure box is within image boundaries
            height, width = image.shape[:2]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)
            
            # Check again after clamping to image boundaries
            if x_max - x_min < 2 or y_max - y_min < 2:
                continue
            
            # Crop region
            cropped_region = image[y_min:y_max, x_min:x_max]
            
            # Validate cropping
            if cropped_region.size == 0 or cropped_region.shape[0] == 0 or cropped_region.shape[1] == 0:
                print(f"Warning: Empty cropped region - skipping")
                continue
            
            # Save the cropped image
            cv2.imwrite(output_path, cropped_region)
            
            # Add to records with book name
            df_records.append((output_filename, text, book_name))
            global_index += 1
    
    # Create DataFrame and save CSV with book column
    df = pd.DataFrame(df_records, columns=["Image", "label", "book"])
    df.to_csv(csv_output_path, index=False)
    
    print(f"Processed {len(df_records)} regions from {len(alignment_files)} images")
    print(f"Images saved to: {output_root}")
    print(f"CSV saved to: {csv_output_path}")
    
    return df

def analyze_image_sizes(directory):
    """Analyze the minimum, maximum, and average sizes of all images in a directory."""
    # Lists to store dimensions and file sizes
    widths = []
    heights = []

    # Count valid images
    total_images = 0

    # Supported image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

    # Process each file in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Skip if not a file or not an image file
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_extensions:
            continue

        try:
            # Get image dimensions using PIL
            with Image.open(file_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
            total_images += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Calculate statistics
    if total_images > 0:
        # Dimensions
        min_width, max_width = min(widths), max(widths)
        min_height, max_height = min(heights), max(heights)
        avg_width = sum(widths) / total_images
        avg_height = sum(heights) / total_images

        # Print results
        print(f"Analyzed {total_images} images in {directory}")
        print(f"Dimensions (width x height):")
        print(f"  Minimum: {min_width} x {min_height} pixels")
        print(f"  Maximum: {max_width} x {max_height} pixels")
        print(f"  Average: {avg_width:.1f} x {avg_height:.1f} pixels")
    else:
        print(f"No valid images found in {directory}")

def resize_and_pad(input_dir, output_dir, target_height, target_width):

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    # Process images with progress bar
    for image_file in tqdm(image_files, desc="Resizing and Padding Images", unit="img"):
        try:
            # Load image
            img_path = os.path.join(input_dir, image_file)
            img = Image.open(img_path)
            
            # Calculate aspect ratio
            aspect = img.width / img.height

            # Determine new dimensions that fit within target size while preserving aspect ratio
            if aspect > (target_width / target_height):  # wider than target
                new_width = target_width
                new_height = int(new_width / aspect)
            else:  # taller than target
                new_height = target_height
                new_width = int(new_height * aspect)

            # Resize using LANCZOS for high quality
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)

            # Create new white image of target size
            padded_img = Image.new("RGB", (target_width, target_height), color="white")

            # Center the resized image
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            padded_img.paste(resized_img, (paste_x, paste_y))

            # Save the padded image
            output_path = os.path.join(output_dir, image_file)
            padded_img.save(output_path)
            
            # Close images to free memory
            img.close()
            padded_img.close()
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue
#################################################################################################################################
                                        # Rendering Latex Source Images
#################################################################################################################################

def generate_text_image_dataset(
    csv_path='data/6_word_data/words.csv',
    output_dir='data/final_dataset',
    target_dir_prefix='data/final_dataset/target/',
    image_width=384,
    image_height=128,
    base_fontsize=70,
    matplotlib_config=None,
    custom_font_path=None,  # New parameter for custom font
    progress_bar=True
): 
    # Configure matplotlib styling
    if matplotlib_config is None:
        matplotlib_config = {
            'text.usetex': False,
            'mathtext.fontset': 'cm',
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.weight': 'bold',
            'mathtext.bf': 'bold',
        }
    
    # Handle custom font registration
    custom_font_prop = None
    if custom_font_path:
        if os.path.exists(custom_font_path):
            try:
                # Register the custom font with matplotlib
                custom_font_prop = fm.FontProperties(fname=custom_font_path)
                if progress_bar:
                    print(f"Successfully loaded custom font: {custom_font_path}")
            except Exception as e:
                if progress_bar:
                    print(f"Warning: Could not load custom font {custom_font_path}: {str(e)}")
                    print("Falling back to default font configuration")
                custom_font_prop = None
        else:
            if progress_bar:
                print(f"Warning: Custom font file not found at {custom_font_path}")
                print("Falling back to default font configuration")
    
    # Apply matplotlib configuration (only if not using custom font)
    if not custom_font_prop:
        matplotlib.rcParams.update(matplotlib_config)
    
    # Load dataframe
    df = pd.read_csv(csv_path)
    
    # Create output directories
    source_dir = Path(output_dir) / 'source'
    os.makedirs(source_dir, exist_ok=True)
    
    def render_text_image(word, output_path):
        """Render a single word as an image with perfect centering"""
        # Create figure with exact pixel dimensions
        dpi = 150
        figsize = (image_width/dpi, image_height/dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white')
        
        # Create axis that fills the figure completely
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Adjust fontsize based on word length
        length_adjustment = max(1, min(15, len(word))) / 7.0
        fontsize = int(base_fontsize / length_adjustment)
        
        # Prepare text formatting arguments
        text_kwargs = {
            'fontsize': fontsize,
            'ha': 'center',
            'va': 'center',
            'transform': ax.transAxes
        }
        
        # Use custom font if available, otherwise use matplotlib config
        if custom_font_prop:
            text_kwargs['fontproperties'] = custom_font_prop
            display_text = word  # Use plain text with custom font
        else:
            # Determine if we should use mathtext formatting
            use_mathtext = matplotlib_config.get('mathtext.bf') == 'bold' or \
                          matplotlib_config.get('font.weight') == 'bold'
            
            # Format text based on configuration
            if use_mathtext and 'mathtext' in str(matplotlib_config):
                display_text = f"$\\mathbf{{{word}}}$"
            else:
                display_text = word
                text_kwargs['fontweight'] = 'bold'
        
        # Render text at exact center with precise positioning
        text_obj = ax.text(0.5, 0.5, display_text, **text_kwargs)
        
        # Remove all axes and ensure clean background
        ax.axis('off')
        ax.set_facecolor('white')
        
        # Save with tight layout to maintain centering
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', 
                    pad_inches=0, facecolor='white', edgecolor='none',
                    dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        
        # Load and process the image
        rendered_img = Image.open(buf)
        rendered_img = rendered_img.convert('RGB')
        
        # Create final canvas with exact target dimensions
        final_img = Image.new('RGB', (image_width, image_height), (255, 255, 255))
        
        # Calculate scaling to fit the rendered text optimally while maintaining aspect ratio
        rendered_width, rendered_height = rendered_img.size
        
        # Scale to use 90% of the canvas for better visual appearance
        scale_w = (image_width * 0.9) / rendered_width
        scale_h = (image_height * 0.9) / rendered_height
        scale = min(scale_w, scale_h)
        
        new_width = int(rendered_width * scale)
        new_height = int(rendered_height * scale)
        
        # Resize with high quality resampling
        resample_method = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS
        resized_img = rendered_img.resize((new_width, new_height), resample_method)
        
        # Calculate exact center position
        x_offset = (image_width - new_width) // 2
        y_offset = (image_height - new_height) // 2
        
        # Paste the resized image at the center
        final_img.paste(resized_img, (x_offset, y_offset))
        
        # Save the final perfectly centered image
        final_img.save(output_path, quality=95, optimize=True)
        return str(output_path)
    
    # Process all words
    source_paths = []
    iterator = tqdm(df.iterrows(), total=len(df), desc="Rendering text images") if progress_bar else df.iterrows()
    
    for index, row in iterator:
        word = row['label']
        
        # Determine output filename
        if 'Image' in df.columns:
            output_filename = row['Image']
        else:
            # Create safe filename from word
            safe_filename = "".join([c if c.isalnum() or c in "_-" else "_" for c in word])
            output_filename = f"{safe_filename}.png"
        
        output_path = source_dir / output_filename
        
        try:
            source_path = render_text_image(word, output_path)
            source_paths.append(source_path)
        except Exception as e:
            if progress_bar:
                print(f"Error processing {word}: {str(e)}")
            # Create fallback empty image
            empty_img = Image.new('RGB', (image_width, image_height), (255, 255, 255))
            empty_img.save(output_path)
            source_paths.append(str(output_path))
    
    # Update dataframe
    df['source_path'] = source_paths
    
    # Handle target paths
    if 'Image' in df.columns:
        df['target_path'] = target_dir_prefix + df['Image']
        df = df.drop('Image', axis=1)
    else:
        # Create target paths based on source paths
        df['target_path'] = df['source_path'].apply(
            lambda x: target_dir_prefix + Path(x).name
        )
    
    # Clean up paths and reorder columns
    df['source_path'] = df['source_path'].apply(lambda x: x.replace("\\", "/"))
    df = df[["source_path", "target_path","book","label"]]
    
    # Save final CSV
    output_csv_path = Path(output_dir) / 'data.csv'
    df.to_csv(output_csv_path, index=False)
    
    if progress_bar:
        print(f"Dataset generated successfully!")
    
    return df


#################################################################################################################################
                                        # Creating Grid
#################################################################################################################################

def create_image_grids(df, output_directory, num_grids=1000, grid_size=(3,1), 
                      target_image_size=(384,128), random_seed=42):

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create output directories
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    source_dir = output_path / "source"
    target_dir = output_path / "target"
    
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate grid dimensions
    rows, cols = grid_size
    images_per_grid = rows * cols
    
    # Storage for grid information
    grid_info = []
    
    print(f"Creating {num_grids} image grids...")
    print(f"Grid size: {rows}x{cols} = {images_per_grid} images per grid")
    print(f"Target image size: {target_image_size}")
    
    for grid_idx in tqdm(range(num_grids), desc="Creating grids"):
        try:
            # Randomly sample images for this grid
            sampled_images = df.sample(n=images_per_grid, replace=False)
            
            # Create source and target grids
            source_grid = create_single_grid(sampled_images, 'source_path', 
                                           grid_size, target_image_size)
            target_grid = create_single_grid(sampled_images, 'target_path', 
                                           grid_size, target_image_size)
            
            # Save grids
            source_filename = f"grid_{grid_idx:05d}.png"
            target_filename = f"grid_{grid_idx:05d}.png"
            
            source_path = source_dir / source_filename
            target_path = target_dir / target_filename
            
            source_grid.save(source_path)
            target_grid.save(target_path)
            
            # Store grid information
            grid_info.append({
                'grid_id': grid_idx,
                'source_path': str(source_path),
                'target_path': str(target_path),
                'books_in_grid': list(sampled_images['book'].unique()),
                'labels_in_grid': list(sampled_images['label'].values),
                'num_unique_books': sampled_images['book'].nunique(),
                'num_unique_labels': sampled_images['label'].nunique()
            })
                
        except Exception as e:
            print(f"Error creating grid {grid_idx}: {e}")
            continue
    
    # Create DataFrame with grid information
    grid_df = pd.DataFrame(grid_info)
    
    # Save grid information to CSV
    grid_df.to_csv(output_path / "grid_info.csv", index=False)
    
    print(f"\nSuccessfully created {len(grid_df)} image grids!")
    
    return grid_df

def create_single_grid(sampled_images, image_path_column, grid_size, target_image_size):
    """
    Create a single grid from sampled images
    
    Parameters:
    sampled_images: DataFrame with sampled images
    image_path_column: str, column name for image paths ('source_path' or 'target_path')
    grid_size: tuple, (rows, cols)
    target_image_size: tuple, (width, height)
    
    Returns:
    PIL Image object of the stacked grid
    """
    rows, cols = grid_size
    img_width, img_height = target_image_size
    
    # Create blank canvas for the grid
    grid_width = cols * img_width
    grid_height = rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Place images in grid
    for idx, (_, row) in enumerate(sampled_images.iterrows()):
        try:
            # Load and resize image
            img = Image.open(row[image_path_column])
            img = img.convert('RGB')  # Ensure RGB format
            img = img.resize(target_image_size, Image.Resampling.LANCZOS)
            
            # Calculate position in grid
            grid_row = idx // cols
            grid_col = idx % cols
            
            x = grid_col * img_width
            y = grid_row * img_height
            
            # Paste image into grid
            grid_image.paste(img, (x, y))
            
        except Exception as e:
            print(f"Error loading image {row[image_path_column]}: {e}")
            # Create a placeholder image in case of error
            placeholder = Image.new('RGB', target_image_size, color='gray')
            grid_image.paste(placeholder, (x, y))
    
    return grid_image