from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import logging
import requests
import json
import time
from werkzeug.utils import secure_filename
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
from flask_cors import CORS
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import fitz  # PyMuPDF
from scipy.spatial.distance import cdist
from typing import Tuple, List, Dict
# Gemini API imports and configuration
try:
    import google.generativeai as genai
    from typing import List, Dict, Tuple, Optional
    GEMINI_AVAILABLE = True
    print("Gemini API integration available")
except ImportError:
    GEMINI_AVAILABLE = False
    print("Gemini API not available - install google-generativeai")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
LLAMA_MODEL_PATH = os.getenv('LLAMA_MODEL_PATH')
FALLBACK_CHUNK_SIZE = int(os.getenv('FALLBACK_CHUNK_SIZE', '8000'))
LLM_TIMEOUT_SECONDS = int(os.getenv('LLM_TIMEOUT_SECONDS', '15'))


local_llama_client = None

# Guard imports for optional dependencies
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.info("llama-cpp-python not available")



if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API configured successfully")
    except Exception as e:
        logger.error(f"Error configuring Gemini API: {e}")
        GEMINI_AVAILABLE = False
else:
    GEMINI_AVAILABLE = False
    logger.info("Gemini API not available - missing key or library")

def chunk_text(text: str, chunk_size_chars: int = None) -> List[str]:
    """Split text into chunks if it exceeds threshold"""
    if chunk_size_chars is None:
        chunk_size_chars = FALLBACK_CHUNK_SIZE
    
    if len(text) <= chunk_size_chars:
        return [text]
    
    chunks = []
    for i in range(0, len(text), chunk_size_chars):
        chunks.append(text[i:i + chunk_size_chars])
    
    return chunks

def create_correction_prompt(text: str, context: str = "") -> str:
    """Create structured prompt for text correction with context - centralized for all backends"""
    context_section = ""
    if context:
        context_section = f"""
Previous context:
{context}

"""
    
    return f"""
Correct the following Spanish OCR text while preserving original grammar and style.
Only fix orthographic errors, punctuation, and obvious OCR mistakes.
{context_section}
Text to correct:
{text}

Instructions:
- Fix spelling errors and OCR artifacts
- Preserve historical language patterns  
- Maintain original formatting
- Return ONLY the corrected text

Corrected text:
"""

def try_gemini_correction(text: str, context: str = "", retries: int = 2) -> tuple[str, str]:
    """Try Gemini API for text correction"""
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return text, "gemini_unavailable"
    
    prompt = create_correction_prompt(text, context)
    
    for attempt in range(retries):
        try:
            response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
            if response.candidates and response.text:
                return response.text.strip(), "gemini"
        except Exception as e:
            logger.warning(f"Gemini attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    
    return text, "gemini_failed"

def try_huggingface_correction(text: str, context: str = "", retries: int = 2) -> tuple[str, str]:
    """Try Hugging Face Inference API for text correction"""
    if not HF_API_TOKEN:
        return text, "hf_unavailable"
    
    prompt = create_correction_prompt(text, context)
    
    # Use a general text generation model
    url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    for attempt in range(retries):
        try:
            response = requests.post(
                url, 
                headers=headers,
                json={"inputs": prompt, "parameters": {"max_new_tokens": 512}},
                timeout=LLM_TIMEOUT_SECONDS
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '').strip()
                    if generated_text and generated_text != prompt:
                        return generated_text.replace(prompt, '').strip(), "hf"
            
        except Exception as e:
            logger.warning(f"HuggingFace attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    
    return text, "hf_failed"

def init_local_llama():
    """Initialize local LLaMA client lazily"""
    global local_llama_client
    
    if local_llama_client is not None:
        return True
    
    if not LLAMA_CPP_AVAILABLE or not LLAMA_MODEL_PATH:
        return False
    
    if not os.path.exists(LLAMA_MODEL_PATH):
        logger.warning(f"LLaMA model path not found: {LLAMA_MODEL_PATH}")
        return False
    
    try:
        # Ensure you accepted Llama 3.1 Community License before downloading weights. 
        # See: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
        local_llama_client = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )
        logger.info("Local LLaMA model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize local LLaMA: {e}")
        return False

def try_local_llama_correction(text: str, context: str = "") -> tuple[str, str]:
    """Try local LLaMA for text correction"""
    if not init_local_llama():
        return text, "local_llama_unavailable"
    
    prompt = create_correction_prompt(text, context)
    
    try:
        response = local_llama_client(
            prompt,
            max_tokens=512,
            temperature=0.3,
            stop=["Human:", "\n\n"],
            echo=False
        )
        
        generated_text = response['choices'][0]['text'].strip()
        if generated_text:
            return generated_text, "local_llama"
            
    except Exception as e:
        logger.error(f"Local LLaMA correction failed: {e}")
    
    return text, "local_llama_failed"

def process_text_with_fallbacks(text: str, context: str = "", max_tokens: int = 1024, retries: int = 2) -> tuple[str, str]:
    """
    Main fallback wrapper function that tries all LLM backends in order:
    1. Gemini (primary)
    2. HuggingFace (secondary)  
    3. Local LLaMA (final fallback)
    """
    
    # Try Gemini first
    logger.info("Attempting Gemini correction...")
    result, status = try_gemini_correction(text, context, retries)
    if status == "gemini":
        logger.info("Used: gemini")
        return result, status
    
    # Fallback to HuggingFace
    logger.info("Fallback: attempting HuggingFace correction...")
    result, status = try_huggingface_correction(text, context, retries)
    if status == "hf":
        logger.info("Fallback: hf") 
        return result, status
    
    # Final fallback to local LLaMA
    logger.info("Fallback: attempting local LLaMA correction...")
    result, status = try_local_llama_correction(text, context)
    if status == "local_llama":
        logger.info("Fallback: local_llama")
        return result, status
    
    # All backends failed
    logger.warning("All LLM backends failed, returning original text")
    return text, "all_failed"

def process_text_with_chunking(text: str, context: str = "", max_tokens: int = 1024, retries: int = 2) -> tuple[str, str]:
    """Process text with chunking support for large inputs"""
    if len(text) <= FALLBACK_CHUNK_SIZE:
        return process_text_with_fallbacks(text, context, max_tokens, retries)
    
    # Chunk the text
    chunks = chunk_text(text, FALLBACK_CHUNK_SIZE)
    logger.info(f"Processing {len(chunks)} chunks...")
    
    corrected_chunks = []
    current_context = context
    final_status = "all_failed"
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        corrected_chunk, status = process_text_with_fallbacks(chunk, current_context, max_tokens, retries)
        corrected_chunks.append(corrected_chunk)
        
        # Update status to first successful one
        if final_status == "all_failed" and status != "all_failed":
            final_status = status
        
        # Update context for next chunk (last 2 lines)
        chunk_lines = corrected_chunk.strip().split('\n')
        if len(chunk_lines) >= 2:
            current_context = '\n'.join(chunk_lines[-2:])
        elif len(chunk_lines) == 1:
            current_context = chunk_lines[0]
    
    return '\n'.join(corrected_chunks), final_status

def manual_fallback_test():
    """Test function to verify fallback system"""
    test_text = "Prueba: texto con errores OCR para verificar el sistema de fallback..."
    logger.info("=== Manual Fallback Test ===")
    logger.info(f"Input: {test_text}")
    
    result, status = process_text_with_fallbacks(test_text)
    
    logger.info(f"Output: {result}")
    logger.info(f"Backend used: {status}")
    logger.info("=== Test Complete ===")
    
    return result, status


app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ANNOTATION_FOLDER = 'annotations'
INFERENCE_FOLDER = 'inferences'
LINE_SEGMENTS_FOLDER = 'line_segments'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}
IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Create directories
for folder in [UPLOAD_FOLDER, ANNOTATION_FOLDER, INFERENCE_FOLDER, LINE_SEGMENTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for models (loaded once)
processor = None
model = None
textline_predictor = None
current_image_index = 0

class TextlineExtractor:
    """Advanced textline extraction class from pp.py"""
    def __init__(self, model_path):
        self.cfg = self.setup_cfg(model_path)
        self.predictor = DefaultPredictor(self.cfg)
        
        # Initialize TrOCR
        print("Loading TrOCR model in TextlineExtractor...")
        try:
            self.trocr_processor = TrOCRProcessor.from_pretrained('qantev/trocr-large-spanish')
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained('qantev/trocr-large-spanish')
        except Exception as e:
            print(f"Spanish model failed, using fallback: {e}")
            self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trocr_model.to(self.device)
        print(f"TrOCR model loaded on {self.device}")
        
    def setup_cfg(self, model_path):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # textline, baseline
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_path
        cfg.DATASETS.TEST = ("page_test",)
        cfg.DATALOADER.NUM_WORKERS = 2
        MetadataCatalog.get("page_test").thing_classes = ["textline", "baseline"]
        return cfg
    
    def calculate_dynamic_padding(self, boxes, image_shape):
        """Calculate dynamic padding based on average distances between textboxes"""
        if len(boxes) < 2:
            return {"top": 10, "bottom": 10, "left": 8, "right": 8}
        
        # Calculate centers of bounding boxes
        centers = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append([center_x, center_y])
        
        centers = np.array(centers)
        
        # Calculate vertical and horizontal distances
        vertical_distances = []
        horizontal_distances = []
        
        # Sort boxes by y-coordinate to find vertical neighbors
        sorted_indices = np.argsort(centers[:, 1])
        sorted_boxes = boxes[sorted_indices]
        
        # Calculate vertical gaps between consecutive textlines
        for i in range(len(sorted_boxes) - 1):
            current_box = sorted_boxes[i]
            next_box = sorted_boxes[i + 1]
            
            # Check if boxes are roughly horizontally aligned
            current_center_x = (current_box[0] + current_box[2]) / 2
            next_center_x = (next_box[0] + next_box[2]) / 2
            
            if abs(current_center_x - next_center_x) < image_shape[1] * 0.3:
                gap = next_box[1] - current_box[3]
                if gap > 0:
                    vertical_distances.append(gap)
        
        # Calculate average distances
        avg_vertical_gap = np.median(vertical_distances) if vertical_distances else 20
        avg_horizontal_gap = 15  # Default for horizontal
        
        # Calculate dynamic padding
        vertical_padding = max(5, min(25, avg_vertical_gap / 2))
        horizontal_padding = max(3, min(20, avg_horizontal_gap / 3))
        
        # Consider box heights
        box_heights = [box[3] - box[1] for box in boxes]
        avg_height = np.mean(box_heights)
        height_factor = max(0.1, min(0.3, avg_height / 100))
        vertical_padding = max(vertical_padding, avg_height * height_factor)
        
        padding = {
            "top": int(vertical_padding * 0.95),
            "bottom": int(vertical_padding * 1.2),
            "left": int(horizontal_padding),
            "right": int(horizontal_padding)
        }
        
        return padding

    def filter_margin_boxes_by_area(self, boxes, scores, area_threshold_percent=12.5):
        """Filter out boxes that are significantly smaller than average"""
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Calculate areas
        areas = []
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)
        
        areas = np.array(areas)
        avg_area = np.mean(areas)
        area_threshold = avg_area * (area_threshold_percent / 100.0)
        
        # Filter boxes
        main_boxes = []
        main_scores = []
        margin_boxes = []
        margin_scores = []
        
        for box, score, area in zip(boxes, scores, areas):
            if area >= area_threshold:
                main_boxes.append(box)
                main_scores.append(score)
            else:
                margin_boxes.append(box)
                margin_scores.append(score)
        
        return np.array(main_boxes), np.array(main_scores), np.array(margin_boxes), np.array(margin_scores)
    
    def detect_columns_and_sort_reading_order(self, boxes, scores):
        """Sort textlines in TOP to BOTTOM order (single column)"""
        if len(boxes) == 0:
            return boxes, scores, []
        
        # Calculate centers
        centers = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append([center_x, center_y])
        
        centers = np.array(centers)
        
        # Single column - sort by y-coordinate only
        all_indices = np.arange(len(boxes))
        y_coords = centers[:, 1]
        y_sort_indices = np.argsort(y_coords)
        
        # Create sorted results
        sorted_boxes = []
        sorted_scores = []
        reading_order = []
        
        for position, sort_idx in enumerate(y_sort_indices):
            original_idx = sort_idx
            sorted_boxes.append(boxes[original_idx])
            sorted_scores.append(scores[original_idx])
            
            reading_order.append({
                'original_index': int(original_idx),
                'column': 0,
                'position_in_column': int(position),
                'reading_order_index': int(position)
            })
        
        return np.array(sorted_boxes), np.array(sorted_scores), reading_order
        
    def extract_textlines(self, image):
        """Extract textline predictions from image"""
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        
        # Filter for textline class (assuming class 0 is textline)
        textline_mask = instances.pred_classes == 0
        textline_boxes = instances.pred_boxes[textline_mask].tensor.numpy()
        textline_scores = instances.scores[textline_mask].numpy()
        
        return textline_boxes, textline_scores, outputs
        
    def crop_textlines_with_dynamic_padding(self, image, boxes, use_margin_filtering=True):
        """Crop textline regions from image with dynamic padding"""
        if len(boxes) == 0:
            return [], [], {}
        
        # Filter by area if needed
        if use_margin_filtering:
            main_boxes, main_scores, margin_boxes, margin_scores = self.filter_margin_boxes_by_area(
                boxes, np.ones(len(boxes))
            )
            
            if len(main_boxes) > 0:
                padding = self.calculate_dynamic_padding(main_boxes, image.shape)
                boxes_for_cropping = main_boxes
            else:
                padding = self.calculate_dynamic_padding(boxes, image.shape)
                boxes_for_cropping = boxes
        else:
            padding = self.calculate_dynamic_padding(boxes, image.shape)
            boxes_for_cropping = boxes
        
        cropped_textlines = []
        padded_boxes = []
        
        for box in boxes_for_cropping:
            x1, y1, x2, y2 = box.astype(int)
            
            # Apply dynamic padding
            x1_padded = max(0, x1 - padding["left"])
            y1_padded = max(0, y1 - padding["top"])
            x2_padded = min(image.shape[1], x2 + padding["right"])
            y2_padded = min(image.shape[0], y2 + padding["bottom"])
            
            cropped = image[y1_padded:y2_padded, x1_padded:x2_padded]
            cropped_textlines.append(cropped)
            padded_boxes.append([x1_padded, y1_padded, x2_padded, y2_padded])
        
        return cropped_textlines, padded_boxes, padding
    
    def process_textlines_with_trocr(self, cropped_textlines, reading_order_info):
        """Process cropped textlines with TrOCR in sequential reading order"""
        print(f"Processing {len(cropped_textlines)} textlines with TrOCR...")
        
        # Create list of textlines with their reading order
        textlines_with_order = []
        for i, (textline_crop, ro_info) in enumerate(zip(cropped_textlines, reading_order_info)):
            textlines_with_order.append({
                'crop': textline_crop,
                'reading_order_index': ro_info['reading_order_index'],
                'column': ro_info['column'],
                'position_in_column': ro_info['position_in_column'],
                'original_index': ro_info['original_index']
            })
        
        # Sort by reading order index
        textlines_with_order.sort(key=lambda x: x['reading_order_index'])
        
        # Process textlines in sequential order
        ocr_results = []
        
        for idx, textline_data in enumerate(textlines_with_order):
            try:
                # Convert OpenCV image to PIL Image
                crop_bgr = textline_data['crop']
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(crop_rgb)
                
                # Process with TrOCR
                pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                # Generate text
                with torch.no_grad():
                    generated_ids = self.trocr_model.generate(pixel_values, max_new_tokens=128)
                    generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Store result with reading order information
                ocr_result = {
                    'reading_order_index': textline_data['reading_order_index'],
                    'column': textline_data['column'],
                    'position_in_column': textline_data['position_in_column'],
                    'original_index': textline_data['original_index'],
                    'text': generated_text.strip(),
                    'confidence': 1.0
                }
                
                ocr_results.append(ocr_result)
                    
            except Exception as e:
                print(f"Error processing textline {textline_data['reading_order_index']}: {str(e)}")
                ocr_results.append({
                    'reading_order_index': textline_data['reading_order_index'],
                    'column': textline_data['column'],
                    'position_in_column': textline_data['position_in_column'],
                    'original_index': textline_data['original_index'],
                    'text': '',
                    'confidence': 0.0
                })
        
        # Ensure results are sorted by reading order
        ocr_results.sort(key=lambda x: x['reading_order_index'])
        return ocr_results

# Global textline extractor instance
textline_extractor = None

def load_trocr_model():
    """Load TrOCR model and processor"""
    global processor, model
    if processor is None or model is None:
        print("Loading TrOCR model...")
        try:
            processor = TrOCRProcessor.from_pretrained('qantev/trocr-large-spanish')
            model = VisionEncoderDecoderModel.from_pretrained('qantev/trocr-large-spanish')
            print("TrOCR model loaded successfully!")
        except Exception as e:
            print(f"Error loading TrOCR model: {e}")
            try:
                processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
                model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
                print("Fallback TrOCR model loaded successfully!")
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                raise e2
def perform_line_segmentation_ocr_no_gemini(image_path):
    """Enhanced line segmentation and OCR using advanced pipeline WITHOUT Gemini correction"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'success': False, 'error': 'Could not load image'}
        
        # Try advanced pipeline first
        if textline_extractor is not None:
            try:
                print("Using advanced textline extraction pipeline (no Gemini)...")
                
                # Extract textlines
                boxes, scores, outputs = textline_extractor.extract_textlines(image)
                
                if len(boxes) == 0:
                    return {'success': False, 'error': 'No textlines detected'}
                
                # Filter by area and sort in reading order
                filtered_boxes, filtered_scores, margin_boxes, margin_scores = textline_extractor.filter_margin_boxes_by_area(boxes, scores)
                
                if len(filtered_boxes) == 0:
                    return {'success': False, 'error': 'No textlines after filtering'}
                
                # Sort in reading order
                ordered_boxes, ordered_scores, reading_order_info = textline_extractor.detect_columns_and_sort_reading_order(
                    filtered_boxes, filtered_scores
                )
                
                # Crop textlines with dynamic padding
                cropped_textlines, padded_boxes, padding_info = textline_extractor.crop_textlines_with_dynamic_padding(
                    image, ordered_boxes, use_margin_filtering=False
                )
                
                # Process with TrOCR in sequential order
                ocr_results = textline_extractor.process_textlines_with_trocr(cropped_textlines, reading_order_info)
                
                # Save line segment images and prepare results
                line_segments = []
                for i, (crop, ocr_result, bbox, padded_bbox, score) in enumerate(zip(
                    cropped_textlines, ocr_results, ordered_boxes, padded_boxes, ordered_scores
                )):
                    # Save crop image
                    crop_filename = f"line_{i:03d}_reading_order_{ocr_result['reading_order_index']:03d}.png"
                    crop_path = os.path.join(LINE_SEGMENTS_FOLDER, crop_filename)
                    cv2.imwrite(crop_path, crop)
                    
                    # Convert NumPy values to native Python types
                    bbox_list = bbox.tolist() if hasattr(bbox, 'tolist') else [float(x) for x in bbox]
                    padded_bbox_list = [float(x) for x in padded_bbox]
                    
                    line_segments.append({
                        'line_index': int(i),
                        'reading_order_index': int(ocr_result['reading_order_index']),
                        'column': int(ocr_result['column']),
                        'position_in_column': int(ocr_result['position_in_column']),
                        'image_filename': crop_filename,
                        'bbox': bbox_list,
                        'padded_bbox': padded_bbox_list,
                        'score': float(score),
                        'ocr_text': str(ocr_result['text']),
                        'confidence': float(ocr_result['confidence']),
                        # NOTE: No ocr_text_corrected field - will be added later by Gemini
                    })
                
                # Sort line segments by reading order for final output
                line_segments.sort(key=lambda x: x['reading_order_index'])
                
                # Create full raw text (no Gemini correction)
                raw_text_lines = []
                for segment in line_segments:
                    raw_text = segment.get('ocr_text', '')
                    if raw_text.strip():
                        raw_text_lines.append(raw_text)
                
                full_raw_text = "\n".join(raw_text_lines)
                
                return {
                    'success': True,
                    'line_segments': line_segments,
                    'total_lines': len(line_segments),
                    'pipeline': 'advanced_no_gemini',
                    'padding_info': padding_info,
                    'full_raw_text': full_raw_text,
                    'gemini_processing': False
                }
                
            except Exception as e:
                print(f"Advanced pipeline failed: {e}")
                # Fall back to original method
        
        # Fallback to original method (no Gemini)
        print("Using fallback textline extraction (no Gemini)...")
        boxes, scores, image = extract_textlines_from_image(image_path)
        
        if len(boxes) == 0:
            return {'success': False, 'error': 'No textlines detected'}
        
        # Use original cropping method
        cropped_textlines, padded_boxes = crop_textlines_with_padding(image, boxes)
        
        # Use original OCR method
        load_trocr_model()
        ocr_results = []
        
        for idx, textline_crop in enumerate(cropped_textlines):
            try:
                crop_rgb = cv2.cvtColor(textline_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(crop_rgb)
                pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
                
                with torch.no_grad():
                    generated_ids = model.generate(pixel_values, max_new_tokens=128)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                ocr_results.append({
                    'line_index': idx,
                    'text': generated_text.strip(),
                    'confidence': 1.0
                })
                
            except Exception as e:
                print(f"Error processing textline {idx}: {str(e)}")
                ocr_results.append({
                    'line_index': idx,
                    'text': '',
                    'confidence': 0.0
                })
        
        # Save line segment images
        line_segments = []
        for i, (crop, ocr_result) in enumerate(zip(cropped_textlines, ocr_results)):
            crop_filename = f"line_{i:03d}.png"
            crop_path = os.path.join(LINE_SEGMENTS_FOLDER, crop_filename)
            cv2.imwrite(crop_path, crop)
            
            # Convert NumPy values to native Python types
            bbox_list = boxes[i].tolist() if hasattr(boxes[i], 'tolist') else [float(x) for x in boxes[i]]
            padded_bbox_list = [float(x) for x in padded_boxes[i]]
            
            line_segments.append({
                'line_index': int(i),
                'reading_order_index': int(i),  # Sequential for fallback
                'column': 0,
                'position_in_column': int(i),
                'image_filename': crop_filename,
                'bbox': bbox_list,
                'padded_bbox': padded_bbox_list,
                'score': float(scores[i]),
                'ocr_text': str(ocr_result['text']),
                'confidence': float(ocr_result['confidence'])
                # NOTE: No ocr_text_corrected field - will be added later by Gemini
            })
        
        # Create full raw text (no Gemini correction)
        raw_text_lines = []
        for segment in line_segments:
            raw_text = segment.get('ocr_text', '')
            if raw_text.strip():
                raw_text_lines.append(raw_text)
        
        full_raw_text = "\n".join(raw_text_lines)
        
        return {
            'success': True,
            'line_segments': line_segments,
            'total_lines': len(line_segments),
            'pipeline': 'fallback_no_gemini',
            'full_raw_text': full_raw_text,
            'gemini_processing': False
        }
        
    except Exception as e:
        print(f"Line segmentation OCR error (no Gemini): {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
def load_textline_model():
    """Load advanced Detectron2 textline detection model"""
    global textline_extractor
    if textline_extractor is None:
        print("Loading advanced textline detection model...")
        try:
            # Use your model path here
            model_path = r"C:\Users\prana\Downloads\model_final (8) (1).pth"
            if not os.path.exists(model_path):
                print(f"Model path {model_path} not found, using mock predictor")
                textline_extractor = None
                return
            
            textline_extractor = TextlineExtractor(model_path)
            print("Advanced textline detection model loaded successfully!")
        except Exception as e:
            print(f"Error loading advanced textline model: {e}")
            textline_extractor = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS

def convert_pdf_to_images(pdf_path: str, output_base_name: str, dpi: int = 200) -> List[Dict]:
    """Render a PDF into page images saved in UPLOAD_FOLDER.

    Returns a list of dicts with keys: page_index, image_filename, image_path
    """
    page_infos: List[Dict] = []
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        for i in range(total_pages):
            page = doc.load_page(i)
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            page_filename = f"{output_base_name}_page_{i+1:03d}.png"
            page_path = os.path.join(UPLOAD_FOLDER, page_filename)
            cv2.imwrite(page_path, img)
            page_infos.append({
                'page_index': i,
                'image_filename': page_filename,
                'image_path': page_path
            })
        doc.close()
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        raise
    return page_infos

def process_pdf_upload(pdf_path: str, pdf_filename: str) -> Dict:
    """Process a PDF by rendering each page to an image and running the pipeline per page."""
    try:
        base_name, _ = os.path.splitext(pdf_filename)
        page_infos = convert_pdf_to_images(pdf_path, base_name, dpi=200)
        total_pages = len(page_infos)
        page_filenames: List[str] = []

        for info in page_infos:
            page_filename = info['image_filename']
            page_path = info['image_path']
            page_index = info['page_index']

            # Run line segmentation OCR pipeline per page image
            line_ocr_result = perform_line_segmentation_ocr(page_path)
            # Ensure line segment crop filenames are unique per page
            try:
                page_stem = os.path.splitext(page_filename)[0]
                if line_ocr_result.get('line_segments'):
                    for seg in line_ocr_result['line_segments']:
                        old_name = seg.get('image_filename')
                        if not old_name:
                            continue
                        old_path = os.path.join(LINE_SEGMENTS_FOLDER, old_name)
                        new_name = f"{page_stem}__{old_name}"
                        new_path = os.path.join(LINE_SEGMENTS_FOLDER, new_name)
                        if os.path.exists(old_path):
                            try:
                                os.replace(old_path, new_path)
                            except Exception:
                                pass
                        seg['image_filename'] = new_name
            except Exception as e:
                print(f"Warning renaming line segment images for {page_filename}: {e}")
            # Also run regular OCR for backward compatibility and full text
            regular_ocr_text = perform_ocr_inference(page_path)
            corrected_text = line_ocr_result.get('full_corrected_text', regular_ocr_text)

            # Save per-page inference data
            inference_data = {
                'image': page_filename,
                'pdf_parent': pdf_filename,
                'page_index': page_index,
                'total_pages': total_pages,
                'original_text': regular_ocr_text,
                'corrected_text': corrected_text,
                'line_segments': line_ocr_result.get('line_segments', []),
                'total_lines': line_ocr_result.get('total_lines', 0),
                'pipeline': line_ocr_result.get('pipeline', 'unknown'),
                'gemini_processing': line_ocr_result.get('gemini_processing', False),
                'timestamp': time.time(),
                'is_pdf_page': True
            }

            inference_path = os.path.join(INFERENCE_FOLDER, f"{page_filename}.json")
            with open(inference_path, 'w', encoding='utf-8') as f:
                json.dump(inference_data, f, ensure_ascii=False, indent=2)

            page_filenames.append(page_filename)

        # Optionally save a manifest for the PDF
        try:
            manifest = {
                'pdf_filename': pdf_filename,
                'total_pages': total_pages,
                'page_filenames': page_filenames,
                'created_at': time.time()
            }
            manifest_path = os.path.join(INFERENCE_FOLDER, f"{base_name}_manifest.json")
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: could not save PDF manifest: {e}")

        return {
            'success': True,
            'is_pdf': True,
            'filename': pdf_filename,
            'page_filenames': page_filenames,
            'first_page': page_filenames[0] if page_filenames else None,
            'total_pages': total_pages
        }
    except Exception as e:
        print(f"PDF processing error: {e}")
        return {'success': False, 'error': str(e)}

# --- Gemini API Text Processing Functions --- #
def get_last_two_lines(text: str) -> str:
    """Extract last two lines from text for context"""
    if not text:
        return ""
    
    lines = text.strip().split('\n')
    if len(lines) >= 2:
        return '\n'.join(lines[-2:])
    elif len(lines) == 1:
        return lines[0]
    return ""

def create_correction_prompt(text: str, context: str = "") -> str:
    """Create structured prompt for text correction with context"""
    context_section = ""
    if context:
        context_section = f"""
    Previous page context (last 2 lines):
    {context}
    
    """
    
    return f"""
    Correct the following historical Spanish OCR text while PRESERVING ORIGINAL GRAMMAR AND STYLE.
    Only fix orthographic errors, punctuation, and obvious OCR mistakes. Maintain original capitalization and formatting.
    {context_section}
    Current page text to correct:
    {text}

    Instructions:
    - Fix spelling errors and OCR artifacts
    - Preserve historical language patterns
    - Maintain original line breaks and formatting
    - Use context from previous page if helpful
    - Return ONLY the corrected text without additional comments

    Corrected text:
    """

def process_text_with_gemini(text: str, context: str = "", max_retries: int = 1) -> Tuple[str, str]:
    """Updated to use new three-tier fallback system"""
    result, status = process_text_with_fallbacks(text, context, retries=max_retries)
    
    # Map new status codes to legacy expected return values
    if status in ["gemini", "hf", "local_llama"]:
        return result, "success"
    else:
        return text, "max_retries_exceeded"
    
def process_line_segments_with_gemini(line_segments: List[Dict], context: str = "") -> List[Dict]:
    """Process line segments with Gemini API for text correction"""
    if not GEMINI_AVAILABLE:
        return line_segments
    
    corrected_segments = []
    previous_context = context
    
    for i, segment in enumerate(line_segments):
        original_text = segment.get('ocr_text', '')
        if not original_text.strip():
            corrected_segments.append(segment)
            continue
        
        # Process with Gemini
        corrected_text, status = process_text_with_gemini(original_text, previous_context)
        
        # Update segment with corrected text
        corrected_segment = segment.copy()
        corrected_segment['ocr_text_corrected'] = corrected_text
        corrected_segment['correction_status'] = status
        corrected_segments.append(corrected_segment)
        
        # Update context for next line (use last 2 lines of corrected text)
        if status == "success":
            previous_context = get_last_two_lines(corrected_text)
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    return corrected_segments

def split_image_into_halves(image_path: str, filename: str) -> Dict:
    """Split image into left and right halves and save them"""
    try:
        print(f"ðŸ”„ Splitting image: {filename}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'success': False, 'error': 'Could not load image for splitting'}
        
        height, width = image.shape[:2]
        mid_point = width // 2
        
        print(f"ðŸ“ Image dimensions: {width}x{height}, splitting at {mid_point}")
        
        # Split image into left and right halves
        left_half = image[:, :mid_point]
        right_half = image[:, mid_point:]
        
        # Save split images with unique names
        name, ext = os.path.splitext(filename)
        timestamp = str(int(time.time()))
        left_filename = f"{name}_left_{timestamp}{ext}"
        right_filename = f"{name}_right_{timestamp}{ext}"
        
        left_path = os.path.join(UPLOAD_FOLDER, left_filename)
        right_path = os.path.join(UPLOAD_FOLDER, right_filename)
        
        # Save the split images
        cv2.imwrite(left_path, left_half)
        cv2.imwrite(right_path, right_half)
        
        print(f"âœ… Split images saved:")
        print(f"   Left: {left_path}")
        print(f"   Right: {right_path}")
        
        return {
            'success': True,
            'left_path': left_path,
            'right_path': right_path,
            'left_filename': left_filename,
            'right_filename': right_filename,
            'original_dimensions': (width, height),
            'split_point': mid_point
        }
        
    except Exception as e:
        print(f"âŒ Error splitting image: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def process_split_image(image_path: str, filename: str) -> Dict:
    """Process image by splitting it into left and right halves"""
    try:
        print(f"ðŸš€ Starting split processing for {filename}")
        
        # Step 1: Split the image
        split_result = split_image_into_halves(image_path, filename)
        if not split_result['success']:
            return {'success': False, 'error': f'Split failed: {split_result["error"]}'}
        
        left_path = split_result['left_path']
        right_path = split_result['right_path']
        left_filename = split_result['left_filename']
        right_filename = split_result['right_filename']
        
        # Step 2: Process left half first
        print("ðŸ“„ Processing LEFT half...")
        left_result = perform_line_segmentation_ocr(left_path)
        
        if not left_result['success']:
            print(f"âŒ Left half processing failed: {left_result.get('error', 'Unknown error')}")
            return {'success': False, 'error': f'Left half failed: {left_result.get("error", "Unknown error")}'}
        
        left_segments = left_result.get('line_segments', [])
        print(f"âœ… Left half processed: {len(left_segments)} lines")
        # Prefix and rename left half line segment images to avoid filename collisions
        try:
            for seg in left_segments:
                img_name = seg.get('image_filename')
                if not img_name:
                    continue
                # Avoid double prefixing
                if not img_name.startswith('left_'):
                    old_path = os.path.join(LINE_SEGMENTS_FOLDER, img_name)
                    new_name = f"left_{img_name}"
                    new_path = os.path.join(LINE_SEGMENTS_FOLDER, new_name)
                    if os.path.exists(old_path):
                        try:
                            os.replace(old_path, new_path)
                        except Exception:
                            pass
                    seg['image_filename'] = new_name
        except Exception as e:
            print(f"Warning: could not rename left half line segment images: {e}")
        
        # Step 3: Process right half
        print("ðŸ“„ Processing RIGHT half...")
        right_result = perform_line_segmentation_ocr(right_path)
        
        if not right_result['success']:
            print(f"âŒ Right half processing failed: {right_result.get('error', 'Unknown error')}")
            return {'success': False, 'error': f'Right half failed: {right_result.get("error", "Unknown error")}'}
        
        right_segments = right_result.get('line_segments', [])
        print(f"âœ… Right half processed: {len(right_segments)} lines")
        # Prefix and rename right half line segment images to avoid filename collisions
        try:
            for seg in right_segments:
                img_name = seg.get('image_filename')
                if not img_name:
                    continue
                # Avoid double prefixing
                if not img_name.startswith('right_'):
                    old_path = os.path.join(LINE_SEGMENTS_FOLDER, img_name)
                    new_name = f"right_{img_name}"
                    new_path = os.path.join(LINE_SEGMENTS_FOLDER, new_name)
                    if os.path.exists(old_path):
                        try:
                            os.replace(old_path, new_path)
                        except Exception:
                            pass
                    seg['image_filename'] = new_name
        except Exception as e:
            print(f"Warning: could not rename right half line segment images: {e}")
        
        # Step 4: Combine results
        print(f"ðŸ”— Combining {len(left_segments)} left + {len(right_segments)} right segments")
        
        # Adjust line indices for right half to continue from left half
        for segment in right_segments:
            segment['line_index'] += len(left_segments)
            segment['reading_order_index'] += len(left_segments)
            segment['position_in_column'] += len(left_segments)
            # image_filename already prefixed and renamed earlier
        
        # Combine segments
        combined_segments = left_segments + right_segments
        
        # Combine text from results
        left_text = left_result.get('full_corrected_text', '')
        right_text = right_result.get('full_corrected_text', '')
        combined_text = left_text + '\n' + right_text if left_text and right_text else left_text + right_text
        
        # Fallback: build from segments if combined_text is empty
        if not combined_text:
            try:
                combined_text_lines = []
                for seg in combined_segments:
                    line_txt = seg.get('ocr_text_corrected', seg.get('ocr_text', ''))
                    if line_txt.strip():
                        combined_text_lines.append(line_txt)
                combined_text = "\n".join(combined_text_lines)
            except Exception as e:
                print(f"âš ï¸ Error building combined text from segments: {e}")
                combined_text = ''
        
        print(f"ðŸ“ Combined text length: {len(combined_text)} characters")
        
        # Step 5: Save inference data
        inference_data = {
            'image': filename,
            'split_processing': True,
            'left_half': left_filename,
            'right_half': right_filename,
            'left_lines': len(left_segments),
            'right_lines': len(right_segments),
            'total_lines': len(combined_segments),
            'line_segments': combined_segments,
            'combined_text': combined_text,
            'left_text': left_text,
            'right_text': right_text,
            'original_text': combined_text,
            'corrected_text': combined_text,
            'pipeline': 'split_processing',
            'gemini_processing': left_result.get('gemini_processing', False) or right_result.get('gemini_processing', False),
            'timestamp': time.time(),
            'split_info': {
                'original_dimensions': split_result['original_dimensions'],
                'split_point': split_result['split_point'],
                'left_filename': left_filename,
                'right_filename': right_filename
            }
        }
        
        inference_path = os.path.join(INFERENCE_FOLDER, f"{filename}.json")
        with open(inference_path, 'w', encoding='utf-8') as f:
            json.dump(inference_data, f, ensure_ascii=False, indent=2)

        # Also save separate inference files for each half to enable per-page viewing
        try:
            left_inference = {
                'image': left_filename,
                'parent_image': filename,
                'split_side': 'left',
                'split_processing': True,
                'original_text': left_text,
                'corrected_text': left_text,
                'line_segments': left_segments,
                'total_lines': len(left_segments),
                'pipeline': left_result.get('pipeline', 'split_processing'),
                'gemini_processing': left_result.get('gemini_processing', False),
                'timestamp': time.time()
            }
            with open(os.path.join(INFERENCE_FOLDER, f"{left_filename}.json"), 'w', encoding='utf-8') as f:
                json.dump(left_inference, f, ensure_ascii=False, indent=2)

            right_inference = {
                'image': right_filename,
                'parent_image': filename,
                'split_side': 'right',
                'split_processing': True,
                'original_text': right_text,
                'corrected_text': right_text,
                'line_segments': right_segments,
                'total_lines': len(right_segments),
                'pipeline': right_result.get('pipeline', 'split_processing'),
                'gemini_processing': right_result.get('gemini_processing', False),
                'timestamp': time.time()
            }
            with open(os.path.join(INFERENCE_FOLDER, f"{right_filename}.json"), 'w', encoding='utf-8') as f:
                json.dump(right_inference, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: could not save separate half inference files: {e}")
        
        print(f"ðŸ’¾ Split processing completed successfully!")
        print(f"   Saved to: {inference_path}")
        print(f"   Total lines: {len(combined_segments)}")
        print(f"   Gemini processing: {inference_data['gemini_processing']}")
        
        return {
            'success': True,
            'left_lines': len(left_segments),
            'right_lines': len(right_segments),
            'combined_segments': combined_segments,
            'combined_text': combined_text,
            'left_segments': left_segments,
            'right_segments': right_segments,
            'left_text': left_text,
            'right_text': right_text,
            'left_image': left_filename,
            'right_image': right_filename,
            'gemini_processing': inference_data['gemini_processing'],
            'split_info': split_result
        }
        
    except Exception as e:
        print(f"âŒ Split image processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def perform_ocr_inference(image_path):
    """Perform TrOCR inference on an image (fallback method)"""
    try:
        load_trocr_model()
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values, max_new_tokens=128)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return f"Error during OCR: {str(e)}"

def extract_textlines_from_image(image_path):
    """Extract textlines from image using advanced pipeline or fallback"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        
        # Try advanced pipeline first
        if textline_extractor is not None:
            try:
                boxes, scores, outputs = textline_extractor.extract_textlines(image)
                return boxes, scores, image
            except Exception as e:
                print(f"Advanced textline detection failed: {e}")
        
        # Fallback to mock data if advanced pipeline fails
        print("Using fallback mock textline detection")
        height, width = image.shape[:2]
        mock_boxes = []
        mock_scores = []
        
        # Create some mock textlines
        line_height = height // 10
        for i in range(5):
            y1 = i * line_height + 50
            y2 = (i + 1) * line_height - 20
            x1 = 50
            x2 = width - 50
            mock_boxes.append([x1, y1, x2, y2])
            mock_scores.append(0.9)
        
        return np.array(mock_boxes), np.array(mock_scores), image
        
    except Exception as e:
        print(f"Textline extraction error: {str(e)}")
        return np.array([]), np.array([]), None

def crop_textlines_with_padding(image, boxes, padding=10):
    """Crop textline regions from image with padding"""
    if len(boxes) == 0:
        return [], []
    
    cropped_textlines = []
    padded_boxes = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Apply padding
        x1_padded = max(0, x1 - padding)
        y1_padded = max(0, y1 - padding)
        x2_padded = min(image.shape[1], x2 + padding)
        y2_padded = min(image.shape[0], y2 + padding)
        
        cropped = image[y1_padded:y2_padded, x1_padded:x2_padded]
        cropped_textlines.append(cropped)
        padded_boxes.append([x1_padded, y1_padded, x2_padded, y2_padded])
    
    return cropped_textlines, padded_boxes

def process_textlines_with_trocr(cropped_textlines):
    """Process cropped textlines with TrOCR"""
    if not cropped_textlines:
        return []
    
    load_trocr_model()
    ocr_results = []
    
    for idx, textline_crop in enumerate(cropped_textlines):
        try:
            # Convert OpenCV image to PIL Image
            crop_rgb = cv2.cvtColor(textline_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(crop_rgb)
            
            # Process with TrOCR
            pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
            
            # Generate text
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=128)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            ocr_results.append({
                'line_index': idx,
                'text': generated_text.strip(),
                'confidence': 1.0
            })
            
        except Exception as e:
            print(f"Error processing textline {idx}: {str(e)}")
            ocr_results.append({
                'line_index': idx,
                'text': '',
                'confidence': 0.0
            })
    
    return ocr_results


def perform_line_segmentation_ocr(image_path, skip_gemini=False):
    """Enhanced line segmentation and OCR using advanced pipeline with optional Gemini correction"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'success': False, 'error': 'Could not load image'}
        
        # Try advanced pipeline first
        if textline_extractor is not None:
            try:
                print("Using advanced textline extraction pipeline...")
                
                # Extract textlines
                boxes, scores, outputs = textline_extractor.extract_textlines(image)
                
                if len(boxes) == 0:
                    return {'success': False, 'error': 'No textlines detected'}
                
                # Filter by area and sort in reading order
                filtered_boxes, filtered_scores, margin_boxes, margin_scores = textline_extractor.filter_margin_boxes_by_area(boxes, scores)
                
                if len(filtered_boxes) == 0:
                    return {'success': False, 'error': 'No textlines after filtering'}
                
                # Sort in reading order
                ordered_boxes, ordered_scores, reading_order_info = textline_extractor.detect_columns_and_sort_reading_order(
                    filtered_boxes, filtered_scores
                )
                
                # Crop textlines with dynamic padding
                cropped_textlines, padded_boxes, padding_info = textline_extractor.crop_textlines_with_dynamic_padding(
                    image, ordered_boxes, use_margin_filtering=False
                )
                
                # Process with TrOCR in sequential order
                ocr_results = textline_extractor.process_textlines_with_trocr(cropped_textlines, reading_order_info)
                
                # Save line segment images and prepare results
                line_segments = []
                for i, (crop, ocr_result, bbox, padded_bbox, score) in enumerate(zip(
                    cropped_textlines, ocr_results, ordered_boxes, padded_boxes, ordered_scores
                )):
                    # Save crop image
                    crop_filename = f"line_{i:03d}_reading_order_{ocr_result['reading_order_index']:03d}.png"
                    crop_path = os.path.join(LINE_SEGMENTS_FOLDER, crop_filename)
                    cv2.imwrite(crop_path, crop)
                    
                    # Convert NumPy values to native Python types
                    bbox_list = bbox.tolist() if hasattr(bbox, 'tolist') else [float(x) for x in bbox]
                    padded_bbox_list = [float(x) for x in padded_bbox]
                    
                    line_segments.append({
                        'line_index': int(i),
                        'reading_order_index': int(ocr_result['reading_order_index']),
                        'column': int(ocr_result['column']),
                        'position_in_column': int(ocr_result['position_in_column']),
                        'image_filename': crop_filename,
                        'bbox': bbox_list,
                        'padded_bbox': padded_bbox_list,
                        'score': float(score),
                        'ocr_text': str(ocr_result['text']),
                        'confidence': float(ocr_result['confidence'])
                    })
                
                # Sort line segments by reading order for final output
                line_segments.sort(key=lambda x: x['reading_order_index'])
                
                # Apply Gemini post-processing if available and not skipped
                if GEMINI_AVAILABLE and not skip_gemini:
                    print("Applying Gemini API text correction...")
                    corrected_segments = process_line_segments_with_gemini(line_segments)
                    
                    # Create full corrected text
                    corrected_text_lines = []
                    for segment in corrected_segments:
                        corrected_text = segment.get('ocr_text_corrected', segment.get('ocr_text', ''))
                        if corrected_text.strip():
                            corrected_text_lines.append(corrected_text)
                    
                    full_corrected_text = "\n".join(corrected_text_lines)
                    
                    return {
                        'success': True,
                        'line_segments': corrected_segments,
                        'total_lines': len(corrected_segments),
                        'pipeline': 'advanced_with_gemini',
                        'padding_info': padding_info,
                        'full_corrected_text': full_corrected_text,
                        'gemini_processing': True
                    }
                else:
                    # Create full raw text (no Gemini correction)
                    raw_text_lines = []
                    for segment in line_segments:
                        raw_text = segment.get('ocr_text', '')
                        if raw_text.strip():
                            raw_text_lines.append(raw_text)
                    
                    full_raw_text = "\n".join(raw_text_lines)
                    
                    return {
                        'success': True,
                        'line_segments': line_segments,
                        'total_lines': len(line_segments),
                        'pipeline': 'advanced_no_gemini' if skip_gemini else 'advanced',
                        'padding_info': padding_info,
                        'full_raw_text': full_raw_text,
                        'gemini_processing': False
                    }
                
            except Exception as e:
                print(f"Advanced pipeline failed: {e}")
                # Fall back to original method
        
        # Fallback to original method
        print("Using fallback textline extraction...")
        boxes, scores, image = extract_textlines_from_image(image_path)
        
        if len(boxes) == 0:
            return {'success': False, 'error': 'No textlines detected'}
        
        # Use original cropping method
        cropped_textlines, padded_boxes = crop_textlines_with_padding(image, boxes)
        
        # Use original OCR method
        load_trocr_model()
        ocr_results = []
        
        for idx, textline_crop in enumerate(cropped_textlines):
            try:
                crop_rgb = cv2.cvtColor(textline_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(crop_rgb)
                pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
                
                with torch.no_grad():
                    generated_ids = model.generate(pixel_values, max_new_tokens=128)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                ocr_results.append({
                    'line_index': idx,
                    'text': generated_text.strip(),
                    'confidence': 1.0
                })
                
            except Exception as e:
                print(f"Error processing textline {idx}: {str(e)}")
                ocr_results.append({
                    'line_index': idx,
                    'text': '',
                    'confidence': 0.0
                })
        
        # Save line segment images
        line_segments = []
        for i, (crop, ocr_result) in enumerate(zip(cropped_textlines, ocr_results)):
            crop_filename = f"line_{i:03d}.png"
            crop_path = os.path.join(LINE_SEGMENTS_FOLDER, crop_filename)
            cv2.imwrite(crop_path, crop)
            
            # Convert NumPy values to native Python types
            bbox_list = boxes[i].tolist() if hasattr(boxes[i], 'tolist') else [float(x) for x in boxes[i]]
            padded_bbox_list = [float(x) for x in padded_boxes[i]]
            
            line_segments.append({
                'line_index': int(i),
                'reading_order_index': int(i),  # Sequential for fallback
                'column': 0,
                'position_in_column': int(i),
                'image_filename': crop_filename,
                'bbox': bbox_list,
                'padded_bbox': padded_bbox_list,
                'score': float(scores[i]),
                'ocr_text': str(ocr_result['text']),
                'confidence': float(ocr_result['confidence'])
            })
        
        # Apply Gemini post-processing if available and not skipped
        if GEMINI_AVAILABLE and not skip_gemini:
            print("Applying Gemini API text correction to fallback results...")
            corrected_segments = process_line_segments_with_gemini(line_segments)
            
            # Create full corrected text
            corrected_text_lines = []
            for segment in corrected_segments:
                corrected_text = segment.get('ocr_text_corrected', segment.get('ocr_text', ''))
                if corrected_text.strip():
                    corrected_text_lines.append(corrected_text)
            
            full_corrected_text = "\n".join(corrected_text_lines)
            
            return {
                'success': True,
                'line_segments': corrected_segments,
                'total_lines': len(corrected_segments),
                'pipeline': 'fallback_with_gemini',
                'full_corrected_text': full_corrected_text,
                'gemini_processing': True
            }
        else:
            # Create full raw text (no Gemini correction)
            raw_text_lines = []
            for segment in line_segments:
                raw_text = segment.get('ocr_text', '')
                if raw_text.strip():
                    raw_text_lines.append(raw_text)
            
            full_raw_text = "\n".join(raw_text_lines)
            
            return {
                'success': True,
                'line_segments': line_segments,
                'total_lines': len(line_segments),
                'pipeline': 'fallback_no_gemini' if skip_gemini else 'fallback',
                'full_raw_text': full_raw_text,
                'gemini_processing': False
            }
        
    except Exception as e:
        print(f"Line segmentation OCR error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
    

@app.route('/')
def home():
    """Serve the main HTML file directly"""
    try:
        # Look for the HTML file in the current directory
        html_files = ['index.html', 'paste.html', 'main.html']
        for html_file in html_files:
            if os.path.exists(html_file):
                return send_file(html_file)
        
        # If no HTML file found, return a basic error page
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>HTML file not found</h1>
            <p>Please ensure your HTML file is in the same directory as app.py</p>
            <p>Expected files: index.html, paste.html, or main.html</p>
        </body>
        </html>
        """, 404
    except Exception as e:
        return f"Error serving HTML: {str(e)}", 500

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload route called")
    print("Files in request:", request.files)
    print("Form data:", request.form)
    global current_image_index
    
    # Check if file part exists
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'success': False, 'error': 'No file part in the request'}), 400

    file = request.files['file']
    split_image = request.form.get('split_image', 'false').lower() == 'true'

    # Validate filename
    if file.filename == '':
        print("No selected file")
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    # Validate extension
    if not allowed_file(file.filename):
        print("Invalid file type:", file.filename)
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400

    try:
        # Save file
        filename = secure_filename(file.filename)
        
        # Add timestamp to filename to avoid conflicts
        name, ext = os.path.splitext(filename)
        timestamp = str(int(time.time()))
        filename = f"{name}_{timestamp}{ext}"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"File saved to: {file_path}")

        # Special handling for PDF uploads: render each page and process per-page
        if ext.lower() == '.pdf':
            print("Processing uploaded PDF...")
            pdf_result = process_pdf_upload(file_path, filename)
            if pdf_result.get('success'):
                # Set current image index to the first rendered page if available
                images = [f for f in os.listdir(UPLOAD_FOLDER) if is_image_file(f)]
                first_page = pdf_result.get('first_page')
                if first_page and first_page in images:
                    current_image_index = images.index(first_page)
                return jsonify(pdf_result)
            else:
                return jsonify({'success': False, 'error': pdf_result.get('error', 'PDF processing failed')}), 500

        if split_image:
            # Process image in two halves
            print("Processing image in two halves...")
            split_result = process_split_image(file_path, filename)
            
            if split_result['success']:
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'split_processing': True,
                    'left_lines': split_result['left_lines'],
                    'right_lines': split_result['right_lines'],
                    'total_lines': split_result['left_lines'] + split_result['right_lines'],
                    'line_segments': split_result['combined_segments'],
                    'corrected_text': split_result['combined_text'],
                    'pipeline': 'split_processing',
                    'gemini_processing': split_result.get('gemini_processing', False)
                })
            else:
                # Fallback to regular processing
                print("Split processing failed, falling back to regular processing")
                split_image = False

        if not split_image:
            # Perform regular line segmentation and OCR
            print("Starting regular line segmentation and OCR...")
            line_ocr_result = perform_line_segmentation_ocr(file_path)
            
            if line_ocr_result['success']:
                # Also perform regular OCR for backward compatibility
                regular_ocr_text = perform_ocr_inference(file_path)
                
                # Use Gemini-corrected text if available, otherwise use regular OCR text
                corrected_text = line_ocr_result.get('full_corrected_text', regular_ocr_text)
                
                # Save inference data with line segments
                inference_data = {
                    'image': filename,
                    'original_text': regular_ocr_text,
                    'corrected_text': corrected_text,
                    'line_segments': line_ocr_result['line_segments'],
                    'total_lines': line_ocr_result['total_lines'],
                    'pipeline': line_ocr_result.get('pipeline', 'unknown'),
                    'gemini_processing': line_ocr_result.get('gemini_processing', False),
                    'timestamp': time.time()
                }
            else:
                # Fallback to regular OCR
                regular_ocr_text = perform_ocr_inference(file_path)
                inference_data = {
                    'image': filename,
                    'original_text': regular_ocr_text,
                    'corrected_text': regular_ocr_text,
                    'line_segments': [],
                    'total_lines': 0,
                    'pipeline': 'fallback',
                    'gemini_processing': False,
                    'timestamp': time.time()
                }

            inference_path = os.path.join(INFERENCE_FOLDER, f"{filename}.json")
            with open(inference_path, 'w', encoding='utf-8') as f:
                json.dump(inference_data, f, ensure_ascii=False, indent=2)

            # Update current image index to point to the new image
            images = [f for f in os.listdir(UPLOAD_FOLDER) if is_image_file(f)]
            if filename in images:
                current_image_index = images.index(filename)

            return jsonify({
                'success': True, 
                'filename': filename, 
                'inference': regular_ocr_text,
                'corrected_text': line_ocr_result.get('full_corrected_text', regular_ocr_text),
                'line_segments': line_ocr_result.get('line_segments', []),
                'total_lines': line_ocr_result.get('total_lines', 0),
                'pipeline': line_ocr_result.get('pipeline', 'unknown'),
                'gemini_processing': line_ocr_result.get('gemini_processing', False)
            })

    except Exception as e:
        print("Upload error:", str(e))
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/image/<filename>')
def get_image(filename):
    """Serve images from upload folder"""
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except Exception as e:
        print(f"Error serving image {filename}: {e}")
        return jsonify({'error': 'Image not found'}), 404

@app.route('/line_segment/<filename>')
def get_line_segment(filename):
    """Serve line segment images"""
    try:
        return send_from_directory(LINE_SEGMENTS_FOLDER, filename)
    except Exception as e:
        print(f"Error serving line segment {filename}: {e}")
        return jsonify({'error': 'Line segment not found'}), 404

@app.route('/get_current_image')
def get_current_image():
    """Get current image name"""
    images = [f for f in os.listdir(UPLOAD_FOLDER) if is_image_file(f)]
    global current_image_index
    
    if images and 0 <= current_image_index < len(images):
        return jsonify({'image_name': images[current_image_index]})
    return jsonify({'image_name': None})

@app.route('/next_image', methods=['POST'])
def next_image():
    """Move to next image"""
    images = [f for f in os.listdir(UPLOAD_FOLDER) if is_image_file(f)]
    global current_image_index
    
    if images:
        current_image_index = (current_image_index + 1) % len(images)
        return jsonify({'image_name': images[current_image_index]})
    else:
        return jsonify({'image_name': None})

@app.route('/get_inference/<filename>')
def get_inference(filename):
    """Get inference data for a specific image"""
    try:
        inference_path = os.path.join(INFERENCE_FOLDER, f"{filename}.json")
        if os.path.exists(inference_path):
            with open(inference_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return jsonify(data)
        else:
            return jsonify({'error': 'No inference found', 'image': filename})
    except Exception as e:
        print(f"Error loading inference for {filename}: {e}")
        return jsonify({'error': f'Error loading inference: {str(e)}'})

@app.route('/update_inference', methods=['POST'])
def update_inference():
    """Update corrected text for an image"""
    try:
        data = request.json
        filename = data.get('image')
        corrected_text = data.get('corrected_text')
        
        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided'})
        
        inference_path = os.path.join(INFERENCE_FOLDER, f"{filename}.json")
        
        if os.path.exists(inference_path):
            with open(inference_path, 'r', encoding='utf-8') as f:
                inference_data = json.load(f)
            
            inference_data['corrected_text'] = corrected_text
            inference_data['last_updated'] = time.time()
            
            with open(inference_path, 'w', encoding='utf-8') as f:
                json.dump(inference_data, f, ensure_ascii=False, indent=2)
            
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Inference file not found'})
    
    except Exception as e:
        print(f"Error updating inference: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/update_line_ocr', methods=['POST'])
def update_line_ocr():
    """Update OCR text for a specific line"""
    try:
        data = request.json
        filename = data.get('image')
        line_index = data.get('line_index')
        corrected_text = data.get('corrected_text')
        
        if not filename or line_index is None:
            return jsonify({'success': False, 'error': 'Missing filename or line_index'})
        
        inference_path = os.path.join(INFERENCE_FOLDER, f"{filename}.json")
        
        if os.path.exists(inference_path):
            with open(inference_path, 'r', encoding='utf-8') as f:
                inference_data = json.load(f)
            
            # Update the specific line's OCR text
            if 'line_segments' in inference_data:
                for segment in inference_data['line_segments']:
                    if segment['line_index'] == line_index:
                        segment['ocr_text'] = corrected_text
                        # Keep a parallel corrected field if present in pipeline
                        segment['ocr_text_corrected'] = corrected_text
                        break
                # Recompute combined corrected text for convenience
                try:
                    joined = []
                    for seg in sorted(inference_data['line_segments'], key=lambda s: s.get('line_index', 0)):
                        joined.append(seg.get('ocr_text_corrected', seg.get('ocr_text', '')))
                    inference_data['corrected_text'] = "\n".join(joined)
                except Exception:
                    pass
            
            inference_data['last_updated'] = time.time()
            
            with open(inference_path, 'w', encoding='utf-8') as f:
                json.dump(inference_data, f, ensure_ascii=False, indent=2)
            
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Inference file not found'})
    
    except Exception as e:
        print(f"Error updating line OCR: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/rerun_inference', methods=['POST'])
def rerun_inference():
    """Re-run OCR inference on an image"""
    try:
        data = request.json
        filename = data.get('image')
        
        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided'})
        
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Image file not found'})
        
        # Check if client requests split processing on rerun
        split_image = bool(data.get('split_image', False))
        
        if split_image:
            print(f"Re-running split processing for {filename}...")
            split_result = process_split_image(image_path, filename)
            if not split_result.get('success'):
                return jsonify({'success': False, 'error': split_result.get('error', 'Split processing failed')})
            # Load the just-saved inference for response consistency
            inference_path = os.path.join(INFERENCE_FOLDER, f"{filename}.json")
            with open(inference_path, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            response_payload = {
                'success': True,
                'inference': saved.get('original_text', saved.get('combined_text', '')),
                'corrected_text': saved.get('corrected_text', saved.get('combined_text', '')),
                'line_segments': saved.get('line_segments', []),
                'total_lines': saved.get('total_lines', 0),
                'pipeline': saved.get('pipeline', 'split_processing'),
                'gemini_processing': saved.get('gemini_processing', False),
                'split_processing': True
            }
            return jsonify(response_payload)
        
        # Perform line segmentation and OCR
        print(f"Re-running line segmentation and OCR for {filename}...")
        line_ocr_result = perform_line_segmentation_ocr(image_path)
        
        # Also perform regular OCR
        regular_ocr_text = perform_ocr_inference(image_path)
        
        # Use Gemini-corrected text if available
        corrected_text = line_ocr_result.get('full_corrected_text', regular_ocr_text)
        
        # Update inference file
        inference_data = {
            'image': filename,
            'original_text': regular_ocr_text,
            'corrected_text': corrected_text,
            'line_segments': line_ocr_result.get('line_segments', []),
            'total_lines': line_ocr_result.get('total_lines', 0),
            'pipeline': line_ocr_result.get('pipeline', 'unknown'),
            'gemini_processing': line_ocr_result.get('gemini_processing', False),
            'timestamp': time.time(),
            'rerun_count': 1
        }
        
        # Check if file exists and increment rerun count
        inference_path = os.path.join(INFERENCE_FOLDER, f"{filename}.json")
        if os.path.exists(inference_path):
            with open(inference_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                inference_data['rerun_count'] = existing_data.get('rerun_count', 0) + 1
        
        with open(inference_path, 'w', encoding='utf-8') as f:
            json.dump(inference_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True, 
            'inference': regular_ocr_text,
            'corrected_text': corrected_text,
            'line_segments': line_ocr_result.get('line_segments', []),
            'total_lines': line_ocr_result.get('total_lines', 0),
            'pipeline': line_ocr_result.get('pipeline', 'unknown'),
            'gemini_processing': line_ocr_result.get('gemini_processing', False),
            'split_processing': False
        })
    
    except Exception as e:
        print(f"Error re-running inference: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/apply_gemini_correction', methods=['POST'])
def apply_gemini_correction():
    """Apply Gemini API correction to existing OCR results"""
    try:
        data = request.json
        filename = data.get('image')
        
        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided'})
        
        if not GEMINI_AVAILABLE:
            return jsonify({'success': False, 'error': 'Gemini API not available'})
        
        # Load existing inference data
        inference_path = os.path.join(INFERENCE_FOLDER, f"{filename}.json")
        if not os.path.exists(inference_path):
            return jsonify({'success': False, 'error': 'No inference data found'})
        
        with open(inference_path, 'r', encoding='utf-8') as f:
            inference_data = json.load(f)
        
        # Get line segments
        line_segments = inference_data.get('line_segments', [])
        if not line_segments:
            return jsonify({'success': False, 'error': 'No line segments found'})
        
        # Apply Gemini correction
        print(f"Applying Gemini correction to {len(line_segments)} line segments...")
        corrected_segments = process_line_segments_with_gemini(line_segments)
        
        # Create full corrected text
        corrected_text_lines = []
        for segment in corrected_segments:
            corrected_text = segment.get('ocr_text_corrected', segment.get('ocr_text', ''))
            if corrected_text.strip():
                corrected_text_lines.append(corrected_text)
        
        full_corrected_text = "\n".join(corrected_text_lines)
        
        # Update inference data
        inference_data['corrected_text'] = full_corrected_text
        inference_data['line_segments'] = corrected_segments
        inference_data['gemini_processing'] = True
        inference_data['gemini_correction_timestamp'] = time.time()
        
        # Save updated data
        with open(inference_path, 'w', encoding='utf-8') as f:
            json.dump(inference_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'corrected_text': full_corrected_text,
            'line_segments': corrected_segments,
            'total_lines': len(corrected_segments),
            'gemini_processing': True
        })
    
    except Exception as e:
        print(f"Error applying Gemini correction: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/save', methods=['POST'])
def save_annotations():
    """Save annotation data"""
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'Invalid data'})
        
        # Add timestamp
        data['saved_at'] = time.time()
        
        annotation_path = os.path.join(ANNOTATION_FOLDER, f"{data['image']}.json")
        with open(annotation_path, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Annotations saved for {data['image']}: {len(data.get('annotations', []))} boxes")
        return jsonify({'success': True})
    
    except Exception as e:
        print(f"Error saving annotations: {e}")
        return jsonify({'success': False, 'error': str(e)})
    

@app.route('/apply_gemini_to_pdf', methods=['POST'])
def apply_gemini_to_pdf():
    """Apply Gemini correction to all pages of an existing PDF"""
    try:
        data = request.json
        pdf_filename = data.get('pdf_filename')
        
        if not pdf_filename:
            return jsonify({'success': False, 'error': 'No PDF filename provided'})
        
        if not GEMINI_AVAILABLE:
            return jsonify({'success': False, 'error': 'Gemini API not available'})
        
        # Load PDF manifest to get page filenames
        base_name, _ = os.path.splitext(pdf_filename)
        manifest_path = os.path.join(INFERENCE_FOLDER, f"{base_name}_manifest.json")
        
        if not os.path.exists(manifest_path):
            return jsonify({'success': False, 'error': 'PDF manifest not found'})
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        page_filenames = manifest.get('page_filenames', [])
        if not page_filenames:
            return jsonify({'success': False, 'error': 'No pages found in manifest'})
        
        # Apply Gemini correction to all pages
        print(f"Applying Gemini correction to {len(page_filenames)} pages of {pdf_filename}...")
        previous_context = ""
        processed_pages = 0
        
        for i, page_filename in enumerate(page_filenames):
            print(f"Processing page {i + 1}/{len(page_filenames)}: {page_filename}")
            
            inference_path = os.path.join(INFERENCE_FOLDER, f"{page_filename}.json")
            
            if not os.path.exists(inference_path):
                print(f"Warning: Inference file not found for {page_filename}")
                continue
            
            # Load existing inference data
            with open(inference_path, 'r', encoding='utf-8') as f:
                inference_data = json.load(f)
            
            original_text = inference_data.get('original_text', '')
            
            if original_text.strip():
                # Apply Gemini correction with context from previous page
                corrected_text, status = process_text_with_gemini(original_text, previous_context)
                
                # Update inference data with Gemini results
                inference_data['corrected_text'] = corrected_text
                inference_data['gemini_processing'] = (status == "success")
                inference_data['gemini_correction_status'] = status
                inference_data['gemini_correction_timestamp'] = time.time()
                inference_data['gemini_reprocessed'] = True
                
                # Update context for next page
                if status == "success":
                    previous_context = get_last_two_lines(corrected_text)
                else:
                    previous_context = get_last_two_lines(original_text)
                
                processed_pages += 1
            else:
                # Empty text - mark as processed
                inference_data['gemini_processing'] = True
                inference_data['gemini_correction_status'] = "empty_text"
                inference_data['gemini_correction_timestamp'] = time.time()
                inference_data['gemini_reprocessed'] = True
            
            # Save updated inference data
            with open(inference_path, 'w', encoding='utf-8') as f:
                json.dump(inference_data, f, ensure_ascii=False, indent=2)
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Update manifest
        manifest['gemini_processing_completed'] = True
        manifest['gemini_reprocessed_at'] = time.time()
        manifest['processed_pages_count'] = processed_pages
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'pdf_filename': pdf_filename,
            'total_pages': len(page_filenames),
            'processed_pages': processed_pages,
            'gemini_processing_completed': True
        })
    
    except Exception as e:
        print(f"Error applying Gemini to PDF: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health_check():
    """Enhanced health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': processor is not None and model is not None,
        'textline_model_loaded': textline_extractor is not None,
        'advanced_pipeline_available': textline_extractor is not None,
        'gemini_api_available': GEMINI_AVAILABLE,
        'upload_folder': UPLOAD_FOLDER,
        'folders_exist': {
            'uploads': os.path.exists(UPLOAD_FOLDER),
            'annotations': os.path.exists(ANNOTATION_FOLDER),
            'inferences': os.path.exists(INFERENCE_FOLDER),
            'line_segments': os.path.exists(LINE_SEGMENTS_FOLDER)
        }
    })
@app.route('/llm_health')
def llm_health():
    """Health check for LLM backends"""
    
    # Check Gemini availability
    gemini_available = GEMINI_AVAILABLE and bool(GEMINI_API_KEY)
    
    # Check HuggingFace availability  
    hf_available = bool(HF_API_TOKEN)
    
    # Check local LLaMA availability
    local_llama_available = (LLAMA_CPP_AVAILABLE and 
                           bool(LLAMA_MODEL_PATH) and 
                           os.path.exists(LLAMA_MODEL_PATH) if LLAMA_MODEL_PATH else False)
    
    # Check if Llama license acceptance is required
    llama_license_required = True  # Always true since it's a gated model
    
    notes = []
    if not gemini_available:
        notes.append("Gemini: Missing GEMINI_API_KEY")
    if not hf_available:
        notes.append("HuggingFace: Missing HF_API_TOKEN") 
    if not local_llama_available:
        if not LLAMA_CPP_AVAILABLE:
            notes.append("LLaMA: llama-cpp-python not installed")
        elif not LLAMA_MODEL_PATH:
            notes.append("LLaMA: Missing LLAMA_MODEL_PATH")
        else:
            notes.append("LLaMA: Model file not found")
    
    return jsonify({
        "gemini": gemini_available,
        "hf": hf_available, 
        "local_llama": local_llama_available,
        "llama_license_required": llama_license_required,
        "llama_license_url": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
        "notes": "; ".join(notes) if notes else "All available backends ready"
    })

if __name__ == '__main__':
    print("Starting Flask app with three-tier LLM fallback system...")
    print(f"Environment variables:")
    print(f"  GEMINI_API_KEY: {'✓' if GEMINI_API_KEY else '✗'}")
    print(f"  HF_API_TOKEN: {'✓' if HF_API_TOKEN else '✗'}")
    print(f"  LLAMA_MODEL_PATH: {'✓' if LLAMA_MODEL_PATH else '✗'}")
    print(f"  FALLBACK_CHUNK_SIZE: {FALLBACK_CHUNK_SIZE}")
    print(f"  LLM_TIMEOUT_SECONDS: {LLM_TIMEOUT_SECONDS}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024)}MB")
    
    # Test the fallback system
    try:
        manual_fallback_test()
    except Exception as e:
        logger.warning(f"Fallback test failed: {e}")
    
    # Pre-load the models on startup
    try:
        load_textline_model()
        load_trocr_model()
    except Exception as e:
        print(f"Warning: Could not pre-load models: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
