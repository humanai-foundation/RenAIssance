# from flask import Flask, request, jsonify, send_from_directory, send_file
# import os
# import json
# import time
# from werkzeug.utils import secure_filename
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image
# import torch
# from flask_cors import CORS
# import cv2
# import numpy as np
# from detectron2.config import get_cfg
# from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog
# from detectron2 import model_zoo
# import fitz  # PyMuPDF
# from scipy.spatial.distance import cdist

# app = Flask(__name__)
# CORS(app)

# # Configuration
# UPLOAD_FOLDER = 'uploads'
# ANNOTATION_FOLDER = 'annotations'
# INFERENCE_FOLDER = 'inferences'
# LINE_SEGMENTS_FOLDER = 'line_segments'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}
# IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# # Create directories
# for folder in [UPLOAD_FOLDER, ANNOTATION_FOLDER, INFERENCE_FOLDER, LINE_SEGMENTS_FOLDER]:
#     os.makedirs(folder, exist_ok=True)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# # Global variables for models (loaded once)
# processor = None
# model = None
# textline_predictor = None
# current_image_index = 0

# def load_trocr_model():
#     """Load TrOCR model and processor"""
#     global processor, model
#     if processor is None or model is None:
#         print("Loading TrOCR model...")
#         try:
#             processor = TrOCRProcessor.from_pretrained('qantev/trocr-large-spanish')
#             model = VisionEncoderDecoderModel.from_pretrained('qantev/trocr-large-spanish')
#             print("TrOCR model loaded successfully!")
#         except Exception as e:
#             print(f"Error loading TrOCR model: {e}")
#             # Fallback to base model if Spanish model fails
#             try:
#                 processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
#                 model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
#                 print("Fallback TrOCR model loaded successfully!")
#             except Exception as e2:
#                 print(f"Error loading fallback model: {e2}")
#                 raise e2

# def load_textline_model():
#     """Load Detectron2 textline detection model"""
#     global textline_predictor
#     if textline_predictor is None:
#         print("Loading textline detection model...")
#         try:
#             # For now, we'll use a basic setup - you'll need to provide the actual model path
#             cfg = get_cfg()
#             cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
#             cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # textline, baseline
#             cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
#             cfg.MODEL.WEIGHTS = r"C:\Users\prana\Downloads\model_final (8) (1).pth"  # You'll need to provide this
#             cfg.DATASETS.TEST = ("page_test",)
#             MetadataCatalog.get("page_test").thing_classes = ["textline", "baseline"]
            
#             # For now, we'll create a mock predictor - replace with actual model loading
#             textline_predictor = "mock_predictor"  # Replace with actual model loading
#             print("Textline detection model loaded successfully!")
#         except Exception as e:
#             print(f"Error loading textline model: {e}")
#             textline_predictor = None

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def is_image_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS

# def is_pdf_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

# def perform_ocr_inference(image_path):
#     """Perform TrOCR inference on an image"""
#     try:
#         # Load model if not already loaded
#         load_trocr_model()
        
#         # Load and process image
#         image = Image.open(image_path).convert("RGB")
#         pixel_values = processor(images=image, return_tensors="pt").pixel_values
        
#         # Generate text
#         generated_ids = model.generate(pixel_values)
#         generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
#         return generated_text
#     except Exception as e:
#         print(f"OCR Error: {str(e)}")
#         return f"Error during OCR: {str(e)}"

# def extract_textlines_from_image(image_path):
#     """Extract textlines from image using Detectron2"""
#     try:
#         # Load image
#         image = cv2.imread(image_path)
#         if image is None:
#             raise ValueError("Could not load image")
        
#         # For now, return mock data since we don't have the actual model
#         # In production, you would use the actual textline_predictor here
#         height, width = image.shape[:2]
        
#         # Mock textline detection - replace with actual model inference
#         # This creates some sample textlines for demonstration
#         mock_boxes = []
#         mock_scores = []
        
#         # Create some mock textlines (replace with actual detection)
#         line_height = height // 10
#         for i in range(5):  # 5 mock textlines
#             y1 = i * line_height + 50
#             y2 = (i + 1) * line_height - 20
#             x1 = 50
#             x2 = width - 50
#             mock_boxes.append([x1, y1, x2, y2])
#             mock_scores.append(0.9)
        
#         return mock_boxes, mock_scores, image
        
#     except Exception as e:
#         print(f"Textline extraction error: {str(e)}")
#         return [], [], None

# def crop_textlines_with_padding(image, boxes, padding=10):
#     """Crop textline regions from image with padding"""
#     if len(boxes) == 0:
#         return [], []
    
#     cropped_textlines = []
#     padded_boxes = []
    
#     for i, box in enumerate(boxes):
#         x1, y1, x2, y2 = [int(coord) for coord in box]
        
#         # Apply padding
#         x1_padded = max(0, x1 - padding)
#         y1_padded = max(0, y1 - padding)
#         x2_padded = min(image.shape[1], x2 + padding)
#         y2_padded = min(image.shape[0], y2 + padding)
        
#         cropped = image[y1_padded:y2_padded, x1_padded:x2_padded]
#         cropped_textlines.append(cropped)
#         padded_boxes.append([x1_padded, y1_padded, x2_padded, y2_padded])
    
#     return cropped_textlines, padded_boxes

# def process_textlines_with_trocr(cropped_textlines):
#     """Process cropped textlines with TrOCR"""
#     if not cropped_textlines:
#         return []
    
#     load_trocr_model()
#     ocr_results = []
    
#     for idx, textline_crop in enumerate(cropped_textlines):
#         try:
#             # Convert OpenCV image to PIL Image
#             crop_rgb = cv2.cvtColor(textline_crop, cv2.COLOR_BGR2RGB)
#             pil_image = Image.fromarray(crop_rgb)
            
#             # Process with TrOCR
#             pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
            
#             # Generate text
#             with torch.no_grad():
#                 generated_ids = model.generate(pixel_values)
#                 generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
#             ocr_results.append({
#                 'line_index': idx,
#                 'text': generated_text.strip(),
#                 'confidence': 1.0
#             })
            
#         except Exception as e:
#             print(f"Error processing textline {idx}: {str(e)}")
#             ocr_results.append({
#                 'line_index': idx,
#                 'text': '',
#                 'confidence': 0.0
#             })
    
#     return ocr_results

# def perform_line_segmentation_ocr(image_path, segment_prefix=None):
#     """Perform line segmentation and OCR on each line"""
#     try:
#         # Extract textlines
#         boxes, scores, image = extract_textlines_from_image(image_path)
        
#         if len(boxes) == 0:
#             return {
#                 'success': False,
#                 'error': 'No textlines detected'
#             }
        
#         # Crop textlines with padding
#         cropped_textlines, padded_boxes = crop_textlines_with_padding(image, boxes)
        
#         # Process with TrOCR
#         ocr_results = process_textlines_with_trocr(cropped_textlines)
        
#         # Save line segment images
#         line_segments = []
#         for i, (crop, ocr_result) in enumerate(zip(cropped_textlines, ocr_results)):
#             # Save crop image
#             if segment_prefix:
#                 crop_filename = f"{segment_prefix}line_{i:03d}.png"
#             else:
#                 crop_filename = f"line_{i:03d}.png"
#             crop_path = os.path.join(LINE_SEGMENTS_FOLDER, crop_filename)
#             cv2.imwrite(crop_path, crop)
            
#             line_segments.append({
#                 'line_index': i,
#                 'image_filename': crop_filename,
#                 'bbox': boxes[i],
#                 'padded_bbox': padded_boxes[i],
#                 'score': scores[i],
#                 'ocr_text': ocr_result['text'],
#                 'confidence': ocr_result['confidence']
#             })
        
#         return {
#             'success': True,
#             'line_segments': line_segments,
#             'total_lines': len(line_segments)
#         }
        
#     except Exception as e:
#         print(f"Line segmentation OCR error: {str(e)}")
#         return {
#             'success': False,
#             'error': str(e)
#         }

# def convert_pdf_to_images(pdf_path, base_name, dpi=200):
#     """Convert each page of a PDF into an image saved in UPLOAD_FOLDER.

#     Returns a list of image filenames (one per page).
#     """
#     page_filenames = []
#     try:
#         doc = fitz.open(pdf_path)
#         for page_index in range(len(doc)):
#             page = doc.load_page(page_index)
#             matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
#             pix = page.get_pixmap(matrix=matrix, alpha=False)
#             page_filename = f"{base_name}_page_{page_index + 1:03d}.png"
#             out_path = os.path.join(UPLOAD_FOLDER, page_filename)
#             pix.save(out_path)
#             page_filenames.append(page_filename)
#         doc.close()
#     except Exception as e:
#         print(f"PDF to images conversion error: {e}")
#         raise
#     return page_filenames

# @app.route('/')
# def home():
#     """Serve the main HTML file directly"""
#     try:
#         # Look for the HTML file in the current directory
#         html_files = ['index.html', 'paste.html', 'main.html']
#         for html_file in html_files:
#             if os.path.exists(html_file):
#                 return send_file(html_file)
        
#         # If no HTML file found, return a basic error page
#         return """
#         <!DOCTYPE html>
#         <html>
#         <head><title>Error</title></head>
#         <body>
#             <h1>HTML file not found</h1>
#             <p>Please ensure your HTML file is in the same directory as app.py</p>
#             <p>Expected files: index.html, paste.html, or main.html</p>
#         </body>
#         </html>
#         """, 404
#     except Exception as e:
#         return f"Error serving HTML: {str(e)}", 500

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     print("Upload route called")
#     print("Files in request:", request.files)
#     print("Form data:", request.form)
    
#     # Check if file part exists
#     if 'file' not in request.files:
#         print("No file part in request")
#         return jsonify({'success': False, 'error': 'No file part in the request'}), 400

#     file = request.files['file']

#     # Validate filename
#     if file.filename == '':
#         print("No selected file")
#         return jsonify({'success': False, 'error': 'No selected file'}), 400

#     # Validate extension
#     if not allowed_file(file.filename):
#         print("Invalid file type:", file.filename)
#         return jsonify({'success': False, 'error': 'Invalid file type'}), 400

#     try:
#         # Save file
#         filename = secure_filename(file.filename)
        
#         # Add timestamp to filename to avoid conflicts
#         name, ext = os.path.splitext(filename)
#         timestamp = str(int(time.time()))
#         filename = f"{name}_{timestamp}{ext}"
        
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)
#         print(f"File saved to: {file_path}")

#         # If PDF, convert to images and process page-wise
#         if is_pdf_file(filename):
#             base = os.path.splitext(filename)[0]
#             print("PDF detected. Converting to images (one per page)...")
#             page_filenames = convert_pdf_to_images(file_path, base)
#             pages_summary = []

#             for page_idx, page_filename in enumerate(page_filenames):
#                 page_image_path = os.path.join(UPLOAD_FOLDER, page_filename)
#                 print(f"Processing page {page_idx + 1}/{len(page_filenames)}: {page_filename}")

#                 # Perform line segmentation and OCR for the page
#                 segment_prefix = f"{base}_p{page_idx + 1:03d}_"
#                 page_line_result = perform_line_segmentation_ocr(page_image_path, segment_prefix=segment_prefix)

#                 # Regular OCR text for the page
#                 page_regular_text = perform_ocr_inference(page_image_path)

#                 # Save page-level inference JSON (compatible with existing frontend)
#                 page_inference_data = {
#                     'image': page_filename,
#                     'original_text': page_regular_text,
#                     'corrected_text': page_regular_text,
#                     'line_segments': page_line_result.get('line_segments', []),
#                     'total_lines': page_line_result.get('total_lines', 0),
#                     'timestamp': time.time(),
#                     'source_pdf': filename,
#                     'page_index': page_idx,
#                     'page_number': page_idx + 1,
#                 }
#                 page_inference_path = os.path.join(INFERENCE_FOLDER, f"{page_filename}.json")
#                 with open(page_inference_path, 'w', encoding='utf-8') as f:
#                     json.dump(page_inference_data, f, ensure_ascii=False, indent=2)

#                 pages_summary.append({
#                     'filename': page_filename,
#                     'total_lines': page_inference_data['total_lines']
#                 })

#             # Update current image index to the first page image
#             global current_image_index
#             images_only = [f for f in os.listdir(UPLOAD_FOLDER) if is_image_file(f)]
#             if page_filenames and page_filenames[0] in images_only:
#                 current_image_index = images_only.index(page_filenames[0])

#             return jsonify({
#                 'success': True,
#                 'is_pdf': True,
#                 'document_base': base,
#                 'page_count': len(page_filenames),
#                 'pages': pages_summary,
#                 'filename': page_filenames[0] if page_filenames else None,
#                 'total_lines': pages_summary[0]['total_lines'] if pages_summary else 0
#             })
#         else:
#             # Image flow: Perform line segmentation and OCR
#             print("Starting line segmentation and OCR...")
#             line_ocr_result = perform_line_segmentation_ocr(file_path, segment_prefix=os.path.splitext(filename)[0] + "_")

#             if line_ocr_result['success']:
#                 # Also perform regular OCR for backward compatibility
#                 regular_ocr_text = perform_ocr_inference(file_path)
                
#                 # Save inference data with line segments
#                 inference_data = {
#                     'image': filename,
#                     'original_text': regular_ocr_text,
#                     'corrected_text': regular_ocr_text,
#                     'line_segments': line_ocr_result['line_segments'],
#                     'total_lines': line_ocr_result['total_lines'],
#                     'timestamp': time.time()
#                 }
#             else:
#                 # Fallback to regular OCR
#                 regular_ocr_text = perform_ocr_inference(file_path)
#                 inference_data = {
#                     'image': filename,
#                     'original_text': regular_ocr_text,
#                     'corrected_text': regular_ocr_text,
#                     'line_segments': [],
#                     'total_lines': 0,
#                     'timestamp': time.time()
#                 }

#             inference_path = os.path.join(INFERENCE_FOLDER, f"{filename}.json")
#             with open(inference_path, 'w', encoding='utf-8') as f:
#                 json.dump(inference_data, f, ensure_ascii=False, indent=2)

#             # Update current image index to point to the new image
#             global current_image_index
#             images_only = [f for f in os.listdir(UPLOAD_FOLDER) if is_image_file(f)]
#             if filename in images_only:
#                 current_image_index = images_only.index(filename)

#             return jsonify({
#                 'success': True, 
#                 'filename': filename, 
#                 'inference': inference_data['original_text'],
#                 'line_segments': inference_data.get('line_segments', []),
#                 'total_lines': inference_data.get('total_lines', 0)
#             })

#     except Exception as e:
#         print("Upload error:", str(e))
#         return jsonify({'success': False, 'error': str(e)}), 500

# @app.route('/image/<filename>')
# def get_image(filename):
#     """Serve images from upload folder"""
#     try:
#         return send_from_directory(UPLOAD_FOLDER, filename)
#     except Exception as e:
#         print(f"Error serving image {filename}: {e}")
#         return jsonify({'error': 'Image not found'}), 404

# @app.route('/line_segment/<filename>')
# def get_line_segment(filename):
#     """Serve line segment images"""
#     try:
#         return send_from_directory(LINE_SEGMENTS_FOLDER, filename)
#     except Exception as e:
#         print(f"Error serving line segment {filename}: {e}")
#         return jsonify({'error': 'Line segment not found'}), 404

# @app.route('/get_current_image')
# def get_current_image():
#     """Get current image name"""
#     images = [f for f in os.listdir(UPLOAD_FOLDER) if is_image_file(f)]
#     global current_image_index
    
#     if images and 0 <= current_image_index < len(images):
#         return jsonify({'image_name': images[current_image_index]})
#     return jsonify({'image_name': None})

# @app.route('/next_image', methods=['POST'])
# def next_image():
#     """Move to next image"""
#     images = [f for f in os.listdir(UPLOAD_FOLDER) if is_image_file(f)]
#     global current_image_index
    
#     if images:
#         current_image_index = (current_image_index + 1) % len(images)
#         return jsonify({'image_name': images[current_image_index]})
#     else:
#         return jsonify({'image_name': None})

# @app.route('/get_inference/<filename>')
# def get_inference(filename):
#     """Get inference data for a specific image"""
#     try:
#         inference_path = os.path.join(INFERENCE_FOLDER, f"{filename}.json")
#         if os.path.exists(inference_path):
#             with open(inference_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 return jsonify(data)
#         else:
#             return jsonify({'error': 'No inference found', 'image': filename})
#     except Exception as e:
#         print(f"Error loading inference for {filename}: {e}")
#         return jsonify({'error': f'Error loading inference: {str(e)}'})

# @app.route('/update_inference', methods=['POST'])
# def update_inference():
#     """Update corrected text for an image"""
#     try:
#         data = request.json
#         filename = data.get('image')
#         corrected_text = data.get('corrected_text')
        
#         if not filename:
#             return jsonify({'success': False, 'error': 'No filename provided'})
        
#         inference_path = os.path.join(INFERENCE_FOLDER, f"{filename}.json")
        
#         if os.path.exists(inference_path):
#             with open(inference_path, 'r', encoding='utf-8') as f:
#                 inference_data = json.load(f)
            
#             inference_data['corrected_text'] = corrected_text
#             inference_data['last_updated'] = time.time()
            
#             with open(inference_path, 'w', encoding='utf-8') as f:
#                 json.dump(inference_data, f, ensure_ascii=False, indent=2)
            
#             return jsonify({'success': True})
#         else:
#             return jsonify({'success': False, 'error': 'Inference file not found'})
    
#     except Exception as e:
#         print(f"Error updating inference: {e}")
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/update_line_ocr', methods=['POST'])
# def update_line_ocr():
#     """Update OCR text for a specific line"""
#     try:
#         data = request.json
#         filename = data.get('image')
#         line_index = data.get('line_index')
#         corrected_text = data.get('corrected_text')
        
#         if not filename or line_index is None:
#             return jsonify({'success': False, 'error': 'Missing filename or line_index'})
        
#         inference_path = os.path.join(INFERENCE_FOLDER, f"{filename}.json")
        
#         if os.path.exists(inference_path):
#             with open(inference_path, 'r', encoding='utf-8') as f:
#                 inference_data = json.load(f)
            
#             # Update the specific line's OCR text
#             if 'line_segments' in inference_data:
#                 for segment in inference_data['line_segments']:
#                     if segment['line_index'] == line_index:
#                         segment['ocr_text'] = corrected_text
#                         break
            
#             inference_data['last_updated'] = time.time()
            
#             with open(inference_path, 'w', encoding='utf-8') as f:
#                 json.dump(inference_data, f, ensure_ascii=False, indent=2)
            
#             return jsonify({'success': True})
#         else:
#             return jsonify({'success': False, 'error': 'Inference file not found'})
    
#     except Exception as e:
#         print(f"Error updating line OCR: {e}")
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/rerun_inference', methods=['POST'])
# def rerun_inference():
#     """Re-run OCR inference on an image"""
#     try:
#         data = request.json
#         filename = data.get('image')
        
#         if not filename:
#             return jsonify({'success': False, 'error': 'No filename provided'})
        
#         image_path = os.path.join(UPLOAD_FOLDER, filename)
#         if not os.path.exists(image_path):
#             return jsonify({'success': False, 'error': 'Image file not found'})
        
#         # Perform line segmentation and OCR
#         print(f"Re-running line segmentation and OCR for {filename}...")
#         line_ocr_result = perform_line_segmentation_ocr(image_path)
        
#         # Also perform regular OCR
#         regular_ocr_text = perform_ocr_inference(image_path)
        
#         # Update inference file
#         inference_data = {
#             'image': filename,
#             'original_text': regular_ocr_text,
#             'corrected_text': regular_ocr_text,
#             'line_segments': line_ocr_result.get('line_segments', []),
#             'total_lines': line_ocr_result.get('total_lines', 0),
#             'timestamp': time.time(),
#             'rerun_count': 1
#         }
        
#         # Check if file exists and increment rerun count
#         inference_path = os.path.join(INFERENCE_FOLDER, f"{filename}.json")
#         if os.path.exists(inference_path):
#             with open(inference_path, 'r', encoding='utf-8') as f:
#                 existing_data = json.load(f)
#                 inference_data['rerun_count'] = existing_data.get('rerun_count', 0) + 1
        
#         with open(inference_path, 'w', encoding='utf-8') as f:
#             json.dump(inference_data, f, ensure_ascii=False, indent=2)
        
#         return jsonify({
#             'success': True, 
#             'inference': regular_ocr_text,
#             'line_segments': line_ocr_result.get('line_segments', []),
#             'total_lines': line_ocr_result.get('total_lines', 0)
#         })
    
#     except Exception as e:
#         print(f"Error re-running inference: {e}")
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/save', methods=['POST'])
# def save_annotations():
#     """Save annotation data"""
#     try:
#         data = request.json
#         if not data or 'image' not in data:
#             return jsonify({'success': False, 'error': 'Invalid data'})
        
#         # Add timestamp
#         data['saved_at'] = time.time()
        
#         annotation_path = os.path.join(ANNOTATION_FOLDER, f"{data['image']}.json")
#         with open(annotation_path, "w", encoding='utf-8') as f:
#             json.dump(data, f, ensure_ascii=False, indent=2)
        
#         print(f"Annotations saved for {data['image']}: {len(data.get('annotations', []))} boxes")
#         return jsonify({'success': True})
    
#     except Exception as e:
#         print(f"Error saving annotations: {e}")
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/health')
# def health_check():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'healthy',
#         'model_loaded': processor is not None and model is not None,
#         'textline_model_loaded': textline_predictor is not None,
#         'upload_folder': UPLOAD_FOLDER,
#         'folders_exist': {
#             'uploads': os.path.exists(UPLOAD_FOLDER),
#             'annotations': os.path.exists(ANNOTATION_FOLDER),
#             'inferences': os.path.exists(INFERENCE_FOLDER),
#             'line_segments': os.path.exists(LINE_SEGMENTS_FOLDER)
#         }
#     })

# if __name__ == '__main__':
#     print("Starting Flask app with TrOCR and line segmentation integration...")
#     print(f"Upload folder: {UPLOAD_FOLDER}")
#     print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024)}MB")
    
#     # Pre-load the models on startup (optional)
#     # load_trocr_model()
#     # load_textline_model()
    
#     app.run(debug=True, host='0.0.0.0', port=5000)