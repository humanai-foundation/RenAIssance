from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import json
import time
from werkzeug.utils import secure_filename
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ANNOTATION_FOLDER = 'annotations'
INFERENCE_FOLDER = 'inferences'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}

# Create directories
for folder in [UPLOAD_FOLDER, ANNOTATION_FOLDER, INFERENCE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model (loaded once)
processor = None
model = None
current_image_index = 0

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
            # Fallback to base model if Spanish model fails
            try:
                processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
                model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
                print("Fallback TrOCR model loaded successfully!")
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                raise e2

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def perform_ocr_inference(image_path):
    """Perform TrOCR inference on an image"""
    try:
        # Load model if not already loaded
        load_trocr_model()
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        
        # Generate text
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return f"Error during OCR: {str(e)}"

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
    
    # Check if file part exists
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'success': False, 'error': 'No file part in the request'}), 400

    file = request.files['file']

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

        # Perform OCR
        print("Starting OCR inference...")
        inference_text = perform_ocr_inference(file_path)
        print(f"OCR completed: {inference_text[:100]}...")

        # Save inference data
        inference_data = {
            'image': filename,
            'original_text': inference_text,
            'corrected_text': inference_text,
            'timestamp': time.time()
        }
        
        inference_path = os.path.join(INFERENCE_FOLDER, f"{filename}.json")
        with open(inference_path, 'w', encoding='utf-8') as f:
            json.dump(inference_data, f, ensure_ascii=False, indent=2)

        # Update current image index to point to the new image
        global current_image_index
        images = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
        if filename in images:
            current_image_index = images.index(filename)

        return jsonify({
            'success': True, 
            'filename': filename, 
            'inference': inference_text
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

@app.route('/get_current_image')
def get_current_image():
    """Get current image name"""
    images = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
    global current_image_index
    
    if images and 0 <= current_image_index < len(images):
        return jsonify({'image_name': images[current_image_index]})
    return jsonify({'image_name': None})

@app.route('/next_image', methods=['POST'])
def next_image():
    """Move to next image"""
    images = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
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
        
        # Perform OCR inference
        print(f"Re-running OCR inference for {filename}...")
        inference_text = perform_ocr_inference(image_path)
        
        # Update inference file
        inference_data = {
            'image': filename,
            'original_text': inference_text,
            'corrected_text': inference_text,
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
        
        return jsonify({'success': True, 'inference': inference_text})
    
    except Exception as e:
        print(f"Error re-running inference: {e}")
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

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': processor is not None and model is not None,
        'upload_folder': UPLOAD_FOLDER,
        'folders_exist': {
            'uploads': os.path.exists(UPLOAD_FOLDER),
            'annotations': os.path.exists(ANNOTATION_FOLDER),
            'inferences': os.path.exists(INFERENCE_FOLDER)
        }
    })

if __name__ == '__main__':
    print("Starting Flask app with TrOCR integration...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024)}MB")
    
    # Pre-load the model on startup (optional)
    # load_trocr_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000)