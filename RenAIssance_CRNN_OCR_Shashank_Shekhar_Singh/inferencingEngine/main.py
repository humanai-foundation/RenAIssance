from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.data as tfd
import io
import csv
import os
import tempfile
import subprocess
import json
from typing import List, Dict, Any
import logging
import pandas as pd

# Load back
with open("vocab.json", "r") as f:
    vocab_loaded = json.load(f)

# Recreate StringLookup
char_to_num = layers.StringLookup(vocabulary=vocab_loaded, mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(),
                                  mask_token=None, invert=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Spanish OCR API", description="API for Spanish text recognition from images")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for model and configurations
inference_model = None
n_classes=50
IMG_WIDTH = 200
IMG_HEIGHT = 50
MAX_LABEL_LENGTH = None
AUTOTUNE = tfd.AUTOTUNE
BATCH_SIZE = 16

class CTCLayer(layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost
    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
        input_len = tf.cast(tf.shape(y_pred)[1], dtype='int64') * tf.ones(shape=(batch_len, 1), dtype='int64')
        label_len = tf.cast(tf.shape(y_true)[1], dtype='int64') * tf.ones(shape=(batch_len, 1), dtype='int64')
        loss = self.loss_fn(y_true, y_pred, input_len, label_len)
        self.add_loss(loss)
        return y_pred

def load_model_and_setup():
    """Load the trained OCR model and setup character mappings"""
    global inference_model, char_to_num, num_to_char, MAX_LABEL_LENGTH
    
    try:
        model_path = 'ocr_model_NEW.h5'
        if os.path.exists(model_path):
            full_model = keras.models.load_model(model_path, compile=False, custom_objects={'CTCLayer': CTCLayer})
            inference_model = keras.Model(
                inputs=full_model.get_layer(name="image").input,
                outputs=full_model.get_layer(name='dense_1').output
            )
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        MAX_LABEL_LENGTH = 24
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_image(image_path : str):
    image = tf.io.read_file(image_path)
    decoded_image = tf.image.decode_jpeg(contents = image, channels = 1)
    cnvt_image = tf.image.convert_image_dtype(image = decoded_image, dtype = tf.float32)
    resized_image = tf.image.resize(images = cnvt_image, size = (IMG_HEIGHT, IMG_WIDTH))
    image = tf.transpose(resized_image, perm = [1, 0, 2])
    image = tf.cast(image, dtype = tf.float32)
    return image

def apply_craft_detection(image_path: str, output_dir: str) -> str:
    """Apply CRAFT model for text detection"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        craft_command = [
            'python3', 'CRAFT_Model/CRAFT/BoundBoxFunc/test.py',
            '--cuda', '0',
            '--result_folder', output_dir,
            '--test_folder', os.path.dirname(image_path),
            '--trained_model', 'CRAFT_Model/CRAFT/BoundBoxFunc/weights/craft_mlt_25k.pth'
        ]
        
        result = subprocess.run(craft_command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"CRAFT detection failed: {result.stderr}")
            raise Exception(f"CRAFT detection failed: {result.stderr}")
        
        logger.info("CRAFT detection completed successfully")
        return output_dir
        
    except Exception as e:
        logger.error(f"Error in CRAFT detection: {e}")
        raise

def count_files_in_folder(folder_path, extensions_list):
    file_count = 0
    for filename in os.listdir(folder_path):
        for extension in extensions_list:
            if filename.lower().endswith(extension):
                file_count += 1
    return file_count

def process_bounding_boxes(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    bounding_boxes = []
    for line in lines:
        coords = list(map(int, line.strip().split(',')))
        bounding_boxes.append(coords)

    bounding_boxes.sort(key=lambda box: box[1])

    vertical_distance_between_lines = 10
    grouped_boxes = []
    current_group = []
    for box in bounding_boxes:
        if not current_group:
            current_group.append(box)
        else:
            min_y = min(current_group, key=lambda x: x[1])[1]
            max_y = max(current_group, key=lambda x: x[1])[1]
            if box[1] - min_y <= vertical_distance_between_lines:
                current_group.append(box)
            else:
                grouped_boxes.append(current_group)
                current_group = [box]

    if current_group:
        grouped_boxes.append(current_group)

    for group in grouped_boxes:
        group.sort(key=lambda box: box[0])

    return grouped_boxes

def sort_bounding_boxes(bounding_box_file):
    sorted_bounding_boxes = process_bounding_boxes(bounding_box_file)

    output_file_path = f"{os.path.splitext(bounding_box_file)[0]}_sorted.txt"
    with open(output_file_path, "w") as outfile:
        for group in sorted_bounding_boxes:
            for box in group:
                outfile.write(','.join(map(str, box)) + '\n')
            outfile.write((';'))
    return output_file_path

def extract_bounding_boxes(image_path, bounding_boxes_file, output_folder, word):
    main_image = cv2.imread(image_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(bounding_boxes_file, 'r') as f:
        bounding_boxes_data = f.read().split(';')
    bounding_boxes_data = bounding_boxes_data[1:]
    line=0
    for indx in range(len(bounding_boxes_data)-1):
        bounding_box_coords = bounding_boxes_data[indx].strip().split('\n')
        for cnt in range(len(bounding_box_coords)):
            coordinates_list = [int(coord) for coord in bounding_box_coords[cnt].split(',')]
            x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max = coordinates_list

            bounding_box = main_image[y_min:y_max, x_min:x_max]

            output_path = os.path.join(output_folder, f'{word};{line}.png')
            cv2.imwrite(output_path, bounding_box)

            word += 1
        line+=1

    return word

def pad_and_resize_images(folder_path):
    if not os.path.exists(folder_path):
        raise ValueError(f"The folder {folder_path} does not exist")

    target_aspect_ratio = 4
    target_width = 200
    target_height = 40

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    img = img.convert('L')
                    width, height = img.size
                    aspect_ratio = width / height

                    if aspect_ratio < target_aspect_ratio:
                        new_width = height * 4
                        padding = (new_width - width) // 2
                        padded_img = ImageOps.expand(img, border=(padding, 0, padding, 0), fill='white')
                    else:
                        padded_img = img

                    resized_img = padded_img.resize((target_width, target_height))
                    resized_img.save(file_path)

                    print(f"Processed and replaced: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

def create_csv_from_folder(folder_path, csv_file_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['FILENAME', 'IDENTITY'])

        for file_name in files:
            if file_name.lower() == ".png":
                continue

            file_name_without_extension = os.path.splitext(file_name)[0]
            csv_writer.writerow([file_name, file_name_without_extension])

    print(f'CSV file "{csv_file_path}" created successfully.')

def encode_single_sample(image_path : str, label : str):
    image = load_image(image_path)
    chars = tf.strings.unicode_split(label, input_encoding='UTF-8')
    vecs = char_to_num(chars)
    pad_size = MAX_LABEL_LENGTH - tf.shape(vecs)[0]
    vecs = tf.pad(vecs, paddings = [[0, pad_size]], constant_values=n_classes+1)
    return {'image':image, 'label':vecs}

def resize_images_in_folder(input_folder, new_size=(200,50)):
    for filename in os.listdir(input_folder):
        with Image.open(os.path.join(input_folder, filename)) as img:
            resized_img = img.resize(new_size)
            output_filename = os.path.splitext(filename)[0] + '.png'
            resized_img.save(os.path.join(input_folder, output_filename))

def decode_pred(pred_label):
    input_len = np.ones(shape=pred_label.shape[0]) * pred_label.shape[1]
    decode = keras.backend.ctc_decode(pred_label, input_length=input_len, greedy=False, beam_width=5)[0][0][:,:MAX_LABEL_LENGTH]
    chars = num_to_char(decode)
    texts = [tf.strings.reduce_join(inputs=char).numpy().decode('UTF-8') for char in chars]
    filtered_texts = [text.replace('[UNK]', " ").strip() for text in texts]
    return filtered_texts

# HTML template for the frontend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Text Recognition</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            max-width: 800px;
            width: 100%;
            margin-top: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header p {
            color: #666;
            font-size: 1.1em;
        }

        .upload-section {
            margin-bottom: 30px;
        }

        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: linear-gradient(45deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
        }

        .upload-area:hover {
            border-color: #764ba2;
            background: linear-gradient(45deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            transform: translateY(-2px);
        }

        .upload-area.dragover {
            border-color: #764ba2;
            background: linear-gradient(45deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
        }

        .upload-icon {
            font-size: 4em;
            color: #667eea;
            margin-bottom: 20px;
        }

        .upload-text {
            color: #333;
            font-size: 1.2em;
            margin-bottom: 10px;
        }

        .upload-subtext {
            color: #666;
            font-size: 0.9em;
        }

        #fileInput {
            display: none;
        }

        .btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s ease;
            margin: 10px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .preview-section {
            margin-bottom: 30px;
            display: none;
        }

        .preview-container {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }

        .image-preview {
            flex: 1;
            min-width: 300px;
        }

        .image-preview img {
            width: 100%;
            max-height: 400px;
            object-fit: contain;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        .results-section {
            display: none;
        }

        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .results-header h3 {
            color: #333;
            font-size: 1.5em;
        }

        .copy-btn {
            background: #28a745;
            font-size: 0.9em;
            padding: 8px 16px;
        }

        .copy-btn:hover {
            background: #218838;
        }

        .results-text {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 20px;
            min-height: 150px;
            font-family: 'Courier New', monospace;
            font-size: 1.1em;
            line-height: 1.6;
            color: #333;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            border: 1px solid #f5c6cb;
        }

        .success {
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            border: 1px solid #c3e6cb;
        }

        .model-info {
            background: linear-gradient(45deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            text-align: center;
        }

        .model-info h3 {
            color: #333;
            margin-bottom: 10px;
        }

        .model-info p {
            color: #666;
            font-size: 0.9em;
        }

        .api-status {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
            padding: 10px;
            border-radius: 10px;
            font-size: 0.9em;
        }

        .api-status.online {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }

        .api-status.offline {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }

        .status-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }

        .status-indicator.online {
            background: #28a745;
        }

        .status-indicator.offline {
            background: #dc3545;
        }

        @media (max-width: 768px) {
            .container {
                padding: 20px;
                margin-top: 10px;
            }

            .header h1 {
                font-size: 2em;
            }

            .upload-area {
                padding: 20px;
            }

            .preview-container {
                flex-direction: column;
            }

            .results-header {
                flex-direction: column;
                gap: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 OCR Text Recognition</h1>
            <p>Upload an image and extract text using advanced deep learning</p>
        </div>

        <div class="api-status" id="apiStatus">
            <span class="status-indicator" id="statusIndicator"></span>
            <span id="statusText">API is online and ready</span>
        </div>

        <div class="model-info">
            <h3>🤖 Spanish OCR Model</h3>
            <p>Powered by CNN + BiLSTM + CTC architecture with attention mechanism</p>
        </div>

        <div class="upload-section">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📁</div>
                <div class="upload-text">Click to upload or drag and drop</div>
                <div class="upload-subtext">Supports JPG, PNG, JPEG files</div>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <button class="btn" onclick="document.getElementById('fileInput').click()">
                    Choose Image
                </button>
                <button class="btn" id="processBtn" onclick="processImage()" disabled>
                    Process Image
                </button>
            </div>
        </div>

        <div class="preview-section" id="previewSection">
            <h3>📸 Image Preview</h3>
            <div class="preview-container">
                <div class="image-preview">
                    <img id="imagePreview" alt="Preview">
                </div>
            </div>
        </div>

        <div class="loading" id="loadingSection">
            <div class="spinner"></div>
            <p>Processing image... Please wait</p>
        </div>

        <div class="results-section" id="resultsSection">
            <div class="results-header">
                <h3>📝 Extracted Text</h3>
                <button class="btn copy-btn" onclick="copyText()">Copy Text</button>
            </div>
            <div class="results-text" id="resultsText"></div>
        </div>

        <div id="messageArea"></div>
    </div>

    <script>
        // Since we're serving from the same origin, use relative URLs
        const API_BASE_URL = window.location.origin;
        
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const processBtn = document.getElementById('processBtn');
        const previewSection = document.getElementById('previewSection');
        const imagePreview = document.getElementById('imagePreview');
        const loadingSection = document.getElementById('loadingSection');
        const resultsSection = document.getElementById('resultsSection');
        const resultsText = document.getElementById('resultsText');
        const messageArea = document.getElementById('messageArea');
        const apiStatus = document.getElementById('apiStatus');
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');

        let selectedFile = null;
        
        // Set status as online since we're serving from the same app
        apiStatus.className = 'api-status online';
        statusIndicator.className = 'status-indicator online';
        statusText.textContent = 'API is online and ready';

        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelect(files[0]);
            }
        });

        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        function handleFileSelect(file) {
            if (!file.type.startsWith('image/')) {
                showMessage('Please select a valid image file.', 'error');
                return;
            }

            selectedFile = file;
            processBtn.disabled = false;

            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                previewSection.style.display = 'block';
            };
            reader.readAsDataURL(file);

            // Hide previous results
            resultsSection.style.display = 'none';
            clearMessages();
        }

        async function processImage() {
            if (!selectedFile) {
                showMessage('Please select an image first.', 'error');
                return;
            }

            // Show loading
            loadingSection.style.display = 'block';
            resultsSection.style.display = 'none';
            processBtn.disabled = true;

            try {
                const formData = new FormData();
                formData.append('file', selectedFile); 

                const response = await fetch(`${API_BASE_URL}/ocr/predict`, { 
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();

                // Hide loading
                loadingSection.style.display = 'none';
                processBtn.disabled = false;

                // Show results
                if (result.status === 'success' && result.extracted_text) {
                    resultsText.textContent = result.extracted_text;
                    resultsSection.style.display = 'block';
                    showMessage(`Text extraction completed successfully!`, 'success');
                } else if (result.error) {
                    showMessage(`Error: ${result.error}`, 'error');
                } else {
                    showMessage('No text could be extracted from the image.', 'error');
                }

            } catch (error) {
                // Hide loading
                loadingSection.style.display = 'none';
                processBtn.disabled = false;
                
                console.error('Error processing image:', error);
                showMessage(`Error processing image: ${error.message}`, 'error');
            }
        }

        function copyText() {
            const text = resultsText.textContent;
            navigator.clipboard.writeText(text).then(() => {
                showMessage('Text copied to clipboard!', 'success');
            }).catch(() => {
                showMessage('Failed to copy text to clipboard.', 'error');
            });
        }

        function showMessage(message, type) {
            clearMessages();
            const messageDiv = document.createElement('div');
            messageDiv.className = type;
            messageDiv.textContent = message;
            messageArea.appendChild(messageDiv);
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                messageDiv.remove();
            }, 5000);
        }

        function clearMessages() {
            messageArea.innerHTML = '';
        }
    </script>
</body>
</html>
"""

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    try:
        load_model_and_setup()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

# Serve the HTML frontend at the root
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the HTML frontend"""
    return HTMLResponse(content=HTML_TEMPLATE)

@app.post("/ocr/predict", response_model=Dict[str, Any])
async def predict_text(file: UploadFile = File(...)):
    """
    Extract and recognize text from an uploaded image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read uploaded image
        contents = await file.read()
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded image
            image_path = os.path.join(temp_dir, f"input_{file.filename}")
            with open(image_path, 'wb') as f:
                f.write(contents)
            
            # Apply CRAFT detection
            craft_output_dir = os.path.join(temp_dir, "craft_output/")
            craft_output_dir = apply_craft_detection(image_path, craft_output_dir)
            # Find bounding box file
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            bbox_file = os.path.join(craft_output_dir, f"res_{image_basename}.txt")
            if not os.path.exists(bbox_file):
                raise HTTPException(status_code=404, detail="No text detected in image")
            
            # Sort bounding boxes
            sorted_file = sort_bounding_boxes(bbox_file)
            if not sorted_file:
                raise HTTPException(status_code=404, detail="No valid bounding boxes found")
            
            # Extract word images
            word = 0
            output_folder = os.path.join(temp_dir, "extracted_word_images/")
            word = extract_bounding_boxes(image_path, sorted_file, output_folder, word)
            pad_and_resize_images(output_folder)
            test_csv_path = os.path.join(temp_dir, 'testing_data.csv')
            create_csv_from_folder(output_folder, test_csv_path)

            test_csv = pd.read_csv(test_csv_path)
            test_csv['IDENTITY'] = test_csv['IDENTITY'].apply(lambda x: str(x))
            test_csv['FILENAME']  = [output_folder + f"/{filename}" for filename in test_csv['FILENAME']]

            resize_images_in_folder(output_folder)

            df_infer = test_csv

            # Step 1: Sort the dataframe based on values before ';'
            df_infer['before_semicolon'] = df_infer['IDENTITY'].apply(lambda x: int(x.split(';')[0]))
            df_infer['after_semicolon'] = df_infer['IDENTITY'].apply(lambda x: int(x.split(';')[1]))
            sorted_df = df_infer.sort_values(['before_semicolon']).reset_index(drop=True)
            sorted_df.drop(columns=['before_semicolon', 'after_semicolon'], inplace=True)

            sorted_df['IDENTITY'] = sorted_df['IDENTITY'].astype(str)

            sorted_dfs = tf.data.Dataset.from_tensor_slices(
                (np.array(sorted_df['FILENAME'].to_list()), np.array(sorted_df['IDENTITY'].to_list()))
            ).map(encode_single_sample, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

            decoded_predictions = decode_pred(inference_model.predict(sorted_dfs))
            pred = sorted_df['IDENTITY'].tolist()
            formatted_output = []

            current_group = None
            i = 0
            for prediction in pred:
                before, after = map(int, prediction.split(';'))

                if current_group is None:
                    current_group = after

                if after != current_group:
                    formatted_output.append('\n')  # Start a new line for the new group
                    current_group = after

                formatted_output.append(decoded_predictions[i] + ' ')
                i += 1

            formatted_output.append('\n')  # Final new line

            full_text = ''.join(formatted_output)

            print(full_text)
            return {
                "status": "success",
                "extracted_text": full_text,
                "message": "Text extracted successfully"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": inference_model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
