# Advanced OCR Flask Application with Gemini API Integration

This Flask application provides advanced OCR capabilities with line segmentation, TrOCR processing, and Gemini API text correction for historical Spanish documents.

## Features

- **Advanced Line Segmentation**: Uses Detectron2-based textline detection with dynamic padding
- **TrOCR Processing**: Microsoft TrOCR for accurate text recognition
- **Gemini API Integration**: Google Gemini 2.5 Pro for text correction and post-processing
- **Reading Order Detection**: Automatic sorting of textlines in proper reading order
- **RESTful API**: Full API endpoints for upload, processing, and correction
- **Real-time Processing**: Immediate feedback with progress indicators

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Gemini API key:
   - The API key is already configured in the code
   - Or set environment variable: `GEMINI_API_KEY=your_api_key`

3. Download the Detectron2 model:
   - Place your model file at the specified path in the code
   - Update the model path in `load_textline_model()` function

## Usage

1. Start the Flask server:
```bash
python app2.py
```

2. Access the web interface at `http://localhost:5000`

3. Upload images or PDFs for processing

## API Endpoints

### Core Endpoints
- `POST /upload` - Upload and process images
- `GET /image/<filename>` - Retrieve uploaded images
- `GET /line_segment/<filename>` - Retrieve line segment images
- `GET /get_inference/<filename>` - Get processing results

### Processing Endpoints
- `POST /rerun_inference` - Re-run OCR on existing images
- `POST /apply_gemini_correction` - Apply Gemini correction to existing results
- `POST /update_inference` - Update corrected text
- `POST /update_line_ocr` - Update specific line OCR results

### Status Endpoints
- `GET /health` - System health and model status
- `GET /get_current_image` - Get current image information

## Processing Pipeline

1. **Image Upload**: Accepts various image formats and PDFs
2. **Line Segmentation**: Advanced Detectron2-based textline detection
3. **Dynamic Padding**: Intelligent padding based on text spacing
4. **Reading Order**: Automatic sorting in proper reading sequence
5. **TrOCR Processing**: High-quality text recognition
6. **Gemini Correction**: AI-powered text correction and enhancement
7. **Result Storage**: Comprehensive JSON output with metadata

## Output Format

The application generates structured JSON output containing:
- Original and corrected text
- Line segment information with bounding boxes
- Processing pipeline metadata
- Gemini API correction status
- Confidence scores and reading order

## Configuration

Key configuration options:
- `GEMINI_API_KEY`: Your Gemini API key
- `UPLOAD_FOLDER`: Directory for uploaded files
- `MAX_CONTENT_LENGTH`: Maximum file size (16MB default)
- Model paths for Detectron2 and TrOCR

## Error Handling

The application includes comprehensive error handling:
- Graceful fallbacks when models are unavailable
- Retry logic for API calls
- Detailed error messages and logging
- Health check endpoints for monitoring

## Dependencies

- Flask 2.3.3
- PyTorch and Transformers
- OpenCV and NumPy
- Detectron2 for object detection
- Google Generative AI for text correction
- PyMuPDF for PDF processing 