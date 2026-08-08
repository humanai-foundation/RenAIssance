# Document Image Segmentation App with Hi-SAM

A comprehensive GUI application that integrates Hi-SAM (Hierarchical Segment Anything Model) for high-precision text line detection and segmentation from document images, with advanced OCR capabilities using TrOCR and OpenAI-powered enhancement.

## Features

### 1. Hi-SAM Integration
- **Hi-SAM Model**: High-precision hierarchical text line segmentation
- **Segmentation Masks**: Pixel-level masks instead of simple bounding boxes
- **Hierarchical Text Understanding**: Processing at stroke, word, text-line, and paragraph levels
- **Automatic Mask Generation**: Intelligent text region detection

### 2. Advanced OCR Pipeline
- **TrOCR Integration**: Microsoft's state-of-the-art OCR model for handwritten and printed text
- **OCR Enhancement**: OpenAI GPT-powered text correction and improvement
- **Confidence Analysis**: Low-confidence token identification and targeted enhancement
- **Multiple Output Formats**: Plain text, Word documents with detailed analysis

### 3. Document Processing
- **Multi-format Support**: PDF, PNG, JPG, JPEG files
- **PDF Navigation**: Page-by-page processing with navigation controls
- **Preprocessing Pipeline**: Binarization, rotation, distortion correction
- **Interactive Display**: Zoom, pan, and scroll capabilities

### 4. Segmentation Mask Management
- **Interactive Selection**: Click to select individual masks
- **Mask Editing**: Delete unwanted segmentation regions
- **Visual Feedback**: Selected masks highlighted in red, others in green
- **Batch Operations**: Save all detected text regions or selected ones

### 5. Export and Analysis
- **Image Extraction**: Save cropped images for each detected text line
- **OCR Results Export**: Export recognition results to text files
- **Enhanced Word Documents**: Generate Word documents with confidence analysis
- **Flexible Output Options**: Customizable save directories and file naming

## Prerequisites

### System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.8 or higher (3.12 recommended)
- **GPU**: CUDA-compatible GPU recommended for optimal performance
- **Memory**: 8GB RAM minimum, 16GB recommended

### Required Software
- **Git**: For cloning repositories
- **Python Package Manager**: uv (recommended) or pip

## Installation and Setup

### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd line_segmentation_app
```

### Step 2: Install Hi-SAM
```bash
# Clone Hi-SAM repository
git clone https://github.com/ymy-k/Hi-SAM.git
cd Hi-SAM

# Install Hi-SAM requirements
pip install -r requirements.txt
cd ..
```

### Step 3: Install Python Dependencies
Using uv (recommended):
```bash
# Initialize uv environment
uv init

# Install dependencies
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu118
uv add opencv-python pillow numpy pymupdf matplotlib transformers openai python-docx
uv add scipy scikit-image shapely pyclipper tqdm einops timm
```

Using pip:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install opencv-python pillow numpy pymupdf matplotlib
pip install transformers openai python-docx
pip install scipy scikit-image shapely pyclipper tqdm einops timm
```

### Step 4: Download Model Weights

#### Required SAM Weights
Download the base SAM weights and place them in `Hi-SAM/pretrained_checkpoint/`:

1. **SAM ViT-B**: [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
2. **SAM ViT-L**: [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) (optional)

#### Hi-SAM Model Weights
Download at least one Hi-SAM model weight:

**Recommended for beginners:**
- **Efficient Hi-SAM-S**: [Download](https://1drv.ms/u/s!AimBgYV7JjTlgcpZZz-xZiDiRBjfLQ?e=GK4uHo) (Lightweight, faster inference)

**For better accuracy:**
- **Hi-SAM-B**: [Download](https://1drv.ms/u/s!AimBgYV7JjTlgcosk3ZK1dImhxaW9g?e=xTsegH) (Balanced performance)
- **Hi-SAM-L**: [Download](https://1drv.ms/u/s!AimBgYV7JjTlgcovMjJKfH6baFBTGw?e=T3IrUf) (High accuracy)

#### Directory Structure After Setup
```
line_segmentation_app/
├── main.py
├── ocr_enhancement.py
├── settings_dialog.py
├── word_generator.py
├── create_samples.py
├── pyproject.toml
├── README.md
└── Hi-SAM/
    ├── pretrained_checkpoint/
    │   ├── sam_vit_b_01ec64.pth          # SAM base model
    │   ├── efficient_hi_sam_s.pth        # Efficient Hi-SAM (recommended)
    │   ├── hi_sam_b.pth                  # Hi-SAM Base (optional)
    │   └── sam_vit_l_0b3195.pth          # SAM Large (optional)
    ├── hi_sam/
    │   └── modeling/
    └── requirements.txt
```

### Step 5: Configure OpenAI API (Optional)
For OCR enhancement features:
1. Sign up for OpenAI API access
2. Get your API key from the OpenAI dashboard
3. Enter the API key in the application's enhancement settings

## Usage Guide

### Basic Workflow

1. **Launch Application**
   ```bash
   uv run python main.py
   ```

2. **Load Document**
   - Click "Open File" button
   - Select PDF, PNG, JPG, or JPEG file
   - Use navigation buttons for multi-page PDFs

3. **Adjust Display**
   - Use scale slider (0.1x - 3.0x) for zoom
   - Adjust X/Y position sliders for panning
   - Use mouse wheel for additional scrolling

4. **Apply Preprocessing (Optional)**
   - **Binarization**: Enable checkbox and adjust threshold (0-255)
   - **Rotation**: Use slider for -180° to +180° rotation
   - **Distortion Correction**: Enable checkbox for simple correction

5. **Run Text Detection**
   - Click "Run Recognition" button
   - Wait for Hi-SAM to process the image
   - Segmentation masks will appear as colored overlays

6. **Interact with Results**
   - **Select masks**: Click on any mask to select (turns red)
   - **Delete masks**: Select unwanted masks and click "Delete Selected"
   - **Adjust view**: Use display controls for better visibility

7. **Run OCR (Optional)**
   - Click "Run OCR" to extract text using TrOCR
   - Results will be displayed in console
   - Click "Save OCR Results" to export as text file

8. **Enhance OCR (Optional)**
   - Enter OpenAI API key in the enhancement section
   - Adjust enhancement iterations (1-10)
   - Set low confidence token threshold
   - Click "Enhance OCR" for AI-powered correction
   - Click "Save to Word" for detailed analysis document

9. **Export Results**
   - **Select Directory**: Choose where to save results
   - **Save All**: Export all detected text regions as images
   - **Save Selected**: Export only the selected mask region

### Advanced Features

#### OCR Enhancement Pipeline
1. **TrOCR Processing**: Initial OCR using Microsoft's TrOCR model
2. **Confidence Analysis**: Identifies low-confidence predictions
3. **AI Enhancement**: Uses OpenAI GPT for context-aware correction
4. **ROVER Consensus**: Combines multiple enhancement iterations
5. **Word Export**: Generates comprehensive analysis documents

#### Hi-SAM Model Selection
The application automatically selects the best available model:
- **Efficient Hi-SAM-S**: Fastest, good for real-time processing
- **Hi-SAM-B**: Balanced speed and accuracy
- **Hi-SAM-L**: Highest accuracy, slower processing

#### Batch Processing Tips
- Process multiple pages of PDFs sequentially
- Save results to organized directory structures
- Use consistent preprocessing settings for similar documents

## File Structure

```
line_segmentation_app/
├── main.py                    # Main application with GUI
├── ocr_enhancement.py         # OCR enhancement pipeline
├── settings_dialog.py         # Settings configuration dialog
├── word_generator.py          # Word document generation
├── create_samples.py          # Sample data creation utility
├── pyproject.toml            # Project configuration and dependencies
├── README.md                 # This documentation
└── Hi-SAM/                   # Hi-SAM submodule
    ├── hi_sam/               # Hi-SAM model implementation
    ├── pretrained_checkpoint/ # Model weights directory
    └── requirements.txt      # Hi-SAM specific requirements
```

## Configuration Options

### Model Configuration
- **Model Type**: Automatically detected from available weights
- **Device**: Auto-selects CUDA if available, falls back to CPU
- **Batch Processing**: Configurable for memory optimization

### OCR Settings
- **Enhancement Iterations**: 1-10 (default: 3)
- **Low Confidence Threshold**: Number of tokens to focus enhancement on
- **API Rate Limiting**: Built-in delays to respect OpenAI API limits

### Export Settings
- **Image Format**: PNG with transparency support
- **Text Encoding**: UTF-8 for international character support
- **Word Document**: Includes confidence analysis and enhancement details

## Troubleshooting

### Installation Issues

**PyTorch Installation:**
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Hi-SAM Import Errors:**
- Ensure Hi-SAM directory is in the same folder as main.py
- Check that all Hi-SAM requirements are installed
- Verify model weights are in the correct directory

### Runtime Issues

**Model Loading Errors:**
- Check if model files exist in `Hi-SAM/pretrained_checkpoint/`
- Ensure sufficient memory (8GB+ recommended)
- Try using Efficient Hi-SAM-S for lower memory usage

**OCR Enhancement Errors:**
- Verify OpenAI API key is valid and has credits
- Check internet connection for API calls
- Reduce enhancement iterations if hitting rate limits

**Performance Issues:**
- Use GPU acceleration when available
- Reduce image resolution for faster processing
- Close other memory-intensive applications

### Common Error Messages

**"Hi-SAM not available":**
- Hi-SAM repository not cloned or not in correct location
- Missing dependencies in Hi-SAM requirements.txt

**"TrOCR not available":**
- Transformers library not installed: `uv add transformers`
- Model download failed - check internet connection

**"Enhancement modules not available":**
- Missing OpenAI library: `uv add openai`
- Missing python-docx library: `uv add python-docx`

## Performance Optimization

### Memory Management
- Use Efficient Hi-SAM-S for lower memory usage
- Process large documents page by page
- Close application between processing sessions for memory cleanup

### GPU Acceleration
- Ensure CUDA-compatible GPU is available
- Install appropriate PyTorch version with CUDA support
- Monitor GPU memory usage during processing

### Processing Speed
- Use lower resolution images when possible
- Reduce enhancement iterations for faster OCR processing
- Consider batch processing for multiple similar documents

## Contributing

Contributions are welcome! Please consider:
- Bug reports and feature requests via issues
- Code improvements via pull requests
- Documentation enhancements
- Model performance optimizations

## License

This project integrates multiple components with different licenses:
- Hi-SAM: Check the original Hi-SAM repository for license terms
- TrOCR: Microsoft's model with specific usage terms
- OpenAI API: Subject to OpenAI's usage policies

## Citation

If you use this project in academic work, please cite the original Hi-SAM paper:
```bibtex
@article{zhang2024hi,
  title={Hi-SAM: Marrying Segment Anything Model for Hierarchical Text Segmentation},
  author={Zhang, Maoyuan and others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024}
}
uv add opencv-python onnxruntime pillow numpy PyMuPDF matplotlib
```

### 2. Launch Application
```bash
uv run python main.py
```

### 3. Basic Operation Steps
1. **File selection**: Select image or PDF using "Open File" button
2. **Display adjustment**: Adjust zoom and position sliders for better viewing
3. **Preprocessing**: Apply binarization, rotation, or distortion correction as needed
4. **Run recognition**: Detect text regions using "Run Recognition" button
5. **Result adjustment**: Adjust bounding box sizes and positions

## File Structure

```
line_segmentation_app/
├── main.py          # Main application
├── model.onnx       # Detectron2 Mask R-CNN model
├── pyproject.toml   # Project configuration
└── README.md        # This file
```

## About the Model

- **Format**: ONNX
- **Base**: Detectron2 Mask R-CNN
- **Purpose**: Text extraction from document images
- **Input**: Image data (resized to 640x640)
- **Output**: List of bounding box coordinates (output index 0)

## Customization

### Model Output Format Adjustment
Adjust the output processing in the `run_recognition()` method according to your model's output format:

```python
# Assume first element of output is bounding boxes
predictions = outputs[0]

# Adjust according to output format
if len(predictions.shape) >= 2:
    for detection in predictions[0]:  # Get first batch
        if len(detection) >= 4:
            # Convert coordinates to original image size
            x1 = int(detection[0] / scale_x)
            y1 = int(detection[1] / scale_y)
            x2 = int(detection[2] / scale_x)
            y2 = int(detection[3] / scale_y)
            
            # Confidence check (if confidence is included)
            confidence = detection[4] if len(detection) > 4 else 1.0
            if confidence > 0.5:  # Threshold
                self.bounding_boxes.append([x1, y1, x2, y2])
```

### Extending Preprocessing
To add more advanced preprocessing, extend the `apply_preprocessing()` method.

## Acknowledgments

This project builds upon several excellent open-source projects:
- **Hi-SAM**: Hierarchical Segment Anything Model for text segmentation
- **TrOCR**: Microsoft's Transformer-based OCR model  
- **Segment Anything Model (SAM)**: Meta's foundation model for image segmentation
- **OpenAI**: GPT models for text enhancement and correction

## Support

For support and questions:
1. Check this documentation first
2. Review the troubleshooting section
3. Check the original Hi-SAM repository for model-specific issues
4. Create an issue in this repository for application-specific problems

## Version History

- **v0.1.0**: Initial release with Hi-SAM integration, TrOCR OCR, and OpenAI enhancement
