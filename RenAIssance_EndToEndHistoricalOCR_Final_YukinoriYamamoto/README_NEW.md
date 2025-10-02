# Document Image Segmentation App with AI-Enhanced OCR

A comprehensive GUI application for document image processing, text line segmentation using Hi-SAM, OCR with TrOCR, and AI-enhanced transcription using OpenAI API with ROVER consensus algorithm.

## Features

### Core Features
- **Document Loading**: Support for PDF files and images (PNG, JPG, JPEG)
- **Multi-page Support**: Navigate through PDF pages
- **Image Preprocessing**: Binarization, rotation, distortion correction
- **Interactive Display**: Zoom, pan, and view segmentation results

### Advanced Segmentation
- **Hi-SAM Integration**: State-of-the-art text line detection using Hierarchical SAM
- **Model Options**: Support for multiple model sizes (vit_s, vit_b, vit_l)
- **Interactive Editing**: Select, delete, and modify segmentation masks

### OCR Capabilities
- **TrOCR Integration**: Transformer-based OCR for high accuracy
- **Multi-language Support**: Optimized for handwritten Spanish documents
- **Batch Processing**: Process all detected text lines automatically

### AI-Enhanced Transcription (New!)
- **OpenAI Integration**: Use GPT-4 Vision for transcription refinement
- **ROVER Consensus**: Multiple AI corrections combined using ROVER algorithm
- **Confidence Analysis**: Per-token confidence scores for quality assessment
- **Word Document Output**: Professional documents with confidence-based highlighting

## Installation

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (recommended for faster processing)
- OpenAI API key (for AI enhancement features)

### Install with uv (Recommended)
```bash
# Clone the repository
git clone https://github.com/yamanoko/line_segmentation_app.git
cd line_segmentation_app

# Install dependencies with uv
uv sync
```

### Manual Installation
```bash
pip install torch torchvision torchaudio
pip install transformers opencv-python pillow pymupdf numpy scipy scikit-image
pip install shapely pyclipper tqdm einops timm
pip install openai python-docx  # For AI enhancement features
```

## Usage

### Basic Workflow
1. **Open File** - Load a document image or PDF
2. **Preprocessing** - Adjust rotation, binarization if needed
3. **Run Recognition** - Detect text lines using Hi-SAM
4. **Run OCR** - Extract text using TrOCR
5. **Save Results** - Export as text file

### Advanced AI Enhancement Workflow
1. Complete basic workflow through OCR
2. **Settings** - Configure OpenAI API key and parameters
3. **Enhance OCR** - Run AI-enhanced transcription with ROVER
4. **Save Enhanced Word** - Export with confidence highlighting

### Configuration

#### OpenAI Settings
- **API Key**: Required for AI enhancement
- **Model**: Choose from gpt-4o-mini, gpt-4o, or gpt-4-turbo
- **Enhancement Iterations**: Number of AI corrections per line (1-10)
- **Low Confidence Tokens**: Number of tokens to highlight in red

#### Hi-SAM Models
The application supports three model variants:
- **vit_s**: Efficient Hi-SAM Small (fastest, lowest memory)
- **vit_b**: Hi-SAM Base (balanced performance)
- **vit_l**: Hi-SAM Large (highest accuracy, requires more memory)

## File Structure

```
line_segmentation_app/
├── main.py                 # Main application
├── ocr_enhancement.py      # OpenAI and ROVER integration
├── word_generator.py       # Word document generation
├── settings_dialog.py      # Settings management
├── Hi-SAM/                 # Hi-SAM model files
│   ├── pretrained_checkpoint/
│   └── hi_sam/
└── README.md
```

## AI Enhancement Details

### ROVER Algorithm
The ROVER (Recognizer Output Voting Error Reduction) algorithm combines multiple OCR outputs:
1. **Multiple Transcriptions**: Each line processed multiple times with AI
2. **Token Alignment**: Align different transcription variants
3. **Consensus Voting**: Most frequent token becomes consensus
4. **Confidence Calculation**: Based on agreement between variants

### OpenAI Integration
- **Specialized Prompt**: Optimized for historical Spanish handwritten documents
- **Vision API**: Uses GPT-4 Vision to analyze actual text line images
- **Temperature Control**: Low temperature for consistent results
- **Error Handling**: Graceful fallback to original transcription

### Word Document Output
- **Confidence Highlighting**: Low-confidence tokens highlighted in red
- **Detailed Analysis**: Statistics and variant transcriptions included
- **Professional Formatting**: Clean, readable document structure
- **Metadata**: Processing information and confidence metrics

## Performance Optimization

### GPU Usage
- Automatic CUDA detection and utilization
- Memory management for large documents
- Model caching for improved performance

### Processing Tips
- Use lower resolution for faster processing
- Choose appropriate Hi-SAM model based on accuracy/speed needs
- Adjust enhancement iterations based on quality requirements

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce image resolution or use CPU mode
2. **Model Loading Errors**: Ensure checkpoint files are present
3. **API Errors**: Verify OpenAI API key and quota
4. **Slow Processing**: Consider using lighter models or reducing iterations

### Performance Issues
- **Large PDFs**: Process pages individually
- **Memory Usage**: Close application between large documents
- **Network Issues**: Check internet connection for OpenAI API calls

## Dependencies

### Core Dependencies
- torch >= 1.10.0
- transformers >= 4.56.0
- opencv-python >= 4.8.0
- pillow >= 10.0.0
- pymupdf >= 1.26.0

### AI Enhancement Dependencies
- openai >= 1.109.0
- python-docx >= 1.2.0

### Hi-SAM Dependencies
- scipy >= 1.10.1
- scikit-image >= 0.21.0
- shapely >= 2.0.2
- pyclipper >= 1.3.0
- timm >= 0.9.0
- einops >= 0.6.1

## Technical Architecture

### Processing Pipeline
1. **Document Loading**: PDF/image input with page handling
2. **Preprocessing**: Image enhancement and normalization
3. **Segmentation**: Hi-SAM text line detection
4. **OCR**: TrOCR text extraction
5. **AI Enhancement**: OpenAI Vision API corrections
6. **ROVER Consensus**: Multiple variant combination
7. **Output**: Text files and Word documents with confidence analysis

### Model Integration
- **Hi-SAM**: Hierarchical segmentation for precise text line detection
- **TrOCR**: Transformer-based OCR optimized for handwritten text
- **GPT-4 Vision**: Advanced AI proofreading and correction
- **ROVER**: Statistical consensus for improved accuracy

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this application in your research, please cite:
```bibtex
@software{line_segmentation_app,
  title={Document Image Segmentation App with AI-Enhanced OCR},
  author={Your Name},
  year={2025},
  url={https://github.com/yamanoko/line_segmentation_app}
}
```

## Acknowledgments

- Hi-SAM: Hierarchical Segment Anything Model
- TrOCR: Transformer-based Optical Character Recognition
- ROVER: Recognizer Output Voting Error Reduction
- OpenAI GPT-4 Vision API