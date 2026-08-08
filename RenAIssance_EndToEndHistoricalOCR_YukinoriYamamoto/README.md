# Lensaissance

A specialized OCR (Optical Character Recognition) application developed as part of the RenAIssance project for digitizing historical Spanish documents. Originally created during Google Summer of Code 2024-2025 by Yukinori Yamamoto (Waseda University, Japan), Lensaissance addresses the unique challenges of processing Renaissance-era documents that conventional OCR tools struggle with due to their distinctive spelling, print styles, and deterioration.

Building upon a custom AI model trained with Self Supervised Learning that achieves 95% accuracy on historical document characters, this application provides an intuitive interface for humanities researchers to digitize documents without requiring technical expertise. The tool features interactive text selection, advanced preprocessing capabilities, and AI-powered text correction through OpenAI's GPT models.

## Project Background

### RenAIssance Initiative
The RenAIssance project is dedicated to developing specialized OCR tools for historical Spanish documents, specifically targeting the needs of humanities researchers who may not have extensive computer engineering backgrounds. Renaissance-era Spanish documents present unique digitization challenges due to:

- **Historical Spelling Variations**: Significant differences from modern Spanish orthography
- **Distinctive Print Styles**: Typography and formatting conventions of the era
- **Document Deterioration**: Age-related damage affecting text clarity
- **Unique Characteristics**: Features that cause conventional OCR tools to fail

### Development History
This application represents the evolution of work begun during Google Summer of Code, where the core AI model was developed with Self Supervised Learning techniques achieving 95% accuracy on historical document character recognition. The current version addresses practical implementation challenges, focusing on preprocessing and user accessibility to bridge the gap between advanced AI capabilities and real-world research needs.

### Design Philosophy
Rather than attempting end-to-end document recognition, Lensaissance adopts a focused approach: enabling users to select specific regions for digitization. This interactive methodology allows researchers to maintain control over the digitization process while leveraging powerful AI models for accurate text extraction.

## Features

### 🔍 OCR Capabilities
- **Historical Document Specialization**: Optimized for Renaissance-era Spanish documents with unique characteristics
- **Interactive Region Selection**: Click and drag to select specific text areas for precise digitization
- **Advanced AI Engine**: Powered by Surya OCR toolkit supporting 90+ languages with specialized recognition models
- **Multi-format Support**: Process both image files (PNG, JPG, JPEG) and PDF documents
- **Multi-page Support**: Handle multi-page PDFs with individual page processing
- **Local Processing**: Runs entirely offline for security and accessibility

### 🖼️ Image Processing
- **Real-time Image Preview**: View and navigate through your documents with zoom controls
- **Image Preprocessing**: Built-in binarization with adjustable threshold for better OCR accuracy
- **Zoom Controls**: Scale images from 0.5x to 3x for detailed inspection
- **Scrollable Canvas**: Navigate large documents with horizontal and vertical scrollbars

### 🤖 AI-Powered Text Correction
- **OpenAI Integration**: Leverage GPT-3.5-turbo for intelligent text proofreading
- **Error Correction**: Automatically fix spelling, grammar, and formatting errors
- **Interactive Review**: Compare original and corrected text before applying changes
- **Batch Processing**: Combine text from multiple pages into a single document

### 🎓 Accessibility for Researchers
- **No Technical Expertise Required**: Designed specifically for humanities researchers without programming backgrounds
- **Standalone Executable**: Packaged as a .exe file requiring no additional software installation
- **Intuitive Interface**: Simple drag-and-select workflow for document digitization
- **Fast Processing**: Optimized for quick, interactive text extraction sessions
### 💾 File Management
- **Flexible Output**: Choose your preferred output directory
- **Automatic Saving**: OCR results are automatically saved as text files
- **Page-based Organization**: Each page is saved as a separate text file
- **Text Merging**: Combine all pages into a single text file

## Target Users

This application is specifically designed for:
- **Humanities Researchers**: Scholars working with historical Spanish documents
- **Digital Humanities Projects**: Teams digitizing Renaissance-era manuscripts and texts
- **Academic Institutions**: Universities and research centers with historical document collections
- **Archives and Libraries**: Organizations preserving and digitizing historical materials

While optimized for Spanish Renaissance documents, the application's Surya engine supports over 90 languages, making it valuable for broader historical document research.

## Installation

### Prerequisites
- Python 3.12 or higher
- Windows operating system (optimized for Windows)

### Dependencies
The application requires the following Python packages:
- `customtkinter` - Modern UI framework
- `opencv-python` - Image processing
- `PyMuPDF` - PDF handling
- `numpy` - Numerical operations
- `openai` - AI text correction
- `Pillow` - Image manipulation
- `surya-ocr` - OCR detection and recognition

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yamanoko/Lenaissance.git
   cd Lenaissance
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## Usage

### Basic OCR Workflow

1. **Select Input Files**
   - Click "Select the input file" to choose image or PDF files
   - Multiple files can be selected simultaneously
   - Supported formats: PNG, JPG, JPEG, PDF

2. **Configure Output Directory**
   - Click "Select the output directory" to choose where text files will be saved
   - Default: Current working directory

3. **Navigate Pages**
   - Use the dropdown menu to select specific pages
   - Use "Previous" and "Next" buttons for sequential navigation

4. **Extract Text**
   - Click and drag on the image to select the text region
   - Release the mouse button to start OCR processing
   - Right-click to cancel the current selection

5. **Process Images (Optional)**
   - Enable "Activate Binarization" for better OCR on low-contrast images
   - Adjust the threshold slider to optimize image processing
   - Use zoom controls for detailed inspection

### AI Text Correction

1. **Set up OpenAI API**
   - Click "Set API Key" and enter your OpenAI API key
   - The system will validate the key automatically

2. **Proofread Text**
   - After extracting text, click "Proofread Text"
   - Review the original and corrected text in the comparison window
   - Choose to apply or cancel the changes

3. **Batch Operations**
   - Use "Combine text" to merge all page texts into a single file
   - Individual page files remain available for reference

## Interface Overview

### Left Panel (Controls)
- File selection and output directory settings
- Page navigation controls
- Image processing options (binarization, zoom)
- Text combination and AI proofreading features
- OpenAI API configuration

### Right Panel (Workspace)
- **Canvas Area**: Interactive image viewer with selection capabilities
- **Text Display**: Real-time view of extracted text for the current page
- Scrollbars for navigation of large documents

## Technical Details

### OCR Technology
- **Surya Engine**: State-of-the-art open-source OCR toolkit supporting 90+ languages
- **Text Line Segmentation**: Advanced structural analysis for proper text organization

### AI Integration
- **GPT-3.5-turbo**: Advanced language model for text correction
- **Context-aware Correction**: Maintains original meaning while fixing errors
- **Secure API Handling**: API keys are handled securely and not stored

### Image Processing
- **OpenCV Backend**: Robust image processing capabilities
- **Real-time Preview**: Instant feedback on processing parameters
- **Memory Efficient**: Optimized for handling large documents

## File Structure

```
Lenaissance/
├── main.py              # Main application file
├── pyproject.toml       # Project configuration
├── README.md           # This file
├── Lensaissance.spec   # PyInstaller spec for executable creation
├── OCRApp.spec         # Alternative PyInstaller spec
└── build/              # Build artifacts (auto-generated)
```

## Building Executable

The project includes PyInstaller specification files for creating standalone executables, making the application accessible to users without Python installations:

```bash
pyinstaller Lensaissance.spec
```

The executable will be created in the `dist/` directory and can be distributed as a standalone application for Windows systems.

## Academic Context

This project was developed as part of:
- **Google Summer of Code 2024-2025**: Continuation of previous year's foundational work
- **RenAIssance Project**: Collaborative initiative for historical document digitization
- **Waseda University Research**: Academic contribution to digital humanities

## Author

**Yukinori Yamamoto**  
Junior Student, Waseda University, Japan  
Google Summer of Code Participant (2024, 2025)  
RenAIssance Project Contributor

---

**Lensaissance** - Bringing Renaissance-level clarity to your documents through modern AI and OCR technology.
