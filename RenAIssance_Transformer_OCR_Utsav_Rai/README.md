# transformer-ocr
A custom transformer based OCR


---

# Python Scripts for Document Processing

This collection of Python scripts is designed to facilitate the processing of document images, starting from PDF format conversion, segmenting pages into line segments, and preparing the final dataset for further analysis or machine learning tasks.

### 1. `preprocess_pdf.py`

**Purpose**: Converts PDF documents into high-resolution images suitable for text recognition tasks.

**Requirements**:
- Python libraries: `PIL`, `pdf2image`
- Ensure that poppler is installed on your system.

**Usage**:
```bash
python preprocess_pdf.py --input_dir="path/to/pdf_directory" --output_dir="path/to/image_directory"
```

**Description**:
This script takes PDF files from a specified directory and converts each page into a separate image file, saving the results in the specified output directory.

### 2. `segment_page_to_line_segment.py`

**Purpose**: Segments page images into individual line segments, preparing them for line-level text recognition.

**Requirements**:
- Python libraries: `cv2`, `numpy`, `PIL`
- Tesseract OCR installed on your system.

**Usage**:
```bash
python segment_page_to_line_segment.py --input_dir="path/to/page_images" --output_dir="path/to/line_segments"
```

**Description**:
Given a directory of page images, this script uses OpenCV and PyTesseract to identify and extract individual lines of text as separate image segments. The segments are saved in the specified output directory, organized by their source page.

### 3. `finaldata_processing.py`

**Purpose**: Processes the line segments to generate a dataset ready for training or analysis, including any necessary preprocessing steps.

**Requirements**:
- Python libraries: `PIL`, `torch`, `transformers`

**Usage**:
```bash
python finaldata_processing.py --input_dir="path/to/line_segments" --output_dir="path/to/final_dataset"
```

**Description**:
This script further processes the line segment images, applying any required preprocessing steps (e.g., normalization, resizing) and organizing the data into a structured format suitable for use in machine learning models or analytical tasks.

### General Notes

- Ensure all dependencies are installed using `pip install -r requirements.txt`.
- Modify the script parameters in the "Usage" sections according to your specific directories and requirements.
- These scripts are part of a workflow designed for processing document images for text recognition tasks. They are best used in the sequence presented here.

---
