# Custom Data Preparation Utility Tools

## Overview

This README provides a comprehensive guide for processing a PDF book into individual images, running a text detection model on these images, and splitting detected text contours into line segments. The processing sequence involves three main steps, executed using Python scripts.

## Prerequisites

- Python 3.x
- Required Python packages: `fitz` (PyMuPDF), `PIL` (Pillow), `math`, `cv2` (OpenCV), `numpy`, `skimage`, `deskew`, `argparse`
- Trained models for text detection: `craft_mlt_25k.pth`, `craft_refiner_CTW1500.pth`

## File Descriptions

1. **`process_main_utils.py`**: Processes a PDF into individual images.
2. **`test.py`**: Runs a text detection model on the images.
3. **`contour_line_splitter.py`**: Splits detected text contours into line segments.
4. **`main.py`**: Automating first three files

## Steps to Achieve Results

### Step 1: Process PDF into Individual Images

This step involves converting a PDF book into individual processed images using `process_main_utils.py`.

**Usage**:
```bash
python process_main_utils.py <book_path> <output_dir> [--dpi DPI] [--remove_borders]
```

**Arguments**:
- `book_path`: Path to the PDF book.
- `output_dir`: Directory to save the processed images.
- `--dpi`: (Optional) DPI for rendering PDF pages (default: 300).
- `--remove_borders`: (Optional) Flag to remove borders from the processed images.

**Example**:
```bash
python process_main_utils.py "../../path/to/your/book.pdf" "path/to/processed/images" --dpi 300 --remove_borders
```

### Step 2: Run Text Detection Model

This step runs a text detection model on the processed images using `test.py`.

**Usage**:
```bash
python test.py --trained_model <trained_model> --text_threshold <text_threshold> --test_folder <test_folder> --refine --refiner_model <refiner_model>
```

**Arguments**:
- `--trained_model`: Path to the trained model file (e.g., `weights/craft_mlt_25k.pth`).
- `--text_threshold`: Text detection threshold (e.g., `0.9`).
- `--test_folder`: Folder containing the images to be processed.
- `--refine`: Flag to refine the detected text contours.
- `--refiner_model`: Path to the refiner model file (e.g., `weights/craft_refiner_CTW1500.pth`).

**Example**:
```bash
python test.py --trained_model weights/craft_mlt_25k.pth --text_threshold 0.9 --test_folder path/to/processed/images --refine --refiner_model weights/craft_refiner_CTW1500.pth
```

### Step 3: Split Detected Text Contours into Line Segments

This step processes the images and contours to draw bounding boxes and save the processed images using `contour_line_splitter.py`.

**Usage**:
```bash
python contour_line_splitter.py <image_dir> <contour_dir> <output_dir> [--padding PADDING] [--min_width MIN_WIDTH] [--threshold THRESHOLD] [--margin MARGIN] [--visualize]
```

**Arguments**:
- `image_dir`: Directory containing the input images.
- `contour_dir`: Directory containing the contour text files.
- `output_dir`: Directory to save the processed images.
- `--padding`: Padding around bounding boxes (default: 20).
- `--min_width`: Minimum width of bounding boxes (default: 100).
- `--threshold`: Threshold for splitting bounding boxes (default: 0.2).
- `--margin`: Margin to exclude short lines at the top and bottom (default: 0.1).
- `--visualize`: (Optional) Option to visualize bounding boxes on images.

**Example**:
```bash
python contour_line_splitter.py path/to/processed/images path/to/result path/to/line_segments --padding 100 --min_width 0 --threshold 0.6 --margin 0.0
```

## Automated Script: `main.py`

### Overview

The `main.py` script automates the entire process of converting a PDF book into individual processed images, running a text detection model on these images, and splitting the detected text contours into line segments. This script handles all necessary steps in sequence, ensuring that intermediate results are correctly passed from one stage to the next.

### What `main.py` Does

1. **Checks if the Output Directory is Empty**:
   - The script first checks whether the `output_dir` (the directory where processed images will be stored) is empty.
   - If the directory is empty, it proceeds to step 1 to process the PDF into individual images.
   - If the directory is not empty, it assumes that the images have already been processed and skips directly to steps 2 and 3.

2. **Step 1: Processing the PDF into Individual Images**:
   - The script runs `process_main_utils.py` to convert the PDF book into individual images.
   - It takes the path to the PDF (`book_path`) and the directory to save the processed images (`output_dir`).
   - Additional parameters like DPI (`--dpi`) and border removal (`--remove_borders`) are used to control the processing.

3. **Step 2: Running the Text Detection Model**:
   - The script executes `test.py` to run the text detection CRAFT model on the processed images.
   - It uses the images from `output_dir` as input and saves the results in the `result_folder`.
   - Parameters include the paths to the trained model (`--trained_model`), text detection threshold (`--text_threshold`), and refiner model (`--refiner_model`).

4. **Step 3: Splitting Detected Text Contours into Line Segments**:
   - Finally, the script runs `contour_line_splitter.py` to process the images and detected text contours.
   - It takes the images from `output_dir` and the text contour results from `result_folder`.
   - The script saves the processed line segments in `line_segments_folder`.
   - Additional parameters control the padding around bounding boxes (`--padding`), minimum width (`--min_width`), threshold for splitting bounding boxes (`--threshold`), and margin for excluding short lines (`--margin`).

### How to Run `main.py`

1. **Prepare Your Environment**:
   - Ensure you have all necessary dependencies installed and that your environment is set up correctly.

2. **Set the Parameters**:
   - Adjust the paths and parameters in the `main.py` script to match your specific setup and file locations.
   - Ensure that `book_path` correctly points to your PDF file and that `output_dir`, `result_folder`, and `line_segments_folder` are set to appropriate directories.

3. **Run the Script**:
   - Execute the `main.py` script from the command line:
     ```bash
     python main.py
     ```

### Summary of Actions in `main.py`:

- **Initialization**:
  - Creates necessary output directories if they do not exist.

- **Conditional Processing**:
  - Checks if the `output_dir` is empty.
  - If empty, processes the PDF into images.
  - If not empty, skips directly to running the text detection model and contour splitting.

- **Execution of Steps**:
  - Processes the PDF into images using `process_main_utils.py`.
  - Runs the text detection model using `test.py`.
  - Splits detected text contours into line segments using `contour_line_splitter.py`.

By following these instructions, you can efficiently process a PDF book through the entire workflow, ensuring that each step's output is correctly utilized in the subsequent steps.

### Conclusion

By following these steps and using the provided scripts, you can efficiently process a PDF book into individual images, detect text within these images, and split detected text contours into line segments. Adjust paths and parameters as necessary to fit your specific setup and file locations.