# RenAIssance: Historical Transformer OCR for Early Spanish Texts

<p align="center">
  <img src="figs/humanai.jpg" alt="HumanAI Foundation" style="height: 100px; margin-right: 20px;"/>
  <img src="figs/gsoc_logo.png" alt="Google Summer of Code" style="height: 56px;"/>
</p>

This project builds an end-to-end OCR workflow for historical Spanish documents using TrOCR, CRAFT-based text detection, document preprocessing, line segmentation, and CPU-friendly quantized deployment. It was developed as part of the HumanAI Foundation initiative during Google Summer of Code 2024 and 2025.

![](figs/app.gif)

## Table of Contents

- [Project Snapshot](#project-snapshot)
- [Why Historical OCR Is Hard](#why-historical-ocr-is-hard)
- [Pipeline Overview](#pipeline-overview)
- [Repository Map](#repository-map)
- [Quick Start](#quick-start)
- [Docker Workflow](#docker-workflow)
- [Local Installation](#local-installation)
- [Required Assets](#required-assets)
- [Interactive App Usage](#interactive-app-usage)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Inference](#inference)
  - [Quantization](#quantization)
  - [Synthetic Data Generation](#synthetic-data-generation)
- [Model Performance](#model-performance)
- [Practical Notes](#practical-notes)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Links](#links)

## Project Snapshot

RenAIssance is more than a model-training repository. It combines:

1. PDF preprocessing for historical scans
2. text detection with CRAFT
3. contour-based line segmentation
4. TrOCR fine-tuning and inference
5. ONNX export and quantized CPU inference
6. an interactive Streamlit application for page inspection and OCR

The repository includes training code, test-time OCR scripts, app interfaces, quantization utilities, data preparation helpers, synthetic data generation assets, and sample PDFs for experimentation.

<img src="figs/screenshot1.png" /><br><br>
<img src="figs/screenshot1.png" />

## Why Historical OCR Is Hard

OCR for seventeenth-century material is challenging because these documents often contain:

- irregular printing and degraded scans
- bleed-through and border artifacts
- skewed or uneven page layouts
- interchangeable historical glyphs such as `u/v` and `f/s`
- inconsistent spelling and line-break conventions

This project tackles those issues by combining image cleanup, document splitting, line detection, transformer-based recognition, and deployment-aware optimization instead of relying on a single OCR step.

## Pipeline Overview

### 1. Preprocessing

The app and utility scripts can:

- render PDFs into page images using PyMuPDF
- detect and split double-page scans into virtual left and right pages
- deskew pages
- remove borders
- apply thresholding and noise filtering

### 2. Text Detection

The project uses CRAFT to locate text regions before OCR. The app caches contour outputs and uses them to build line-level reading order.

### 3. Line Segmentation

Detected contours are converted into bounding boxes, filtered, optionally split if unusually tall, aligned around a common text column, and sorted top-to-bottom.

### 4. Text Recognition

Line crops are transcribed with Hugging Face TrOCR models through `TrOCRProcessor` and `VisionEncoderDecoderModel`.

### 5. Optimization and Deployment

The repository also supports ONNX export and CPU-focused quantized inference through `optimum` and `onnxruntime`, with a dedicated Streamlit deployment app in `code/app/qapp.py`.

## Repository Map

```text
RenAIssance_Transformer_OCR_Utsav_Rai/
|-- code/
|   |-- app/                         # Streamlit applications and app-side OCR logic
|   |-- CRAFT/                       # Text detection model code
|   |-- datautils/                   # PDF processing and contour-to-line utilities
|   |-- finetuning/                  # Extra finetuning notebooks and training experiments
|   |-- quantization/                # ONNX export, quantization, and comparison scripts
|   |-- synthetic_data_generation/   # Synthetic OCR training data generation tools
|   |-- config.yaml                  # Training and inference configuration
|   |-- train.py                     # TrOCR fine-tuning pipeline
|   |-- test.py                      # Folder-based OCR inference
|   `-- utils.py                     # Dataset, metrics, plotting, and helper functions
|-- data/
|   |-- test/                        # Example segmented page folders
|   |-- test_books/                  # Sample PDFs for app testing
|   `-- train/                       # Training PDFs, processed pages, line segments, transcriptions
|-- figs/                            # Logos and demo GIF
|-- models/                          # Place model files here
|-- weights/                         # Place CRAFT weight files here
|-- Dockerfile
|-- requirements.txt
|-- README.md
`-- readmeNEW.md
```

## Quick Start

If you already have the required model files, the fastest way to see the app is:

```bash
cd code/app
streamlit run app_streamlit.py
```

For the containerized quantized version:

```bash
docker pull utsavrai27/ocr-quantized
docker run -p 8502:8501 utsavrai27/ocr-quantized
```

Then open:

```text
http://localhost:8502
```

## Docker Workflow

The Docker image is built around the CPU-optimized quantized app.

### Run the published image

```bash
docker pull utsavrai27/ocr-quantized
docker run -p 8502:8501 utsavrai27/ocr-quantized
```

### Build locally

Before building locally, make sure the following exist:

- CRAFT weights in `weights/`
- TrOCR processor/model config files in `models/`
- quantized ONNX files in `quantized_model/`

Then run:

```bash
docker build -t renaissance-ocr .
docker run -p 8501:8501 renaissance-ocr
```

The container starts `code/app/qapp.py`, which uses the quantized ONNX recognizer with `CPUExecutionProvider`.

## Local Installation

Python `3.10` is a safe choice because the Dockerfile uses `python:3.10-slim`.

Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- `requirements.txt` does not install PyTorch directly.
- The Dockerfile installs CPU-only PyTorch separately.
- For local GPU training, install the PyTorch build that matches your CUDA environment before running training.

## Required Assets

Large model files are intentionally not committed to git. In the current tracked tree, `models/` and `weights/` only contain `.gitkeep`, so a fresh clone is not fully runnable until assets are added.

### Required downloads

| Component | Description | Destination |
|---|---|---|
| Training dataset | Line segments and transcriptions | `data/train/` |
| OCR model | Fine-tuned TrOCR model | `models/` |
| Quantized ONNX model | CPU-optimized deployment model | `quantized_model/` |
| CRAFT weights | Text detection and refinement weights | `weights/` |

### Reference download links

| Component | Link |
|---|---|
| Training dataset | [Google Drive](https://drive.google.com/drive/folders/1FX6H3IXh-GyeNFEN2SOBkQy4_m_cQ4DX?usp=sharing) |
| OCR model | [Google Drive](https://drive.google.com/drive/folders/1NMngL384GpGohOpwm3yxYaYJ_Oe_ikpv?usp=sharing) |
| Quantized ONNX model | [Google Drive](https://drive.google.com/drive/folders/1uDek2tO4AxSoSXWApRe5D7OkDQS4pGqI?usp=sharing) |
| `craft_mlt_25k.pth` | [Google Drive](https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view) |
| `craft_ic15_20k.pth` | [Google Drive](https://drive.google.com/file/d/1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf/view) |
| `craft_refiner_CTW1500.pth` | [Google Drive](https://drive.google.com/file/d/1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO/view) |

## Interactive App Usage

The repository contains two closely related Streamlit apps:

- `code/app/app_streamlit.py`: standard app using the local TrOCR model
- `code/app/qapp.py`: quantized CPU deployment app used by Docker

### App capabilities

- upload a PDF and navigate page by page
- split wide scanned images into virtual left and right pages
- adjust DPI, noise threshold, and see-through intensity
- toggle border removal and deskewing
- enable line segmentation
- adjust segmentation parameters such as padding, minimum width, margin, and threshold
- run OCR on the current page and review line-ordered text output

### App preview

The repository already includes `figs/app.gif` as the built-in visual demo of the interface and workflow.

### Screenshot note

The two screenshots shared in chat are not present as local image files inside this repository, so they cannot be embedded here automatically as static Markdown assets. Once those images are saved into `figs/`, they can be added directly under this section.

## Usage

### Data Preparation

If you are starting from PDFs rather than ready-made line crops, use the utilities in `code/datautils/`.

Main automation command:

```bash
cd code/datautils
python main.py
```

This stage covers:

- PDF-to-image conversion
- document cleanup
- text detection
- contour extraction
- line segmentation

Practical note:

- `code/datautils/main.py` uses example paths such as `book_name = "book2"`, so adjust paths and parameters before running it on your own data.

### Training

Training is configured through `code/config.yaml`.

Important configuration fields include:

- `image_dir`
- `text_dir`
- `model_dir`
- `inf_model_dir`
- `base_dir`
- `train_batch_size`
- `eval_batch_size`
- `num_train_epochs`
- `learning_rate`
- `use_wandb`
- `model_name`

To train:

```bash
cd code
python train.py
```

The training pipeline:

- loads TrOCR from Hugging Face
- builds a paired image-text dataset
- splits train and evaluation subsets
- trains with `Seq2SeqTrainer`
- computes CER, WER, and BLEU
- saves the best model and processor
- runs folder-based inference after training

### Inference

To run OCR on folders of segmented line images:

```bash
cd code
python test.py
```

Expected structure:

```text
data/test/
|-- 2/
|-- 220/
`-- 251/
```

Each page folder should contain numbered `.jpg` line segments. The script sorts them numerically, transcribes them in order, and writes `output.txt` into each page folder.

### Quantization

The repository includes ONNX conversion and evaluation scripts under `code/quantization/`.

Export and quantize:

```bash
cd code/quantization
python onnx_quat.py
```

Compare PyTorch and quantized models:

```bash
cd code/quantization
python evaluate_quat.py
```

`evaluate_quat.py` measures:

- inference time
- CER
- WER
- model output similarity
- storage size difference
- result visualizations

### Synthetic Data Generation

The folder `code/synthetic_data_generation/VRD-image-text-generator/` contains tools for generating OCR training pairs from text.

Example command:

```bash
cd code/synthetic_data_generation/VRD-image-text-generator
python auto_generation.py --input_file your_text_file.txt --font_size 24 --add_random_text True --apply_data_augmentation True
```

This is useful when expanding training data or experimenting with augmentation-heavy OCR pipelines.

## Model Performance

The repository reports the following metrics for the historical OCR model:

| Metric | Value | Accuracy |
|---|---:|---:|
| CER | 0.019 | 98.1% |
| WER | 0.048 | 95.2% |

BLEU = `0.92`

## Practical Notes

- Large assets are excluded from git, so setup is not zero-config after cloning.
- Some scripts use hard-coded example paths and should be reviewed before reuse.
- The local Streamlit app and the Dockerized quantized app use different model-loading paths.
- The Dockerfile expects specific TrOCR config/tokenizer files under `models/`.
- Sample PDFs are available in `data/test_books/` for interface testing.
- The repository also contains extra notebooks in `code/finetuning/` and sample processed data under `data/train/processed_book/`.

## Acknowledgements

This project is supported by the [HumanAI Foundation](https://humanai.foundation/) and Google Summer of Code 2024 and 2025.

Related write-ups:

- [2024 blog post 1](https://utsavrai.substack.com/p/a-journey-into-historical-text-recognition)
- [2024 blog post 2](https://utsavrai.substack.com/p/decoding-history-advancing-text-recognition)
- [2025 midterm blog](https://utsavrai.substack.com/p/efficient-transformer-based-ocr-for?r=3ypuho)
- [2025 final blog](https://open.substack.com/pub/utsavrai/p/containerised-quantised-transformer?utm_campaign=post-expanded-share&utm_medium=web)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Links

- [Google Summer of Code 2024 Project](https://summerofcode.withgoogle.com/programs/2024/projects/IxqaG5cU)
- [Google Summer of Code 2025 Project](https://summerofcode.withgoogle.com/programs/2025/projects/SWLdu59R)
- [HumanAI Foundation](https://humanai.foundation/)

