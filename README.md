![thumbnail](https://github.com/user-attachments/assets/b0aa865c-416c-4a3a-92be-56a1a77c8f4e)

# RenAIssance

> AI-powered OCR for 16th–17th century printed Spanish documents — a [HumanAI Foundation](https://humanai.foundation/) GSoC project.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![GSoC](https://img.shields.io/badge/GSoC-2024%2F2025-orange)

The analysis of historical documents is a critical yet costly method in the Humanities. To reduce these costs, AI technology — specifically OCR (Optical Character Recognition) — has started to be utilized. However, for many years there was a lack of accurate OCR tools for Spanish documents from the Renaissance period, despite their academic importance. To address this, the HumanAI Foundation launched the **RenAIssance** project, where contributors implement accurate OCR models using various approaches. All models achieve **>90% character accuracy** on the target corpus.

---

## Table of Contents

- [Contributors & Approaches](#contributors--approaches)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [CRNN (Shashank)](#1-crnn-shashank-shekhar-singh)
  - [Self-Supervised Learning (Yukinori)](#2-self-supervised-learning-yukinori-yamamoto)
  - [Synthetic Image Generation (Saarthak)](#3-synthetic-image-generation-saarthak-gupta)
  - [Transformer OCR (Arsh)](#4-transformer-ocr-arsh-khan)
  - [Transformer OCR + App (Utsav)](#5-transformer-ocr--app-utsav-rai)
- [Docker](#docker)
- [Windows Setup & Troubleshooting](#windows-setup--troubleshooting)
- [Contributing](#contributing)

---

## Contributors & Approaches

| Contributor | GSoC Year | Approach | Architecture | CER | Folder |
|---|---|---|---|---|---|
| Shashank Shekhar Singh | 2024 | CRNN | CNN → BiLSTM → CTC | 0.027 (95.8% acc) | [→](RenAIssance_CRNN_OCR_Shashank_Shekhar_Singh/) |
| Yukinori Yamamoto | 2024 | Self-Supervised (SeqCLR) | ResNet50 + BiLSTM + Attention Decoder | ~0.04 (96% acc) | [→](RenAIssance_SelfSupervisedLearning_OCR_YukinoriYamamoto/) |
| Saarthak Gupta | 2025 | Synthetic Data Gen (Pix2Pix GAN) | U-Net Generator + PatchGAN Discriminator | — | [→](RenAIssance_SyntheticImageGeneration_Saarthak_Gupta/) |
| Arsh Khan | 2024 | Transformer (TrOCR + Calibration) | ViT Encoder + BERT-init Decoder + SLiC | 0.03 (97% acc) | [→](RenAIssance_Transformer_OCR_Arsh_Khan/) |
| Utsav Rai | 2024/2025 | Transformer + ONNX + Streamlit App | TrOCR-large + CRAFT seg + ONNX quant | 0.019 (98.1% acc) | [→](RenAIssance_Transformer_OCR_Utsav_Rai/) |

---

## Dataset

![letters](https://github.com/user-attachments/assets/c10584db-8f68-4897-a6c4-c70411ed9515)

The dataset consists of scanned pages of printed documents from 16th–17th century Spain, partially labeled by expert mentors. The following corpora are used across sub-projects:

| Corpus | Type | Used By |
|---|---|---|
| *Padilla — Nobleza Virtuosa* | Printed book (~31 pages) | Shashank, Arsh, Utsav |
| *Perfecto* | Printed book (word images) | Yukinori |
| *Ezcaray* | Printed book (word images) | Yukinori |
| *Paredes — Reglas Generales* | Printed document | Arsh, Utsav |
| *Porcones* | Legal printed documents | Arsh, Utsav |

**Download links:**

- Shashank's PDF + DOCX: linked in [CRNN Readme](RenAIssance_CRNN_OCR_Shashank_Shekhar_Singh/Readme.md)
- Saarthak's full data + CRAFT weights: [SharePoint](https://iitbhu365-my.sharepoint.com/:f:/g/personal/saarthak_gupta_mec22_iitbhu365_onmicrosoft_com/EtFan2TQidhNhTXXK45qTGwBAvyxOfpaJNxhSPWy16N0EA?e=fbdyuR)
- Utsav's training data, models, CRAFT weights: linked in [Utsav README](RenAIssance_Transformer_OCR_Utsav_Rai/README.md)

**Dataset challenges:**

- **Interchangeable characters:** `u`/`v` and `f`/`s` were used interchangeably
- **Tildes and diacritical marks:** used to save space or due to type-mold reuse
- **Old spellings:** variations between historical and modern Spanish
- **Line-end hyphens:** words split across lines were not always hyphenated
- **Document degradation:** fading ink, staining, and irregular layouts

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.8 (3.10 recommended) | 3.10 used in Docker image |
| PyTorch | ≥ 2.0 | Install separately — see [pytorch.org](https://pytorch.org/get-started/locally/) |
| CUDA Toolkit | 12.4 (optional) | Required only for GPU training; CPU inference works via ONNX |
| Git | any | For cloning the repo |
| Docker | any (optional) | Quickest path to run Utsav's Streamlit app |
| Tesseract OCR | ≥ 5.0 (optional) | Required only for Saarthak's data pipeline — [Windows installer](https://github.com/UB-Mannheim/tesseract/wiki) |

**Recommended Python environment:**

```bash
git clone https://github.com/HumanAI/RenAIssance.git
cd RenAIssance
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

Then follow the per-project install steps below.

---

## Quick Start

### 1. CRNN — Shashank Shekhar Singh

```bash
cd RenAIssance_CRNN_OCR_Shashank_Shekhar_Singh
# All dependencies are installed in the first cell of Model.ipynb
jupyter notebook Model.ipynb
```

The notebook is self-contained and can be run on Google Colab, Kaggle, or local Jupyter. To regenerate the training data from scratch (optional), run `Dataset_Generation.ipynb` first.

Pre-trained model: `Model/ocr_model.h5` — download link in [Readme.md](RenAIssance_CRNN_OCR_Shashank_Shekhar_Singh/Readme.md).

---

### 2. Self-Supervised Learning — Yukinori Yamamoto

```bash
cd RenAIssance_SelfSupervisedLearning_OCR_YukinoriYamamoto

# For GPU (CUDA 12.4):
pip install -r requirements.txt

# For CPU-only or different CUDA version, install PyTorch manually first:
# https://pytorch.org/get-started/locally/
pip install torch torchvision
pip install huggingface_hub==0.24.2 matplotlib numpy==1.24.4 opencv-python pandas Pillow torchmetrics
```

1. Edit `config.json` — update all dataset paths to point to your local data folders
2. Run `encoder_training.ipynb` (SSL pre-training)
3. Run `decoder_training.ipynb` (supervised fine-tuning)
4. Run `test_model.ipynb` (inference)

---

### 3. Synthetic Image Generation — Saarthak Gupta

```bash
cd RenAIssance_SyntheticImageGeneration_Saarthak_Gupta
pip install -r requirements.txt

# Windows: install Tesseract OCR binary and add to PATH
# https://github.com/UB-Mannheim/tesseract/wiki
```

1. Download data from the [SharePoint link](https://iitbhu365-my.sharepoint.com/:f:/g/personal/saarthak_gupta_mec22_iitbhu365_onmicrosoft_com/EtFan2TQidhNhTXXK45qTGwBAvyxOfpaJNxhSPWy16N0EA?e=fbdyuR) and place in `data/raw/`
2. Open and run `experimentation.ipynb` for the full pipeline

**Inference example:**

```python
from src.model_utils import SyntheticPageGenerator
spg = SyntheticPageGenerator()
page_img = spg.create_synthetic_page_from_text("los documentos del periodo renacentista")
page_img.save("outputs/page1.png")
```

---

### 4. Transformer OCR — Arsh Khan

```bash
cd RenAIssance_Transformer_OCR_Arsh_Khan
pip install -r requirements.txt

# Install PyTorch with your CUDA version:
# https://pytorch.org/get-started/locally/
```

Open and run `ViT_Transformer_Model.ipynb`. Utility notebooks for line segmentation and preprocessing are in the [`utils/`](RenAIssance_Transformer_OCR_Arsh_Khan/utils/) folder.

---

### 5. Transformer OCR + App — Utsav Rai

```bash
cd RenAIssance_Transformer_OCR_Utsav_Rai
pip install -r requirements.txt

# Install PyTorch (GPU or CPU) separately:
# https://pytorch.org/get-started/locally/
```

Download model weights and datasets from the links in [Utsav's README](RenAIssance_Transformer_OCR_Utsav_Rai/README.md), then:

```bash
# Training
python code/train.py

# Inference / evaluation
python code/test.py

# Streamlit web app
streamlit run code/app/app_streamlit.py
```

See [Docker](#docker) below for the easiest way to run the app without a local GPU.

---

## Docker

Only Utsav's sub-project ships a Docker image. It uses CPU-only PyTorch + ONNX Runtime, so **no GPU is required** to run the app.

```bash
# Pull the pre-built image
docker pull utsavrai27/ocr-quantized

# Run (maps internal port 8501 → host port 8502)
docker run -p 8502:8501 utsavrai27/ocr-quantized

# Open in browser
# http://localhost:8502
```

**Windows note:** Enable the WSL 2 backend in Docker Desktop settings for best performance.

To build locally, download the required model files first (see [Utsav's README](RenAIssance_Transformer_OCR_Utsav_Rai/README.md)) then run `docker build`.

---

## Windows Setup & Troubleshooting

| Issue | Affected Sub-project | Fix |
|---|---|---|
| `torch==2.4.1+cu124` install fails | SSL / Yukinori | No CUDA 12.4 toolkit installed. Install the matching toolkit from [nvidia.com](https://developer.nvidia.com/cuda-toolkit) or switch to a CPU wheel: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| Linux absolute paths in `config.json` | SSL / Yukinori | Open `config.json` and update all dataset paths (e.g. `/home/yukinori/…`) to your Windows paths (e.g. `C:\data\…`) |
| `ModuleNotFoundError: No module named 'fitz'` | CRNN / Shashank | Install PyMuPDF: `pip install PyMuPDF`. If using PyMuPDF ≥ 1.25, the import changed — use `import pymupdf as fitz` |
| `pytesseract.pytesseract.TesseractNotFoundError` | GAN / Saarthak | Install the [Tesseract Windows binary](https://github.com/UB-Mannheim/tesseract/wiki) and add its install directory to your system PATH |
| `RomanAntique` font not found | GAN / Saarthak | Copy the font from the `fonts/` folder into your Windows Fonts directory (`C:\Windows\Fonts`) or update the font path in `src/data_utils.py` |
| `datasets.load_metric` deprecation warning/error | Transformer / Utsav | Replace `from datasets import load_metric` with `import evaluate; metric = evaluate.load("cer")` — install with `pip install evaluate` |
| OpenCV `cv2.imshow` crashes | All OpenCV users | Install the [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) — or use headless OpenCV (`pip install opencv-python-headless`) if display is not needed |
| Docker container not accessible | Transformer / Utsav | Make sure you map ports with `-p 8502:8501` and open `http://localhost:8502` (not 8501) |

---

## Contributing

Contributions are welcome! If you find a bug, documentation gap, or would like to add a new OCR approach, please open an issue or submit a pull request.

Please follow the existing code style, include a `requirements.txt` in any new sub-project folder, and document your approach in a sub-project-level `README.md`.

This project is licensed under the **MIT License** — see the [`LICENSE`](RenAIssance_CRNN_OCR_Shashank_Shekhar_Singh/LICENSE) file for details.
