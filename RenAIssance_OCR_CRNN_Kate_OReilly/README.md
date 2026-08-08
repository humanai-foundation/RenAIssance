# RenAIssance OCR  
**Character-Level Recognition for Historical Spanish Documents**

This project builds a robust Optical Character Recognition (OCR) pipeline for early modern Spanish texts using a CRNN-based architecture. It integrates modern text detection models (CRAFT, DBNet, PSENet) with a custom-trained CRNN for character recognition. The aim is to make Renaissance-era Spanish literature accessible and machine-readable.

---

## Table of Contents

- [Overview](#-overview)  
- [Installation](#-installation)  
  - [Data Preparation](#-data-preparation)  
  - [Training](#-training)  
  - [Evaluation](#-evaluation)  
- [Architecture](#-architecture)  
- [Challenges](#-challenges)  
- [Acknowledgements](#-acknowledgements)  
- [License](#-license)  

---

## Overview

Traditional OCR systems struggle with historical documents due to:

- Font inconsistencies  
- Page degradation  
- Outdated orthography  

This pipeline solves the above through:

- **CRAFT** for bounding box detection  
- **CRNN + CTC Loss** for character-level text recognition  
- **Beam search + lexicon post-correction**  
- **Custom augmentation** for improving generalization  

The focus is on early modern Spanish prints from the 16th–17th centuries.

---

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Install submodule-specific dependencies:

```bash
pip install -r CRAFTModel/requirements.txt


### Training

Train the CRNN with your dataset:

Supports:

- Early stopping  
- Data augmentation  
- Lexicon-based decoding  

---

### Evaluation

Evaluate on validation or test sets:
Generates:

- CER / WER scores  
- Misclassified visualizations  
- Beam search + lexicon predictions  

---

## Architecture

- **Text Detection:** CRAFT 
- **Text Recognition:** CRNN with CTC Loss  
- **Decoder:** Greedy + Beam Search + Fuzzy Lexicon  
- **Augmentation:** Rotation, perspective warp, occlusion, noise  

---

## Challenges

- Box sorting on skewed or curved layouts  
- OCR drift from historical glyphs and degraded inputs  
- Beam search tuning for lexicon sensitivity and runtime  
- Inference failures due to model confidence collapse  

---

##  Acknowledgements

- [CRAFT](https://github.com/clovaai/CRAFT-pytorch)  
- [MMOCR](https://github.com/open-mmlab/mmocr)  
- [PSENet](https://github.com/whai362/PSENet)  
- [DBNet](https://github.com/MhLiao/DB)  

Developed under the **RenAIssance** research initiative.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file.
