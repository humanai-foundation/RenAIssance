# Historical Text Recognition using TrOCR

<p align="center">
  <img src="figs/humanai.jpg" alt="Image 1" style="height: 100px; margin-right: 20px;"/>
  <img src="figs/gsoc_logo.png" alt="Image 2" style="height: 50px;" />
</p>

This project aims to develop a robust Optical Character Recognition (OCR) system for historical texts using the TrOCR model from Hugging Face's `transformers` library. This project is part of the HumanAI Foundation initiative and was developed during Google Summer of Code 2024.


![](figs/app.gif)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Inference](#inference)
- [Datasets and Models](#datasets-and-models)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Links](#links)

## Overview

In the digital age, preserving historical documents is paramount for cultural and academic research. Traditional OCR tools often struggle with the intricacies of aged manuscripts, especially those from the seventeenth century, which present unique challenges due to handwriting variability, font differences, and document degradation. This project, under the Google Summer of Code (GSoC) initiative, leverages transformer models to develop a state-of-the-art OCR system capable of accurately transcribing historical Spanish texts.

The primary objective is to create an advanced, transformer-based OCR model tailored for seventeenth-century Spanish documents, focusing on minimizing Character Error Rate (CER) and Word Error Rate (WER) to achieve high accuracy in text extraction.

### Key Challenges and Solutions

#### Printing Irregularities and Ambiguities
- **Interchangeable Characters**: Characters like 'u' and 'v', and 'f' and 's' were used interchangeably.
- **Tildes and Diacritical Marks**: Used to save space or due to type mold reuse.
- **Old Spellings and Modern Interpretations**: Differences in character usage between historical and modern Spanish.
- **Line End Hyphens**: Words split across lines might not always be hyphenated.

#### Data Preparation and Augmentation
- **PDF to Images**: Converting PDFs to high-resolution images and preprocessing them.
- **Enhancements**: Deskewing, noise removal, and augmentation techniques like rotation, perspective changes, and Gaussian noise addition.

#### Model Architecture
- **Vision Transformer (ViT) Encoder**: Processes images as sequences of fixed-size patches.
- **Text Transformer Decoder**: Generates text sequences from visual features, initialized with a pretrained BERT model.
- **Pretrained Models**: Uses pretrained models for both encoder and decoder, leveraging rich prior knowledge in visual and textual domains.

#### Training and Evaluation
- **Hyperparameter Optimization**: Selection through empirical experimentation and Bayesian optimization.
- **Model Calibration**: Utilizes margin loss and other techniques to align sequence likelihoods with quality, improving output accuracy.
- **Evaluation Metrics**: Performance evaluated using CER, WER, and BLEU scores.

For a detailed walkthrough of the project's development, challenges, and solutions, read the complete blog post [here](https://utsavrai.substack.com/p/a-journey-into-historical-text-recognition) & [here](https://utsavrai.substack.com/p/decoding-history-advancing-text-recognition).

## Installation

Ensure you have Python 3.x and the necessary packages installed. You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```
## App Usage
This tool is designed to make document digitization easier by allowing interactive control over processing and real-time feedback on the results.
This tool allows you to process PDF documents with advanced image processing techniques like line segmentation, deskewing, border removal, and OCR (Optical Character Recognition). Using the CRAFT model, the tool identifies text regions and provides OCR on the segmented lines. You can interactively adjust parameters and view the results, including bounding boxes around detected text.

### Features:
- **Line Segmentation**: Automatically detects text regions and draws bounding boxes around them.
- **Deskewing**: Corrects skewed text in the document for better recognition.
- **Border Removal**: Removes borders to clean up images.
- **OCR**: Extracts text from segmented regions using the TrOCR model.
- **Interactive Sliders**: Adjust parameters like padding, noise threshold, and text region width.
- **PDF Page Navigation**: Easily navigate between pages and save processed images.

### Running the Application
To start the application:
1. Navigate to the project directory:
   ```bash
   cd code/app
   ```
2. Run the application:
   ```bash
   python main.py
   ```
  
### How to Use:
1. **Select a PDF File**: Click the "Select PDF" button to open a PDF file.
2. **Adjust Processing Options**: Use the provided checkboxes and sliders to enable features like deskewing, border removal, and line segmentation. You can adjust parameters such as noise threshold, padding, and minimum width of the text region.
3. **Enable Line Segmentation**: Toggle the "Line Segmentation" option to activate text detection. This will run the CRAFT model to detect text regions and visualize bounding boxes.
4. **Run OCR**: After line segmentation, click "Run OCR" to extract text from the detected regions. The OCR output is displayed line-by-line.
5. **Navigate Pages**: Use the "Next PDF Page" and "Previous PDF Page" buttons to move through the document.
6. **Save Processed Images**: You can save the left or right page as an image with detected bounding boxes.

This tool is designed to make document digitization easier by allowing interactive control over processing and real-time feedback on the results.
## Usage

### Data Preparation
1. **Automated Processing**:
   ```bash
   python main.py
   ```
   This script automates the entire process of converting PDFs to images, running text detection, and splitting text into line segments. More information can be found in the `code/datautils/` folder.

### Training

To fine-tune the TrOCR model, use the `train.py` script. More information can be found in the `code/` folder.

```bash
python train.py
```

### Inference

To perform inference using the fine-tuned model, use the `test.py` script. More information can be found in the `code/` folder.

```bash
python test.py
```

## Datasets and Models

- Download the dataset containing two folders `All_line_segments` and `All_line_texts` and extract it into `data/train/` folder from [Google Drive](https://drive.google.com/drive/folders/1FX6H3IXh-GyeNFEN2SOBkQy4_m_cQ4DX?usp=drive_link).
- Download the fine-tuned model named as `printed_large` and extract it into the `weights/` folder from [Google Drive](https://drive.google.com/drive/folders/1NMngL384GpGohOpwm3yxYaYJ_Oe_ikpv?usp=drive_link).

## Model Performance

| Metric | Value | Accuracy |
|--------|-------|----------|
| CER    | 0.02498 | 97.5%   |
| WER    | 0.04777 | 95.223% |

## Acknowledgements

This project is supported by the [HumanAI Foundation](https://humanai.foundation/) and Google Summer of Code 2024. Detailed documentation and a journey of this project can be found in the [blog post 1](https://utsavrai.substack.com/p/a-journey-into-historical-text-recognition) & [blag post 2](https://utsavrai.substack.com/p/decoding-history-advancing-text-recognition).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Links

- [Google Summer of Code 2024 Project](https://summerofcode.withgoogle.com/programs/2024/projects/IxqaG5cU)
- [HumanAI Foundation](https://humanai.foundation/)

Feel free to fork the repository and submit pull requests. For major changes, please open an issue to discuss your ideas first. Contributions are always welcome!