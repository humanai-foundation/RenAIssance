# Project for HumanAI Foundation: Self-Supervised Learning for Text Recognition in Early Modern Spanish Documents
It features a model designed using self-supervised learning methods to recognize text in early modern Spanish documents.

## Architecture

The model was implemented with reference to SeqCLR (Aberdam, A., et al., [2021](https://arxiv.org/abs/2012.10873)). According to the paper, SeqCLR employs a Contrastive Learning method, wherein its encoder learns to become robust against certain image transformations. Although the original paper proposes this method for sub-word level text detection, I hypothesized that it could be effectively adapted for line-level detection by optimizing the encoder's output (i.e., frame sequences) to represent each character within a line accurately.

I chose SeqCLR as the architecture for this test because it is a pioneering self-supervised learning model for OCR. It is relatively straightforward to understand and serves as an excellent starting point for this GSoC project, which aims to build a self-supervised model with an accuracy of above 80%.

The architecture includes a combination of ResNet34 and a 2-layer BiLSTM as the Encoder, and an Attention LSTM Decoder.

## Dataset

The model was trained using the Rodrigo Database(Nicolas Serrano et al., [2010](https://aclanthology.org/L10-1330/)), a "specific dataset" provided by HumanAI for this test, and an automatically generated dataset containing 35K text line images. Bounding boxes were manually annotated on images in the specific dataset. The Rodrigo Database consists of historical documents written in Spanish, similar to the background of the specific dataset. However, it primarily contains handwritten texts, unlike the printed texts in the specific dataset. This discrepancy in text types between datasets might impede the model's learning efficiency.

To address this issue, I postulated that a comparable amount of printed text data to the Rodrigo Database would be necessary. However, due to the scarcity of datasets of printed Spanish documents online, I could not find a suitable dataset for this purpose. Consequently, I generated the necessary dataset using the Text Recognition Data Generator module (TRDG)[https://textrecognitiondatagenerator.readthedocs.io/en/latest/index.html].

## Training

The model's training process is as follows:

1. **Contrastive Learning**: First, the Encoder was trained using the Contrastive Learning method on 20K generated text line images. This process involved extracting essential features from the data by contrasting text images with their transformed copies, including Vertical Crop, Gaussian Blur, Random Perspective, and Random Affine transformations.

2. **Semi-supervised Training Loop**: Next, the Decoder was trained using the teacher-forcing method to predict characters in text line images. This training utilized 15K generated images, 15K images from the Rodrigo Database, and approximately 500 images from the specific dataset. The Character Error Rate (CER) was used to evaluate the model, as it is deemed the most appropriate metric for a Decoder predicting characters.

## Result

After 8 epochs of Contrastive Learning and 20 epochs of semi-supervised learning, the model achieved a CER of under 0.1% for the Validation dataset. However, the performance on the test dataset was inadequate, as evidenced in the final tests conducted in `decoder_training.ipynb`. Despite generating some parts of text lines correctly, the model's overall performance was lacking. I anticipate that I could address these issues through my GSoC project by exploring several solutions, such as changing the architecture (e.g., replacing the Decoder with a Transformer Decoder, making the Encoder deeper), increasing the dataset size, and optimizing the data transformation process in Contrastive Learning, pending acceptance by HumanAI.

## Output of Test in `decoder_training.ipynb`

![スクリーンショット 2024-03-27 224608](https://github.com/yamanoko/SeqCLR_RenaissanceSpanish/assets/81514427/d47d973f-e774-4b75-9ef8-9ce84b02c939)
![スクリーンショット 2024-03-27 224623](https://github.com/yamanoko/SeqCLR_RenaissanceSpanish/assets/81514427/54e8fa71-eeb5-486d-98a2-00dbbab97e7a)
![スクリーンショット 2024-03-27 224646](https://github.com/yamanoko/SeqCLR_RenaissanceSpanish/assets/81514427/f2c21b5c-9ba7-489f-85f8-81e6bed0643c)
