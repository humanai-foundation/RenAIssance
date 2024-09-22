# Spanish Historical OCR using Self-Supervised Learning

## Overview
This repository implements a word-level OCR model for Renaissance Spanish documents using Self-Supervised Learning. The model was developed with reference to SeqCLR ([Aberdam A., et al., 2021](https://arxiv.org/abs/2012.10873)). According to the paper, SeqCLR employs a Contrastive Learning method, wherein its encoder learns to become robust against certain image transformations. The architecture includes a combination of ResNet50(or ViT tiny) and a 2-layer BiLSTM as the Encoder, and an Attention LSTM Decoder. At this point, the model achieves approximately 4% CER. This model can be tested in `test_model.ipynb`. For further information, please refer to my [blog](https://medium.com/@yamanko1234/historical-ocr-with-self-supervised-learning-c4f00da6637f).

## File/Folder Descriptions
- **Tokenizer**: A folder containing Tokenizer pickle files for the Decoder training.
- **test_image**: A folder containing images used for testing.
- **Decoder.py**: Implementation of the SeqCLR’s Decoder.
- **ResNet.py**: Implementation of ResNet, a component of the Encoder.
- **config.json**: A JSON file that sets the configuration for training.
- **custom_dataset.py**: Implementation of a custom dataset used in training.
- **decoder_training.ipynb**: A notebook to train the Decoder.
- **encoder.py**: Implementation of the SeqCLR’s Encoder.
- **ViT_encoder.py** Implementation of ViT version Encoder.
- **encoder_training.ipynb**: A notebook to train the Encoder.
- **test_model.ipynb**: A notebook to test a saved model.

## Testing the Model
First, you need to install the dependencies:
```
pip install -r requirements.txt
```
Then, you can test the saved model by executing the cells in `test_model.ipynb` one by one.