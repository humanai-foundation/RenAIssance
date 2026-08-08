# Spanish Historical OCR using Self-Supervised Learning

## Overview
This repository implements a word-level OCR model for Renaissance Spanish documents using self-supervised learning. The model was developed with reference to SeqCLR ([Aberdam A., et al., 2021](https://arxiv.org/abs/2012.10873)). According to the paper, SeqCLR uses contrastive learning so its encoder becomes robust to image transformations. The architecture combines a ResNet50 (or ViT tiny) and a 2-layer BiLSTM encoder with an attention LSTM decoder.

At this point, the model achieves approximately 4% CER. This model can be tested in `test_model.ipynb`. For more background, see the [project blog post](https://medium.com/@yamanko1234/historical-ocr-with-self-supervised-learning-c4f00da6637f).

## Portable Configuration
The default `config.json` now uses paths relative to this folder instead of machine-specific absolute paths. That makes the project easier to clone and configure on another machine.

Populate the directories below with your local datasets and checkpoints, or update `config.json` to match your own layout:

```text
RenAIssance_SelfSupervisedLearning_OCR_YukinoriYamamoto/
├── config.json
├── data/
│   ├── ssl/
│   │   └── word_images/
│   └── finetuning/
│       ├── perfecto/
│       │   ├── word_images/
│       │   └── word_images.csv
│       ├── ezcaray/
│       │   ├── word_images/
│       │   └── word_images.csv
│       └── virtuosa/
│           ├── word_images/
│           └── word_images.csv
├── models/
└── test_images/
```

The bundled `test_images/` folder is used as the default `test dataset` path so contributors can validate notebook setup without first changing that entry.

Before running the notebooks, you can verify the configured paths:

```bash
python check_config_paths.py
```

## File/Folder Descriptions
- **Tokenizer**: Pickle files used for decoder training and decoding.
- **data**: Local SSL and fine-tuning datasets referenced by `config.json`.
- **models**: Saved encoder and decoder checkpoints.
- **test_images**: Sample images used for testing.
- **Decoder.py**: SeqCLR decoder implementation.
- **ResNet.py**: ResNet implementation used by the encoder.
- **config.json**: Training and inference configuration.
- **check_config_paths.py**: Helper script that verifies configured dataset and model paths exist.
- **custom_dataset.py**: Custom dataset implementations used in training.
- **decoder_training.ipynb**: Notebook for decoder training and evaluation.
- **encoder.py**: SeqCLR encoder implementation.
- **ViT encoder support**: The notebooks include an optional ViT encoder path controlled by `config.json`.
- **encoder_training.ipynb**: Notebook for encoder training.
- **test_model.ipynb**: Notebook for testing a saved model.

## Testing the Model
Install the dependencies:

```bash
pip install -r requirements.txt
```

Confirm `config.json` points to valid paths for your environment:

```bash
python check_config_paths.py
```

Then run the cells in `test_model.ipynb`.
