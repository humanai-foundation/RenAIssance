# Transformer OCR Fine-Tuning and Inference

This directory contains scripts for fine-tuning the TrOCR model for Optical Character Recognition (OCR) on custom datasets and performing inference using the fine-tuned model. The TrOCR model is a Transformer-based model from Hugging Face's `transformers` library.

## Table of Contents

- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Files Description](#files-description)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Configuration

All hyperparameters and configuration settings are stored in `config.yaml`. Edit this file to change paths, hyperparameters, and other settings.

```yaml
# config.yaml

image_dir: '../data/train/All_line_segments'
text_dir: '../data/train/All_line_texts'
model_dir: '../models/custom'
inf_model_dir: '../weights/printed_large'
base_dir: '../data/test'
train_batch_size: 4 # batch size of 4 requires 16 GB of GPU memory, this is per device batch size so for 2 GPU devices it will be 4 batch size each, in total 8
eval_batch_size: 4
fp16: False
gradient_accumulation_steps: 1
num_train_epochs: 5
max_grad_norm: 1.0
logging_dir: './logs'
logging_steps: 10
evaluation_strategy: "steps"
eval_steps: 200
save_steps: 100000
warmup_steps: 500
weight_decay: 0.01
save_total_limit: 1
load_best_model_at_end: True
learning_rate: 5e-6
early_stopping_patience: 3
early_stopping_threshold: 0.0
wandb_project: "trocr-ocr-finetuning"
wandb_key: "Your-wandb-key"
use_wandb: False # switch to true and replace the wandb api key above
model_name: "microsoft/trocr-large-printed"
# model_name: "microsoft/trocr-large-handwritten"

```

## Training

To fine-tune the TrOCR model, use `train.py`. This script will train the model on your dataset and save the fine-tuned model to the specified directory.

### Steps for Training:

1. Ensure your dataset is organized with images and corresponding text files.
2. Update `config.yaml` with the correct paths and settings.
3. Run the training script:
   ```bash
   python train.py
   ```

## Inference

**Note**: Make sure the `inf_model_dir` in `config.yaml` is pointing to the newly finetuned model in models directory or the provided model in weights directory.

To perform inference using the fine-tuned model, use `test.py`. This script will generate text from images in the test dataset and save the results.

### Steps for Inference:

1. Ensure the fine-tuned model is saved in the directory specified in `config.yaml`.
2. Update `config.yaml` with the correct paths and settings for inference.
3. Run the inference script:
   ```bash
   python test.py
   ```

## Files Description

### `config.yaml`

This file contains all the configuration settings required for training and inference. It includes paths to datasets, model directories, hyperparameters for training, and settings for logging and evaluation.

### `train.py`

This script is used for fine-tuning the TrOCR model. It:
- Loads configuration settings.
- Initializes the processor and model.
- Prepares the dataset and data loaders.
- Sets up training arguments, optimizer, and scheduler.
- Trains the model and saves the best-performing model.
- Logs training progress and plots metrics.
- Optionally uses Weights & Biases (wandb) for experiment tracking.

### `test.py`

This script is used for performing inference with the fine-tuned model. It:
- Loads the configuration settings.
- Initializes the processor and model.
- Iterates through the test dataset to generate text from images.
- Saves the generated text to output files.

### `utils.py`

This file contains utility functions and classes, including:
- `SpanishDocumentsDataset`: A custom dataset class for loading image-text pairs.
- `collate_fn`: A function for collating batches of data.
- `compute_metrics`: A function for computing evaluation metrics (CER, WER, BLEU).
- `plot_metrics`: A function for plotting training and evaluation metrics.
- `generate_text_from_image_segment`: A function for generating text from a single image segment.
- `sort_key`: A function for sorting filenames based on numeric values.

### `datautils`

Please refer to rthe eadme provided in this directory

### `CRAFT`

Text detection model by [CRAFT: Character-Region Awareness For Text detection](https://github.com/clovaai/CRAFT-pytorch)

### `finetuning`

This directory contains several examples of finetuning trocr model and epirical findings relevant to achieve higher accuracy.  
## Usage

### Training the Model

1. Make sure your dataset is organized with images and corresponding text files.
2. Update `config.yaml` with the correct paths and settings.
3. Run the training script:
   ```bash
   python train.py
   ```

### Running Inference

1. Ensure the fine-tuned model is saved in the directory specified in `config.yaml`.
2. Update `config.yaml` with the correct paths and settings for inference.
3. Run the inference script:
   ```bash
   python test.py
   ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
