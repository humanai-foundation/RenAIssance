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

image_dir: 'dataset/All_line_segments'
text_dir: 'dataset/All_line_texts'
model_dir: 'model/trocr_finetuned'
train_batch_size: 4
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
wandb_key: "your_wandb_key"
use_wandb: True
model_name: "microsoft/trocr-large-handwritten"
test_data_dir: "dataset/test"
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