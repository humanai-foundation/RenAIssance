import os
import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
    RobertaTokenizerFast,
    ViTImageProcessor
)
from torch.utils.data.dataset import random_split
from transformers import get_cosine_schedule_with_warmup

from torch.optim import AdamW
import matplotlib.pyplot as plt
import evaluate
import nltk
import re
import sys
import logging
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import json
import platform
import torch.cuda as cuda
# At the beginning of your script, add this import:
import Levenshtein

# Set up logging
def setup_logging(model_dir):
    # Create logs directory
    logs_dir = os.path.join(model_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure file handler with timestamp in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("trocr-training")
    
    # Create TensorBoard writer
    tb_dir = os.path.join(model_dir, "tensorboard", timestamp)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    
    # Log system information
    logger.info(f"Starting new training run at {timestamp}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"System: {platform.system()} {platform.release()}")
    
    # Log GPU information
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.info("No GPU available, using CPU")
    
    return logger, writer

# Then modify your install_dependencies function to ensure it's properly installed:
def install_dependencies():
    logger = logging.getLogger("trocr-training")
    logger.info("Checking and installing dependencies...")
    
    try:
        import Levenshtein
        logger.info("Levenshtein already installed")
    except ImportError:
        logger.warning("Levenshtein not found, installing python-Levenshtein...")
        os.system("pip install python-Levenshtein")
        # Try importing again to confirm installation
        try:
            import Levenshtein
            logger.info("Successfully installed python-Levenshtein")
        except ImportError:
            logger.warning("Failed to import Levenshtein module even after installation.")
            logger.warning("Trying alternative package...")
            os.system("pip install Levenshtein")
            try:
                import Levenshtein
                logger.info("Successfully installed Levenshtein")
            except ImportError:
                logger.error("Critical error: Could not import Levenshtein package.")
    
    try:
        import jiwer
        logger.info("jiwer already installed")
    except ImportError:
        logger.warning("jiwer not found, installing...")
        os.system("pip install jiwer")
        logger.info("Successfully installed jiwer")
    
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK punkt tokenizer already installed")
    except LookupError:
        logger.warning("NLTK punkt tokenizer not found, downloading...")
        nltk.download('punkt')
        logger.info("Successfully downloaded NLTK punkt tokenizer")

# Define the dataset class
class SpanishDocumentsDataset(Dataset):
    def __init__(self, image_dir, text_dir, image_processor, tokenizer):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        
        # Support multiple image formats
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        self.filenames = []
        
        # Check if directory exists
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory not found: {image_dir}")
        
        # Get all image files with supported extensions
        for ext in self.image_extensions:
            self.filenames.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
        
        # Log the distribution of file formats
        logger = logging.getLogger("trocr-training")
        format_counts = {}
        for ext in self.image_extensions:
            count = len([f for f in self.filenames if f.lower().endswith(ext)])
            if count > 0:
                format_counts[ext] = count
        
        if format_counts:
            logger.info(f"Image format distribution: {format_counts}")
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        image_file = self.filenames[idx]
        # Get file extension
        file_ext = os.path.splitext(image_file)[1].lower()
        
        # Derive the corresponding text file name by replacing the extension with .txt
        text_file = os.path.splitext(image_file)[0] + '.txt'
        
        image_path = os.path.join(self.image_dir, image_file)
        text_path = os.path.join(self.text_dir, text_file)
        
        # Check if text file exists
        if not os.path.exists(text_path):
            # Try alternative text filename patterns
            alt_text_path = os.path.join(self.text_dir, image_file.replace(file_ext, '.txt'))
            if os.path.exists(alt_text_path):
                text_path = alt_text_path
            else:
                raise FileNotFoundError(f"No text file found for image {image_file}. Tried {text_path} and {alt_text_path}")
        
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze()

        with open(text_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()
        labels = self.tokenizer(text, return_tensors="pt").input_ids.squeeze()
        
        # Treat padding specially for label calculation, if necessary
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {"pixel_values": pixel_values, "labels": labels}

# Collate function for data loader
def collate_fn(batch):
    pixel_values = [item['pixel_values'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Padding value for labels should be -100 to ignore tokens during loss calculation
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    pixel_values = torch.stack(pixel_values)
    
    return {"pixel_values": pixel_values, "labels": labels}

# Custom metrics implementation to avoid dependency issues
def compute_cer(predictions, references):
    total_cer = 0.0
    total_chars = 0
    
    for pred, ref in zip(predictions, references):
        # Count character errors
        errors = Levenshtein.distance(pred, ref)
        total_chars += len(ref)
        total_cer += errors
    
    if total_chars == 0:
        return 0.0
    
    return total_cer / total_chars

def compute_wer(predictions, references):
    total_wer = 0.0
    total_words = 0
    
    for pred, ref in zip(predictions, references):
        # Split into words
        pred_words = pred.split()
        ref_words = ref.split()
        
        # Count word errors using Levenshtein distance on word lists
        errors = Levenshtein.distance(pred_words, ref_words)
        total_words += len(ref_words)
        total_wer += errors
    
    if total_words == 0:
        return 0.0
    
    return total_wer / total_words

def compute_bleu(predictions, references):
    # Simple BLEU implementation (less accurate than proper BLEU)
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoothie = SmoothingFunction().method1
        
        total_bleu = 0.0
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]
            
            # Compute BLEU score
            if len(pred_tokens) > 0 and len(ref_tokens[0]) > 0:
                score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
                total_bleu += score
        
        return total_bleu / len(predictions) if len(predictions) > 0 else 0.0
    except:
        # Fallback if NLTK not available
        return 0.0

# Add this function to your script
def perform_comprehensive_evaluation(model, dataset, processor, tokenizer, device, logger, stage_name, output_dir):
    """Perform comprehensive evaluation on a given dataset and save results."""
    logger.info(f"Performing comprehensive evaluation after {stage_name} training...")
    
    # Create a DataLoader for evaluation
    eval_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    all_preds = []
    all_labels = []
    cer_values = []
    wer_values = []
    bleu_values = []
    
    # Iterate through batches
    with torch.no_grad():
        for batch in eval_loader:
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Generate predictions
            generated_ids = model.generate(
                pixel_values,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode predictions and references
            decoded_preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            decoded_labels = []
            for label in labels:
                label_filtered = [token for token in label.cpu().numpy() if token != -100]
                decoded_label = tokenizer.decode(label_filtered, skip_special_tokens=True)
                decoded_labels.append(decoded_label)
            
            # Calculate metrics for this batch
            batch_cer = compute_cer(decoded_preds, decoded_labels)
            batch_wer = compute_wer(decoded_preds, decoded_labels)
            batch_bleu = compute_bleu(decoded_preds, decoded_labels)
            
            # Store batch values
            cer_values.append(batch_cer)
            wer_values.append(batch_wer)
            bleu_values.append(batch_bleu)
            
            # Store predictions and labels for later analysis
            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_labels)
    
    # Calculate overall metrics
    avg_cer = sum(cer_values) / len(cer_values) if cer_values else 0
    avg_wer = sum(wer_values) / len(wer_values) if wer_values else 0
    avg_bleu = sum(bleu_values) / len(bleu_values) if bleu_values else 0
    
    # Log results
    logger.info(f"=== {stage_name} Evaluation Results ===")
    logger.info(f"CER: {avg_cer:.4f}")
    logger.info(f"WER: {avg_wer:.4f}")
    logger.info(f"BLEU: {avg_bleu:.4f}")
    
    # Save detailed evaluation results
    results_dir = os.path.join(output_dir, "evaluation", stage_name.lower())
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics
    metrics = {
        "cer": avg_cer,
        "wer": avg_wer,
        "bleu": avg_bleu,
        "num_samples": len(all_preds)
    }
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save sample predictions
    num_examples = min(50, len(all_preds))
    sample_results = []
    
    for i in range(num_examples):
        sample = {
            "reference": all_labels[i],
            "prediction": all_preds[i],
            "cer": compute_cer([all_preds[i]], [all_labels[i]]),
            "wer": compute_wer([all_preds[i]], [all_labels[i]])
        }
        sample_results.append(sample)
    
    with open(os.path.join(results_dir, "sample_predictions.json"), "w") as f:
        json.dump(sample_results, f, indent=2)
    
    # Create a markdown report
    md_report = f"# Evaluation Report: {stage_name} Training\n\n"
    md_report += f"## Overall Metrics\n\n"
    md_report += f"- Character Error Rate (CER): {avg_cer:.4f}\n"
    md_report += f"- Word Error Rate (WER): {avg_wer:.4f}\n"
    md_report += f"- BLEU Score: {avg_bleu:.4f}\n\n"
    md_report += f"## Sample Predictions\n\n"
    
    for i, sample in enumerate(sample_results[:10]):  # Show first 10 in the report
        md_report += f"### Sample {i+1}\n\n"
        md_report += f"**Reference:** {sample['reference']}\n\n"
        md_report += f"**Prediction:** {sample['prediction']}\n\n"
        md_report += f"**CER:** {sample['cer']:.4f}, **WER:** {sample['wer']:.4f}\n\n"
        md_report += "---\n\n"
    
    with open(os.path.join(results_dir, "evaluation_report.md"), "w") as f:
        f.write(md_report)
    
    logger.info(f"Evaluation results saved to {results_dir}")
    
    return metrics

def save_model_custom(model, tokenizer, image_processor, output_dir, logger):
    """Custom model saving function that avoids the DTensor import error"""
    # Create output directories
    model_dir = os.path.join(output_dir, "model")
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    processor_dir = os.path.join(output_dir, "image_processor")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)
    os.makedirs(processor_dir, exist_ok=True)
    
    # Save model state dict and config separately
    logger.info("Saving model configuration...")
    model.config.save_pretrained(model_dir)
    
    logger.info("Saving model weights...")
    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
    
    # Save tokenizer and processor
    logger.info("Saving tokenizer...")
    tokenizer.save_pretrained(tokenizer_dir)
    
    logger.info("Saving image processor...")
    image_processor.save_pretrained(processor_dir)
    
    logger.info(f"Model saved to {output_dir}")

# Function to plot and save metrics
def plot_and_save_metrics(logs, metric_name, output_dir, logger):
    logger.info(f"Plotting {metric_name} metrics...")
    train_steps = []
    train_values = []
    eval_steps = []
    eval_values = []

    for log in logs:
        if 'loss' in log and metric_name == 'loss':
            if 'epoch' in log:
                train_steps.append(log['epoch'])
                train_values.append(log['loss'])
        elif f'eval_{metric_name}' in log:
            if 'epoch' in log:
                eval_steps.append(log['epoch'])
                eval_values.append(log[f'eval_{metric_name}'])

    plt.figure(figsize=(10, 6))
    if train_steps and train_values:
        plt.plot(train_steps, train_values, label=f'Training {metric_name.capitalize()}')
    if eval_steps and eval_values:
        plt.plot(eval_steps, eval_values, label=f'Evaluation {metric_name.capitalize()}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'Training and Evaluation {metric_name.capitalize()}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, f'{metric_name}_plot.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved {metric_name} plot to {plot_path}")

# Function to perform inference on test data
def perform_inference(model, image_processor, tokenizer, base_dir, output_dir, device, logger):
    logger.info(f"Starting inference on test data from {base_dir}")
    # Create a processor for inference
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    
    # Function to generate text for a single image segment
    def generate_text_from_image_segment(image_path):
        start_time = time.time()
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        inference_time = time.time() - start_time
        logger.debug(f"Inference for {image_path}: {inference_time:.2f}s")
        return generated_text

    # Sort function for filenames
    def sort_key(filename):
        match = re.search(r"(\d+)\.jpg", filename)
        if match:
            return int(match.group(1))
        return -1

    inference_results = []
    
    # Check if test directory exists
    if not os.path.exists(base_dir):
        logger.warning(f"Test directory {base_dir} does not exist. Skipping inference.")
        return []
    
    # Iterate through each page's folder
    for page_folder in sorted(os.listdir(base_dir)):
        page_path = os.path.join(base_dir, page_folder)
        if os.path.isdir(page_path):
            logger.info(f"Processing folder: {page_folder}")
            page_output = f"Processing {page_folder}:\n"
            page_texts = []

            # Sort the line segment images numerically based on the segment number
            line_segment_images = sorted([f for f in os.listdir(page_path) if f.endswith('.jpg')], key=sort_key)
            logger.info(f"Found {len(line_segment_images)} image segments")

            # Iterate through each sorted line segment in the page folder
            for line_segment_image in line_segment_images:
                line_segment_path = os.path.join(page_path, line_segment_image)
                try:
                    line_text = generate_text_from_image_segment(line_segment_path)
                    page_texts.append((line_segment_image, line_text))
                    logger.debug(f"Generated text for {line_segment_image}: {line_text}")
                except Exception as e:
                    logger.error(f"Error processing {line_segment_image}: {str(e)}")
            
            # Sort the generated texts based on the filenames
            page_texts.sort(key=lambda x: sort_key(x[0]))

            # Print the texts in sorted order
            for line_segment_image, line_text in page_texts:
                page_output += f"  {line_segment_image}: {line_text}\n"

            # Compile and display the full page's text
            full_page_text = "\n".join([text for _, text in page_texts])
            page_output += f"\nFull text for {page_folder}:\n{full_page_text}\n\n{'='*50}\n"
            
            inference_results.append(page_output)
    
    # Save inference results to file
    results_path = os.path.join(output_dir, 'inference_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(inference_results))
    logger.info(f"Saved inference results to {results_path}")
    
    return inference_results

# Define a custom trainer class to add logging to TensorBoard
class LoggingTrainer(Seq2SeqTrainer):
    def __init__(self, tb_writer=None, logger=None, **kwargs):
        super().__init__(**kwargs)
        self.tb_writer = tb_writer
        self.logger = logger
        self.best_metrics = {"eval_loss": float("inf"), "eval_cer": float("inf"), "eval_wer": float("inf"), "eval_bleu": 0}
        
    def log(self, logs, start_time=None):
        # Call the original log method
        super().log(logs, start_time)
        
        if self.tb_writer:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, self.state.global_step)
            
            # Log learning rate
            if self.optimizer and len(self.optimizer.param_groups) > 0:
                self.tb_writer.add_scalar(
                    "learning_rate", 
                    self.optimizer.param_groups[0]["lr"], 
                    self.state.global_step
                )
        
        # Log to file with more detail
        if self.logger:
            if "eval_loss" in logs:
                self.logger.info(f"Step {self.state.global_step}: Evaluation results:")
                for key, value in logs.items():
                    if key.startswith("eval_"):
                        self.logger.info(f"  {key}: {value:.5f}")
                        
                        # Track best metrics
                        if key == "eval_loss" and value < self.best_metrics["eval_loss"]:
                            self.best_metrics["eval_loss"] = value
                            self.logger.info(f"  New best {key}: {value:.5f}")
                        elif key == "eval_cer" and value < self.best_metrics["eval_cer"]:
                            self.best_metrics["eval_cer"] = value
                            self.logger.info(f"  New best {key}: {value:.5f}")
                        elif key == "eval_wer" and value < self.best_metrics["eval_wer"]:
                            self.best_metrics["eval_wer"] = value
                            self.logger.info(f"  New best {key}: {value:.5f}")
                        elif key == "eval_bleu" and value > self.best_metrics["eval_bleu"]:
                            self.best_metrics["eval_bleu"] = value
                            self.logger.info(f"  New best {key}: {value:.5f}")
            else:
                # For training logs, only log occasionally to avoid overwhelming the log file
                if self.state.global_step % 50 == 0:
                    self.logger.info(f"Step {self.state.global_step}: " + 
                                   ", ".join([f"{k}: {v:.5f}" for k, v in logs.items() if isinstance(v, (int, float))]))
    def _save(self, output_dir):
        # Call the original save method
        output = super()._save(output_dir)
        
        # Log the save event
        if self.logger:
            self.logger.info(f"Saved model checkpoint to {output_dir}")
            
            # Save current metrics alongside the model
            metrics_path = os.path.join(output_dir, "metrics.json")
            current_metrics = {k: v for k, v in self.state.log_history[-1].items() if k.startswith("eval_")}
            with open(metrics_path, "w") as f:
                json.dump(current_metrics, f, indent=2)
            self.logger.info(f"Saved current metrics to {metrics_path}")
        
        return output

def main():
    # Set paths for both datasets
    real_image_dir = '../../../data/train/All_line_segments'
    real_text_dir = '../../../data/train/All_line_texts'
    synthetic_image_dir = '../../../data/train/All_synth_line_segments'
    synthetic_text_dir = '../../../data/train/All_synth_line_texts'

    # Create base model directory
    base_model_dir = "../../../models"
    os.makedirs(base_model_dir, exist_ok=True)
    
    # Create a timestamped run directory with sequential training identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_model_dir, f"sequential_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories for different artifacts
    model_dir = os.path.join(run_dir, "model")
    plots_dir = os.path.join(run_dir, "plots")
    logs_dir = os.path.join(run_dir, "logs")
    
    # Create subdirectories for stage-specific models
    synthetic_model_dir = os.path.join(model_dir, "synthetic")
    real_model_dir = os.path.join(model_dir, "real")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(synthetic_model_dir, exist_ok=True)
    os.makedirs(real_model_dir, exist_ok=True)
    
    # Set up logging
    logger, tb_writer = setup_logging(logs_dir)
    
    # Log start of training script
    logger.info("=" * 50)
    logger.info("STARTING SEQUENTIAL TRAINING SCRIPT")
    logger.info("=" * 50)
    logger.info(f"Run directory: {run_dir}")
    
    # Install dependencies first
    install_dependencies()
    
    # Check GPU availability
    logger.info("Checking GPU availability:")
    os.system("nvidia-smi")
    
    # Create plots directory
    plots_dir = os.path.join(model_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Try a different model and tokenizer to avoid dependency issues
    logger.info("Loading components with fallback options...")
    
    try:
        # Try loading the default model first
        model_name = "microsoft/trocr-large-handwritten"
        
        logger.info(f"Loading image processor from {model_name}...")
        start_time = time.time()
        image_processor = ViTImageProcessor.from_pretrained(model_name)
        logger.info(f"Image processor loaded in {time.time() - start_time:.2f}s")
        
        logger.info("Loading tokenizer (RobertaTokenizerFast)...")
        start_time = time.time()
        # Use RobertaTokenizerFast instead of XLMRobertaTokenizer to avoid SentencePiece dependency
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        logger.info(f"Tokenizer loaded in {time.time() - start_time:.2f}s")
        
        logger.info(f"Loading model from {model_name}...")
        start_time = time.time()
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
        logger.info(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
        
    except Exception as e:
        logger.error(f"Error with primary model: {e}")
        logger.info("Trying fallback model...")
        
        # Fallback to a simpler model
        model_name = "qantev/trocr-large-spanish"
        
        logger.info("Loading image processor...")
        image_processor = ViTImageProcessor.from_pretrained(model_name)
        
        logger.info("Loading tokenizer (RobertaTokenizerFast)...")
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        
        logger.info("Loading model...")
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # Create processor for training utilities
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    
    # Configure model
    logger.info("Configuring model...")
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.ignore_pad_token_for_loss = True
    
    # Log model configuration
    logger.info(f"Model configuration: {model.config}")
    
    # Define compute_metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]

        predictions = logits.argmax(-1)
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)

        decoded_labels = []
        for label in labels:
            label_filtered = [token for token in label if token != -100]
            decoded_label = tokenizer.decode(label_filtered, skip_special_tokens=True)
            decoded_labels.append(decoded_label)

        # Use custom metric functions
        cer = compute_cer(decoded_preds, decoded_labels)
        wer = compute_wer(decoded_preds, decoded_labels)
        bleu = compute_bleu(decoded_preds, decoded_labels)
        
        # Log example predictions during evaluation
        if len(decoded_preds) > 0:
            num_examples = min(3, len(decoded_preds))
            logger.info(f"Example predictions ({num_examples}):")
            for i in range(num_examples):
                logger.info(f"  True: {decoded_labels[i]}")
                logger.info(f"  Pred: {decoded_preds[i]}")
                logger.info(f"  CER: {compute_cer([decoded_preds[i]], [decoded_labels[i]]):.4f}")
                logger.info("")

        return {"cer": cer, "wer": wer, "bleu": bleu}
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Initialize both datasets early to avoid reference errors
    logger.info("Initializing all datasets upfront...")
    
    # Prepare synthetic dataset
    if not os.path.exists(synthetic_image_dir) or not os.path.exists(synthetic_text_dir):
        logger.error(f"Synthetic data directories not found: {synthetic_image_dir} or {synthetic_text_dir}")
        logger.error("Please ensure synthetic data is available before running this script.")
        sys.exit(1)
    
    logger.info(f"Preparing synthetic dataset from {synthetic_image_dir} and {synthetic_text_dir}...")
    start_time = time.time()
    synthetic_dataset = SpanishDocumentsDataset(synthetic_image_dir, synthetic_text_dir, image_processor, tokenizer)
    logger.info(f"Synthetic dataset prepared in {time.time() - start_time:.2f}s")
    
    # Split synthetic dataset
    synthetic_dataset_size = len(synthetic_dataset)
    synthetic_eval_size = int(synthetic_dataset_size * 0.1)  # 10% for evaluation
    synthetic_train_size = synthetic_dataset_size - synthetic_eval_size
    
    logger.info(f"Synthetic dataset size: {synthetic_dataset_size} samples")
    logger.info(f"Synthetic training set: {synthetic_train_size} samples")
    logger.info(f"Synthetic evaluation set: {synthetic_eval_size} samples")
    
    # Set seed for reproducibility before splitting
    torch.manual_seed(42)
    synthetic_train_dataset, synthetic_eval_dataset = random_split(
        synthetic_dataset, [synthetic_train_size, synthetic_eval_size]
    )
    
    # Prepare real dataset
    logger.info(f"Preparing real dataset from {real_image_dir} and {real_text_dir}...")
    start_time = time.time()
    real_dataset = SpanishDocumentsDataset(real_image_dir, real_text_dir, image_processor, tokenizer)
    logger.info(f"Real dataset prepared in {time.time() - start_time:.2f}s")
    
    # Split real dataset
    real_dataset_size = len(real_dataset)
    real_eval_size = int(real_dataset_size * 0.1)  # 10% for evaluation
    real_train_size = real_dataset_size - real_eval_size
    
    logger.info(f"Real dataset size: {real_dataset_size} samples")
    logger.info(f"Real training set: {real_train_size} samples")
    logger.info(f"Real evaluation set: {real_eval_size} samples")
    
    # Set seed for reproducibility before splitting
    torch.manual_seed(42)
    real_train_dataset, real_eval_dataset = random_split(
        real_dataset, [real_train_size, real_eval_size]
    )
    
    #################################
    # STAGE 1: Train on Synthetic Data
    #################################
    logger.info("=" * 50)
    logger.info("STAGE 1: TRAINING ON SYNTHETIC DATA")
    logger.info("=" * 50)
    
    # Count synthetic files
    synthetic_files = [f for f in os.listdir(synthetic_image_dir) if any(f.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]
    logger.info(f"Found {len(synthetic_files)} synthetic images for training")
    
    # Set up training arguments for synthetic data
    logger.info("Setting up training arguments for synthetic data...")
    synthetic_training_args = Seq2SeqTrainingArguments(
        output_dir=synthetic_model_dir,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=20,
        fp16=False,
        gradient_accumulation_steps=1,
        num_train_epochs=20,  # Fewer epochs for synthetic data
        max_grad_norm=1.0,
        logging_dir=os.path.join(synthetic_model_dir, 'logs'),
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=100000,
        warmup_steps=100,
        learning_rate=2e-5,  # Higher learning rate for synthetic data
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none",
    )
    
    # Log synthetic training arguments
    logger.info(f"Synthetic training arguments: {synthetic_training_args}")
    
    # Prepare optimizer and scheduler for synthetic training
    synthetic_total_train_steps = (len(synthetic_train_dataset) // synthetic_training_args.per_device_train_batch_size) * synthetic_training_args.num_train_epochs
    logger.info(f"Total synthetic training steps: {synthetic_total_train_steps}")
    
    logger.info("Setting up optimizer and scheduler for synthetic training...")
    synthetic_optimizer = AdamW(model.parameters(), lr=synthetic_training_args.learning_rate)
    synthetic_scheduler = get_cosine_schedule_with_warmup(
        synthetic_optimizer,
        num_warmup_steps=synthetic_training_args.warmup_steps,
        num_training_steps=synthetic_total_train_steps
    )
    
    # Create early stopping callback for synthetic training
    synthetic_early_stopping = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.0
    )
    
    # Create trainer for synthetic data
    logger.info("Setting up trainer for synthetic data...")
    synthetic_trainer = LoggingTrainer(
        model=model,
        args=synthetic_training_args,
        train_dataset=synthetic_train_dataset,
        eval_dataset=synthetic_eval_dataset,
        optimizers=(synthetic_optimizer, synthetic_scheduler),
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[synthetic_early_stopping],
        tb_writer=tb_writer,
        logger=logger
    )
    
    # Train on synthetic data
    logger.info("Starting synthetic data training...")
    start_time = time.time()
    synthetic_trainer.train()
    synthetic_training_duration = time.time() - start_time
    
    # Log synthetic training time
    hours, remainder = divmod(synthetic_training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Synthetic training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Save model after synthetic training
    logger.info("Saving model after synthetic training...")
    save_model_custom(model, tokenizer, image_processor, synthetic_model_dir, logger)
    
    # Perform comprehensive evaluation after synthetic training
    synthetic_eval_metrics = perform_comprehensive_evaluation(
        model, 
        synthetic_eval_dataset,
        processor,
        tokenizer,
        device,
        logger, 
        "Synthetic", 
        run_dir
    )

    # Also evaluate on real data to see how well synthetic training transfers
    real_transfer_metrics = perform_comprehensive_evaluation(
        model, 
        real_eval_dataset,
        processor,
        tokenizer,
        device,
        logger, 
        "SyntheticToReal", 
        run_dir
    )
    
    # Plot and save synthetic training metrics
    logger.info("Plotting synthetic training metrics...")
    synthetic_logs = synthetic_trainer.state.log_history
    synthetic_plots_dir = os.path.join(plots_dir, "synthetic")
    os.makedirs(synthetic_plots_dir, exist_ok=True)
    
    # Save synthetic training logs to file
    synthetic_logs_path = os.path.join(logs_dir, "synthetic_training_logs.json")
    with open(synthetic_logs_path, "w") as f:
        json.dump(synthetic_logs, f, indent=2)
    logger.info(f"Saved synthetic training logs to {synthetic_logs_path}")
    
    for metric in ['loss', 'cer', 'wer', 'bleu']:
        plot_and_save_metrics(synthetic_logs, metric, synthetic_plots_dir, logger)
    
    #################################
    # STAGE 2: Finetune on Real Data
    #################################
    logger.info("=" * 50)
    logger.info("STAGE 2: FINETUNING ON REAL DATA")
    logger.info("=" * 50)
    
    # Set up training arguments for real data finetuning
    logger.info("Setting up training arguments for real data finetuning...")
    real_training_args = Seq2SeqTrainingArguments(
        output_dir=real_model_dir,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=20,
        fp16=False,
        gradient_accumulation_steps=1,
        num_train_epochs=50,  # More epochs for real data
        max_grad_norm=1.0,
        logging_dir=os.path.join(real_model_dir, 'logs'),
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=100,  # More frequent evaluation
        save_steps=100000,
        warmup_steps=50,
        learning_rate=2e-5,  # Lower learning rate for real data fine-tuning
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none",
    )
    
    # Log real training arguments
    logger.info(f"Real training arguments: {real_training_args}")
    
    # Prepare optimizer and scheduler for real data finetuning
    real_total_train_steps = (len(real_train_dataset) // real_training_args.per_device_train_batch_size) * real_training_args.num_train_epochs
    logger.info(f"Total real training steps: {real_total_train_steps}")
    
    logger.info("Setting up optimizer and scheduler for real data finetuning...")
    real_optimizer = AdamW(model.parameters(), lr=real_training_args.learning_rate)
    real_scheduler = get_cosine_schedule_with_warmup(
        real_optimizer,
        num_warmup_steps=real_training_args.warmup_steps,
        num_training_steps=real_total_train_steps
    )
    
    # Create early stopping callback for real data finetuning
    real_early_stopping = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.0
    )
    
    # Create trainer for real data
    logger.info("Setting up trainer for real data...")
    real_trainer = LoggingTrainer(
        model=model,  # Using the model already trained on synthetic data
        args=real_training_args,
        train_dataset=real_train_dataset,
        eval_dataset=real_eval_dataset,
        optimizers=(real_optimizer, real_scheduler),
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[real_early_stopping],
        tb_writer=tb_writer,
        logger=logger
    )
    
    # Train on real data
    logger.info("Starting real data finetuning...")
    start_time = time.time()
    real_trainer.train()
    real_training_duration = time.time() - start_time
    
    # Log real training time
    hours, remainder = divmod(real_training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Real data finetuning completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Save final model after real data finetuning
    logger.info("Saving final model after real data finetuning...")
    save_model_custom(model, tokenizer, image_processor, real_model_dir, logger)
    

    # Perform comprehensive evaluation after real data finetuning
    real_eval_metrics = perform_comprehensive_evaluation(
        model, 
        real_eval_dataset,
        processor,
        tokenizer,
        device,
        logger, 
        "Real", 
        run_dir
    )

    # Also evaluate on synthetic data to check if model still performs well
    synthetic_retention_metrics = perform_comprehensive_evaluation(
        model, 
        synthetic_eval_dataset,
        processor,
        tokenizer,
        device,
        logger, 
        "RealToSynthetic", 
        run_dir
    )
    
    # Plot and save real training metrics
    logger.info("Plotting real training metrics...")
    real_logs = real_trainer.state.log_history
    real_plots_dir = os.path.join(plots_dir, "real")
    os.makedirs(real_plots_dir, exist_ok=True)
    
    # Save real training logs to file
    real_logs_path = os.path.join(logs_dir, "real_training_logs.json")
    with open(real_logs_path, "w") as f:
        json.dump(real_logs, f, indent=2)
    logger.info(f"Saved real training logs to {real_logs_path}")
    
    for metric in ['loss', 'cer', 'wer', 'bleu']:
        plot_and_save_metrics(real_logs, metric, real_plots_dir, logger)
    
    # Save a copy of the final model to the main model directory
    logger.info("Saving final model to main model directory...")
    save_model_custom(model, tokenizer, image_processor, model_dir, logger)
    
    # Perform inference on test data
    logger.info("Performing inference on test data with final model...")
    test_dir = "../../../data/test"
    inference_results = perform_inference(
        model, image_processor, tokenizer, test_dir, run_dir, device, logger
    )
    
    # Create a run summary file with key information
    total_training_duration = synthetic_training_duration + real_training_duration
    hours, remainder = divmod(total_training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    summary = {
        "timestamp": timestamp,
        "training_approach": "sequential_finetuning",
        "synthetic_training_duration_seconds": synthetic_training_duration,
        "real_training_duration_seconds": real_training_duration,
        "total_training_duration_seconds": total_training_duration,
        "total_training_duration_formatted": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
        "synthetic_dataset_size": synthetic_dataset_size,
        "synthetic_train_size": synthetic_train_size,
        "synthetic_eval_size": synthetic_eval_size,
        "real_dataset_size": real_dataset_size,
        "real_train_size": real_train_size,
        "real_eval_size": real_eval_size,
        "synthetic_learning_rate": synthetic_training_args.learning_rate,
        "real_learning_rate": real_training_args.learning_rate,
        "synthetic_epochs": synthetic_training_args.num_train_epochs,
        "real_epochs": real_training_args.num_train_epochs,
        "synthetic_best_metrics": synthetic_trainer.best_metrics,
        "real_best_metrics": real_trainer.best_metrics,
        "model_size_params": sum(p.numel() for p in model.parameters()),
        "evaluation": {
            "after_synthetic_training": {
                "on_synthetic_data": synthetic_eval_metrics,
                "on_real_data": real_transfer_metrics
            },
            "after_real_training": {
                "on_real_data": real_eval_metrics,
                "on_synthetic_data": synthetic_retention_metrics
            }
        }
    }
    
    summary_path = os.path.join(run_dir, "sequential_run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved run summary to {summary_path}")
    
    # Log completion
    logger.info("=" * 50)
    logger.info("SEQUENTIAL TRAINING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"All artifacts saved to {run_dir}")
    logger.info(f"Synthetic model saved to {synthetic_model_dir}")
    logger.info(f"Real model saved to {real_model_dir}")
    logger.info(f"Final model saved to {model_dir}")
    logger.info(f"Plots saved to {plots_dir}")
    logger.info(f"Logs saved to {logs_dir}")
    logger.info(f"Inference results saved to {os.path.join(run_dir, 'inference_results.txt')}")
    logger.info(f"TensorBoard logs saved to {tb_writer.log_dir}")
    logger.info("To view TensorBoard logs, run: tensorboard --logdir=" + os.path.dirname(tb_writer.log_dir))
    
    # Close TensorBoard writer
    tb_writer.close()

if __name__ == "__main__":
    main()