# Import the existing training code
import os
import sys
import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import evaluate
import nltk
import re
import logging
import time
import json
import platform
import argparse
from datetime import datetime
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
import Levenshtein

# Import the EMA implementation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ema import ExponentialMovingAverage

# Use the existing helper functions and classes from train.py
# (Setup_logging, SpanishDocumentsDataset, compute metrics, etc.)
def setup_logging(model_dir):
    # Create logs directory
    logs_dir = os.path.join(model_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure file handler with timestamp in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"training_ema_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("trocr-training-ema")
    
    # Create TensorBoard writer
    tb_dir = os.path.join(model_dir, "tensorboard", f"ema_{timestamp}")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    
    # Log system information
    logger.info(f"Starting new EMA training run at {timestamp}")
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

def install_dependencies():
    logger = logging.getLogger("trocr-training-ema")
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
        self.filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        image_file = self.filenames[idx]
        # Derive the corresponding text file name by changing the extension
        text_file = image_file.replace('.jpg', '.txt')
        
        image_path = os.path.join(self.image_dir, image_file)
        text_path = os.path.join(self.text_dir, text_file)
        
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

# Custom metrics implementation
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
    plt.title(f'Training and Evaluation {metric_name.capitalize()} (with EMA)')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, f'{metric_name}_plot_ema.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved {metric_name} plot to {plot_path}")

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
    results_path = os.path.join(output_dir, 'inference_results_ema.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(inference_results))
    logger.info(f"Saved inference results to {results_path}")
    
    return inference_results

# Define the EMA-enabled trainer class
class EMATrainer(Seq2SeqTrainer):
    def __init__(self, tb_writer=None, logger=None, ema_decay=0.9999, **kwargs):
        super().__init__(**kwargs)
        self.tb_writer = tb_writer
        self.logger = logger
        self.best_metrics = {"eval_loss": float("inf"), "eval_cer": float("inf"), "eval_wer": float("inf"), "eval_bleu": 0}
        
        # Initialize EMA
        self.ema = None
        self.ema_decay = ema_decay
        if ema_decay > 0:
            self.logger.info(f"Initializing EMA with decay {ema_decay}")
            self.ema = ExponentialMovingAverage(self.model, decay=ema_decay)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to update EMA after each step"""
        # Regular training step
        outputs = super().training_step(model, inputs, num_items_in_batch)
        
        # Update EMA
        if self.ema is not None:
            self.ema.update()
            
        return outputs
    
    def evaluate(self, *args, **kwargs):
        """Override evaluate to use EMA weights during evaluation"""
        if self.ema is not None:
            self.logger.info("Using EMA weights for evaluation")
            self.ema.copy_to(self.model)
            
        # Regular evaluation
        output = super().evaluate(*args, **kwargs)
        
        # Restore original weights
        if self.ema is not None:
            self.ema.restore(self.model)
            
        return output
    
    def _save(self, output_dir):
        """Override _save to save both regular and EMA weights"""
        # Save regular model
        output = super()._save(output_dir)
        
        # Save EMA model if available
        if self.ema is not None:
            # Create EMA directory
            ema_dir = os.path.join(output_dir, "ema")
            os.makedirs(ema_dir, exist_ok=True)
            
            # Apply EMA weights
            self.ema.copy_to(self.model)
            
            # Save EMA model
            self.model.save_pretrained(ema_dir)
            
            # Restore original weights
            self.ema.restore(self.model)
            
            if self.logger:
                self.logger.info(f"Saved EMA model to {ema_dir}")
        
        return output
    
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

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR model with EMA")
    parser.add_argument("--ema_decay", type=float, default=0.9999, 
                        help="Decay rate for Exponential Moving Average (0-1, higher is slower decay)")
    parser.add_argument("--model_name", type=str, default="microsoft/trocr-large-printed",
                        help="Base model to use for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of training epochs")
    parser.add_argument("--model_output_dir", type=str, 
                        default="../../../models/printed_large_ema",
                        help="Directory to save the fine-tuned model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set paths
    image_dir = '../../../data/train/All_line_segments'
    text_dir = '../../../data/train/All_line_texts'
    model_dir = args.model_output_dir
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up logging
    logger, tb_writer = setup_logging(model_dir)
    
    # Log start of training script
    logger.info("=" * 50)
    logger.info("STARTING EMA TRAINING SCRIPT")
    logger.info("=" * 50)
    
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
        model_name = args.model_name
        
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
        model_name = "microsoft/trocr-large-printed"
        
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
    
    # Prepare dataset and split
    logger.info(f"Preparing dataset from {image_dir} and {text_dir}...")
    start_time = time.time()
    dataset = SpanishDocumentsDataset(image_dir, text_dir, image_processor, tokenizer)
    logger.info(f"Dataset prepared in {time.time() - start_time:.2f}s")
    
    dataset_size = len(dataset)
    eval_size = int(dataset_size * 0.1)  # 10% for evaluation
    train_size = dataset_size - eval_size
    
    logger.info(f"Dataset size: {dataset_size} samples")
    logger.info(f"Training set: {train_size} samples")
    logger.info(f"Evaluation set: {eval_size} samples")
    
    # Set seed for reproducibility before splitting
    torch.manual_seed(42)
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # Print dataloader information
    logger.info(f"Number of training batches: {len(train_loader)}")
    for batch in train_loader:
        logger.info(f"Sample batch shapes - Pixel values: {batch['pixel_values'].shape}, Labels: {batch['labels'].shape}")
        
        # Log sample image and text to TensorBoard
        sample_idx = 0
        sample_image = batch['pixel_values'][sample_idx].permute(1, 2, 0).cpu().numpy()
        sample_label = batch['labels'][sample_idx]
        sample_label_filtered = [token for token in sample_label if token != -100]
        sample_text = tokenizer.decode(sample_label_filtered, skip_special_tokens=True)
        
        tb_writer.add_image('sample_image', sample_image, 0, dataformats='HWC')
        tb_writer.add_text('sample_text', sample_text, 0)
        
        logger.info(f"Sample text: {sample_text}")
        break
    
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
    
    # Set up training arguments
    logger.info("Setting up training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=False,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        max_grad_norm=1.0,
        logging_dir=os.path.join(model_dir, 'logs'),
        logging_steps=10,
        eval_strategy="steps",  # This is the correct parameter name
        eval_steps=100,
        save_steps=100,
        warmup_steps=500,
        weight_decay=0.01,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",  # Disable HF reporting since we're using our own TB writer
    )
    
    # Log training arguments
    logger.info(f"Training arguments: {training_args}")
    logger.info(f"EMA decay: {args.ema_decay}")
    
    # Prepare optimizer and scheduler
    total_train_steps = (len(train_loader) * training_args.num_train_epochs)
    logger.info(f"Total training steps: {total_train_steps}")
    
    logger.info("Setting up optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=500, 
        num_training_steps=total_train_steps
    )
    
    # Create early stopping callback
    logger.info("Setting up early stopping...")
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3, 
        early_stopping_threshold=0.0
    )
    
    # Define trainer
    logger.info("Setting up EMA trainer...")
    trainer = EMATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, scheduler),
        data_collator=collate_fn,
        processing_class=processor,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
        tb_writer=tb_writer,
        logger=logger,
        ema_decay=args.ema_decay
    )
    
    # Train the model
    logger.info("Starting model training with EMA...")
    start_time = time.time()
    trainer.train()
    training_duration = time.time() - start_time
    
    # Log training time
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    logger.info("Saving model and processor components...")
    # Save the regular model
    save_model_custom(model, tokenizer, image_processor, model_dir, logger)
    
    # Plot and save metrics
    logger.info("Plotting metrics...")
    train_logs = trainer.state.log_history
    
    # Save training logs to file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_path = os.path.join(model_dir, f"training_logs_ema_{timestamp}.json")
    with open(logs_path, "w") as f:
        json.dump(train_logs, f, indent=2)
    logger.info(f"Saved training logs to {logs_path}")
    
    for metric in ['loss', 'cer', 'wer', 'bleu']:
        plot_and_save_metrics(train_logs, metric, plots_dir, logger)
    
    # Log best metrics achieved
    logger.info("Best metrics achieved:")
    for metric, value in trainer.best_metrics.items():
        logger.info(f"  {metric}: {value:.5f}")
    
    # Perform inference on test data, first with standard model
    logger.info("Performing inference on test data with standard model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    test_dir = "../../../data/test"

    # Now load and perform inference with the EMA model
    logger.info("Performing inference with EMA model...")
    ema_model_dir = os.path.join(model_dir, "ema")
    if os.path.exists(ema_model_dir):
        logger.info(f"Loading EMA model from {ema_model_dir}")
        ema_model = VisionEncoderDecoderModel.from_pretrained(ema_model_dir)
        ema_model.to(device)
        
        inference_results_ema = perform_inference(
            ema_model, image_processor, tokenizer, test_dir, 
            os.path.join(model_dir, "ema_inference"), device, logger
        )
    else:
        logger.warning("No EMA model directory found. Skipping EMA inference.")
    
    # Log completion
    logger.info(f"Training complete. Model saved to {model_dir}")
    logger.info(f"Plots saved to {plots_dir}")
    logger.info(f"TensorBoard logs saved to {tb_writer.log_dir}")
    logger.info("To view TensorBoard logs, run: tensorboard --logdir=" + os.path.dirname(tb_writer.log_dir))
    
    # Close TensorBoard writer
    tb_writer.close()

if __name__ == "__main__":
    main()