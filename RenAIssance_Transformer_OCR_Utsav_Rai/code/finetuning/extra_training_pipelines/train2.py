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
    ViTImageProcessor,
    TrainerCallback
)
from torch.utils.data.dataset import random_split
from torch.optim import AdamW
import matplotlib.pyplot as plt
import evaluate
import nltk
import re
import sys
from copy import deepcopy
# At the beginning of your script, add this import:
import Levenshtein

# EMA implementation for model parameters
class EMA:
    """
    Exponential Moving Average for model parameters
    Implementation adapted for PyTorch models
    """
    def __init__(self, model, decay=0.9999, update_every=1):
        """
        Initialize EMA with model parameters
        
        Args:
            model: PyTorch model
            decay: EMA decay rate (higher = slower moving average)
            update_every: Update EMA every N steps
        """
        self.model = model
        self.decay = decay
        self.update_every = update_every
        self.shadow = {}
        self.backup = {}
        self.step_counter = 0
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Initialize with model weights
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters"""
        self.step_counter += 1
        
        # Only update every update_every steps
        if self.step_counter % self.update_every != 0:
            return
        
        # Update EMA parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                
                # Update with decay rate
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model for inference/saving"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # Backup current parameters
                self.backup[name] = param.data.clone()
                # Replace with EMA parameters
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters to the model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    
    # Add a copy method to create a deep copy for checkpointing
    def state_dict(self):
        return {'shadow': deepcopy(self.shadow), 'decay': self.decay}
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


class EMACallback(TrainerCallback):
    """
    Trainer callback for updating EMA after each step
    """
    def __init__(self, ema):
        self.ema = ema
        
    def on_step_end(self, args, state, control, **kwargs):
        self.ema.update()
        
    def on_evaluate(self, args, state, control, **kwargs):
        # Apply EMA for evaluation
        self.ema.apply_shadow()
        
    def on_evaluate_end(self, args, state, control, **kwargs):
        # Restore original weights after evaluation
        self.ema.restore()

# Then modify your install_dependencies function to ensure it's properly installed:
def install_dependencies():
    try:
        import Levenshtein
    except ImportError:
        print("Installing python-Levenshtein...")
        os.system("pip install python-Levenshtein")
        # Try importing again to confirm installation
        try:
            import Levenshtein
        except ImportError:
            print("Warning: Failed to import Levenshtein module even after installation.")
            print("Trying alternative package...")
            os.system("pip install Levenshtein")
            try:
                import Levenshtein
            except ImportError:
                print("Critical error: Could not import Levenshtein package.")
    
    try:
        import jiwer
    except ImportError:
        print("Installing jiwer...")
        os.system("pip install jiwer")
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# Install dependencies first
install_dependencies()

# Check GPU availability
print("Checking GPU availability:")
os.system("nvidia-smi")

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

def save_model_custom(model, tokenizer, image_processor, output_dir):
    """Custom model saving function that avoids the DTensor import error"""
    # Create output directories
    model_dir = os.path.join(output_dir, "model")
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    processor_dir = os.path.join(output_dir, "image_processor")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)
    os.makedirs(processor_dir, exist_ok=True)
    
    # Save model state dict and config separately
    model.config.save_pretrained(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
    
    # Save tokenizer and processor
    tokenizer.save_pretrained(tokenizer_dir)
    image_processor.save_pretrained(processor_dir)
    
    print(f"Model saved to {output_dir}")

# Function to plot and save metrics
def plot_and_save_metrics(logs, metric_name, output_dir):
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
    plt.savefig(os.path.join(output_dir, f'{metric_name}_plot.png'))
    plt.close()

# Function to perform inference on test data
def perform_inference(model, image_processor, tokenizer, base_dir, output_dir, device):
    # Create a processor for inference
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    
    # Function to generate text for a single image segment
    def generate_text_from_image_segment(image_path):
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
        print(f"Warning: Test directory {base_dir} does not exist. Skipping inference.")
        return []
    
    # Iterate through each page's folder
    for page_folder in sorted(os.listdir(base_dir)):
        page_path = os.path.join(base_dir, page_folder)
        if os.path.isdir(page_path):
            page_output = f"Processing {page_folder}:\n"
            page_texts = []

            # Sort the line segment images numerically based on the segment number
            line_segment_images = sorted([f for f in os.listdir(page_path) if f.endswith('.jpg')], key=sort_key)

            # Iterate through each sorted line segment in the page folder
            for line_segment_image in line_segment_images:
                line_segment_path = os.path.join(page_path, line_segment_image)
                line_text = generate_text_from_image_segment(line_segment_path)
                page_texts.append((line_segment_image, line_text))
            
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
    with open(os.path.join(output_dir, 'inference_results.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(inference_results))
    
    return inference_results

def main():
    # Set paths
    image_dir = '../../../data/train/All_line_segments'
    text_dir = '../../../data/train/All_line_texts'
    model_dir = "../../../models/ema"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create plots directory
    plots_dir = os.path.join(model_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Try a different model and tokenizer to avoid dependency issues
    print("Loading components with fallback options...")
    
    try:
        # Try loading the default model first
        model_name = "microsoft/trocr-large-handwritten"
        
        print("Loading image processor...")
        image_processor = ViTImageProcessor.from_pretrained(model_name)
        
        print("Loading tokenizer (RobertaTokenizerFast)...")
        # Use RobertaTokenizerFast instead of XLMRobertaTokenizer to avoid SentencePiece dependency
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        
        print("Loading model...")
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
    except Exception as e:
        print(f"Error with primary model: {e}")
        print("Trying fallback model...")
        
        # Fallback to a simpler model
        model_name = "microsoft/trocr-base-printed"
        
        print("Loading image processor...")
        image_processor = ViTImageProcessor.from_pretrained(model_name)
        
        print("Loading tokenizer (RobertaTokenizerFast)...")
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        
        print("Loading model...")
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # Create processor for training utilities
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    
    # Configure model
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.ignore_pad_token_for_loss = True
    
    # Initialize EMA with appropriate decay rate
    # A higher decay rate (0.9999) is good for longer training runs
    # For shorter runs, you might want a lower value (0.999 or 0.99)
    ema = EMA(model, decay=0.9999, update_every=1)
    ema_callback = EMACallback(ema)
    
    # Prepare dataset and split
    dataset = SpanishDocumentsDataset(image_dir, text_dir, image_processor, tokenizer)
    
    dataset_size = len(dataset)
    eval_size = int(dataset_size * 0.1)  # 10% for evaluation
    train_size = dataset_size - eval_size
    
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=8, collate_fn=collate_fn)
    
    # Print dataloader information
    print(f"Number of training batches: {len(train_loader)}")
    for batch in train_loader:
        print(f"Sample batch shapes - Pixel values: {batch['pixel_values'].shape}, Labels: {batch['labels'].shape}")
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

        return {"cer": cer, "wer": wer, "bleu": bleu}
    
    # Set up training arguments with save_strategy="no" to avoid checkpoint saving issues
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        fp16=False,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_grad_norm=1.0,
        logging_dir=os.path.join(model_dir, 'logs'),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100000,
        warmup_steps=500,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,  # Set to False since we're not saving checkpoints
        report_to="none",
    )
    
    # Prepare optimizer and scheduler
    total_train_steps = (len(train_loader) * training_args.num_train_epochs)
    optimizer = AdamW(model.parameters(), lr=5e-6)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=500, 
        num_training_steps=total_train_steps
    )
    
    # Create early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3, 
        early_stopping_threshold=0.0
    )
    
    # Define trainer with EMA callback
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, scheduler),
        data_collator=collate_fn,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback, ema_callback]  # Add EMA callback
    )
    
    # Train the model
    print("Starting model training...")
    trainer.train()
    
    # Apply EMA weights for final model saving and inference
    print("Applying EMA weights to model for better performance...")
    ema.apply_shadow()
    
    print("Saving model and processor components...")
    save_model_custom(model, tokenizer, image_processor, model_dir)
    
    # Plot and save metrics
    print("Plotting metrics...")
    train_logs = trainer.state.log_history
    
    for metric in ['loss', 'cer', 'wer', 'bleu']:
        plot_and_save_metrics(train_logs, metric, plots_dir)
    
    # Perform inference on test data (model already has EMA weights applied)
    print("Performing inference on test data with EMA weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    test_dir = "../../../data/test"
    inference_results = perform_inference(
        model, image_processor, tokenizer, test_dir, model_dir, device
    )
    
    # Print inference results
    for result in inference_results:
        print(result)
    
    print(f"Training complete. Model saved to {model_dir}")
    print(f"Plots saved to {plots_dir}")
    print(f"Inference results saved to {os.path.join(model_dir, 'inference_results.txt')}")

if __name__ == "__main__":
    main()