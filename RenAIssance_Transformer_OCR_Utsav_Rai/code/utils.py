import os
import re
import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_metric
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor

# Load configuration
try:
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    # Ensure numerical values are properly parsed
    config["max_grad_norm"] = float(config["max_grad_norm"])
    config["learning_rate"] = float(config["learning_rate"])
    config["early_stopping_threshold"] = float(config["early_stopping_threshold"])
except FileNotFoundError:
    raise Exception("Configuration file not found.")
except yaml.YAMLError as e:
    raise Exception(f"Error reading configuration file: {e}")
except ValueError as e:
    raise Exception(f"Error in configuration values: {e}")

class SpanishDocumentsDataset(Dataset):
    def __init__(self, image_dir, text_dir, processor):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.processor = processor
        self.filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        try:
            image_file = self.filenames[idx]
            text_file = image_file.replace('.jpg', '.txt')
            image_path = os.path.join(self.image_dir, image_file)
            text_path = os.path.join(self.text_dir, text_file)
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
            with open(text_path, 'r', encoding='utf-8') as file:
                text = file.read().strip()
            labels = self.processor.tokenizer(text, return_tensors="pt").input_ids.squeeze()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel_values, "labels": labels}
        except Exception as e:
            raise Exception(f"Failed to load data for index {idx}: {e}")

def collate_fn(batch):
    try:
        pixel_values = [item['pixel_values'] for item in batch]
        labels = [item['labels'] for item in batch]
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        pixel_values = torch.stack(pixel_values)
        return {"pixel_values": pixel_values, "labels": labels}
    except Exception as e:
        raise Exception(f"Failed to collate batch: {e}")

def compute_metrics(eval_pred):
    try:
        processor = TrOCRProcessor.from_pretrained(config["model_name"])
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = logits.argmax(-1)
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = []
        for label in labels:
            label_filtered = [token for token in label if token != -100]
            decoded_label = processor.decode(label_filtered, skip_special_tokens=True)
            decoded_labels.append(decoded_label)
        cer = load_metric("cer").compute(predictions=decoded_preds, references=decoded_labels)
        wer = load_metric("wer").compute(predictions=decoded_preds, references=decoded_labels)
        tokenized_preds = [pred.split() for pred in decoded_preds]
        tokenized_refs = [[ref.split()] for ref in decoded_labels]
        bleu = load_metric("bleu").compute(predictions=tokenized_preds, references=tokenized_refs)
        return {"cer": cer, "wer": wer, "bleu": bleu["bleu"]}
    except Exception as e:
        raise Exception(f"Failed to compute metrics: {e}")

def plot_metrics(logs, metric_name):
    try:
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
        plt.plot(train_steps, train_values, label=f'Training {metric_name.capitalize()}')
        plt.plot(eval_steps, eval_values, label=f'Evaluation {metric_name.capitalize()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name.capitalize())
        plt.title(f'Training and Evaluation {metric_name.capitalize()}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{metric_name}.png")
    except Exception as e:
        raise Exception(f"Failed to plot metrics for {metric_name}: {e}")

def generate_text_from_image_segment(image_path, processor, model):
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(model.device)  # Ensure the input is on the same device as the model
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    except Exception as e:
        raise Exception(f"Failed to generate text from image segment {image_path}: {e}")

def sort_key(filename):
    try:
        match = re.search(r"(\d+)\.jpg", filename)
        if match:
            return int(match.group(1))
        return -1
    except Exception as e:
        raise Exception(f"Failed to extract sort key from filename {filename}: {e}")
