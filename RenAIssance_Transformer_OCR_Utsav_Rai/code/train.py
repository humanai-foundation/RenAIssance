import os
import yaml
import torch
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from datasets import load_metric
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from utils import SpanishDocumentsDataset, collate_fn, compute_metrics, plot_metrics, generate_text_from_image_segment, sort_key

# Load configuration
try:
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    # Ensure numerical values are properly parsed
    config["train_batch_size"] = int(config["train_batch_size"])
    config["eval_batch_size"] = int(config["eval_batch_size"])
    config["num_train_epochs"] = int(config["num_train_epochs"])
    config["max_grad_norm"] = float(config["max_grad_norm"])
    config["logging_steps"] = int(config["logging_steps"])
    config["eval_steps"] = int(config["eval_steps"])
    config["save_steps"] = int(config["save_steps"])
    config["warmup_steps"] = int(config["warmup_steps"])
    config["weight_decay"] = float(config["weight_decay"])
    config["learning_rate"] = float(config["learning_rate"])
    config["early_stopping_patience"] = int(config["early_stopping_patience"])
    config["early_stopping_threshold"] = float(config["early_stopping_threshold"])
except FileNotFoundError:
    raise Exception("Configuration file not found.")
except yaml.YAMLError as e:
    raise Exception(f"Error reading configuration file: {e}")
except ValueError as e:
    raise Exception(f"Error in configuration values: {e}")

# Optionally initialize wandb
if config.get("use_wandb"):
    try:
        import wandb
        wandb.login(key=config["wandb_key"])
        wandb.init(project=config["wandb_project"])
    except Exception as e:
        print(f"Failed to initialize wandb: {e}")
        config["use_wandb"] = False

# Initialize processor and model
try:
    processor = TrOCRProcessor.from_pretrained(config["model_name"])
    model = VisionEncoderDecoderModel.from_pretrained(config["model_name"])
except Exception as e:
    raise Exception(f"Failed to load model or processor: {e}")

model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.ignore_pad_token_for_loss = True

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare dataset and data loader
try:
    dataset = SpanishDocumentsDataset(config["image_dir"], config["text_dir"], processor)
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    train_loader = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=config["eval_batch_size"], collate_fn=collate_fn)
except Exception as e:
    raise Exception(f"Failed to prepare dataset or dataloaders: {e}")

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=config["model_dir"],
    per_device_train_batch_size=config["train_batch_size"],
    per_device_eval_batch_size=config["eval_batch_size"],
    fp16=config["fp16"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    num_train_epochs=config["num_train_epochs"],
    max_grad_norm=config["max_grad_norm"],
    logging_dir=config["logging_dir"],
    logging_steps=config["logging_steps"],
    evaluation_strategy=config["evaluation_strategy"],
    eval_steps=config["eval_steps"],
    save_steps=config["save_steps"],
    warmup_steps=config["warmup_steps"],
    weight_decay=config["weight_decay"],
    save_total_limit=config["save_total_limit"],
    load_best_model_at_end=config["load_best_model_at_end"],
    report_to="wandb" if config["use_wandb"] else None,
)

# Optimizer and scheduler
try:
    total_train_steps = len(train_loader) * training_args.num_train_epochs
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config["warmup_steps"], num_training_steps=total_train_steps)
except Exception as e:
    raise Exception(f"Failed to set up optimizer or scheduler: {e}")

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizers=(optimizer, scheduler),
    data_collator=collate_fn,
    tokenizer=processor,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"], early_stopping_threshold=config["early_stopping_threshold"])]
)

# Train and save model
try:
    trainer.train()
    trainer.save_model(config["model_dir"])
    processor.save_pretrained(config["model_dir"])
except Exception as e:
    raise Exception(f"Failed during training or saving model: {e}")

# Optionally finish wandb run
if config.get("use_wandb"):
    wandb.finish()

# Extract logs and plot
try:
    train_logs = trainer.state.log_history
    plot_metrics(train_logs, 'loss')
    plot_metrics(train_logs, 'cer')
    plot_metrics(train_logs, 'wer')
    plot_metrics(train_logs, 'bleu')
except Exception as e:
    raise Exception(f"Failed to extract logs or plot metrics: {e}")

# Perform inference on test data and save results
base_dir = config["base_dir"]
try:
    for page_folder in sorted(os.listdir(base_dir)):
        page_path = os.path.join(base_dir, page_folder)
        if os.path.isdir(page_path):
            print(f"Processing {page_folder}:")
            page_texts = []
            line_segment_images = sorted([f for f in os.listdir(page_path) if f.endswith('.jpg')], key=sort_key)
            for line_segment_image in line_segment_images:
                line_segment_path = os.path.join(page_path, line_segment_image)
                line_text = generate_text_from_image_segment(line_segment_path, processor, model)
                page_texts.append((line_segment_image, line_text))
            page_texts.sort(key=lambda x: sort_key(x[0]))
            full_page_text = "\n".join([text for _, text in page_texts])
            with open(os.path.join(page_path, "output.txt"), "w", encoding="utf-8") as f:
                f.write(full_page_text)
except Exception as e:
    raise Exception(f"Failed during inference or saving results: {e}")
