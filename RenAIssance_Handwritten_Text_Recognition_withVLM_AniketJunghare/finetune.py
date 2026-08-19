import os
# Force offline mode globally
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# Reduce CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import json
import torch
from PIL import Image
# Disable PIL decompression bomb limit for large scans (same fix as vlm_ocr_pipeline.py)
Image.MAX_IMAGE_PIXELS = None
# Training image size: 1280px → ~91 patches/dim → ~2080 image tokens (fits 80GB DGX A100)
# (was 980px for 44GB L40)
MAX_IMAGE_DIM = 1280
import docx
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def extract_text_from_file(file_path):
    """Read ground-truth text from either a .docx or a plain .txt file."""
    try:
        if file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            full_text = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
            return "\n".join(full_text)
        else:  # .txt
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception as e:
        return ""

def sanitize_filename_for_match(name):
    base = os.path.splitext(name)[0]
    base = base.replace('_transcription', '')
    return re.sub(r'[^a-zA-Z0-9]', '', base).lower()

def align_files(image_dir, gt_dir):
    """Pair images with their ground-truth file (.docx or .txt, same base name)."""
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    gt_files = [f for f in os.listdir(gt_dir)
                if f.endswith('.docx') or f.endswith('.txt')]

    img_map = {sanitize_filename_for_match(f): f for f in image_files}
    gt_map  = {sanitize_filename_for_match(f): f for f in gt_files}

    aligned_pairs = []
    for base_key in gt_map:
        if base_key in img_map:
            aligned_pairs.append({
                "image_path": os.path.join(image_dir, img_map[base_key]),
                "gt_path":    os.path.join(gt_dir,    gt_map[base_key]),
            })
    return aligned_pairs

PROMPTS = [
    # Task 1: Literal reading (matches inference Stage 2)
    ("Transcribe every line of handwritten text in this image exactly as it appears.\n"
     "Strict rules:\n"
     "- Copy each word precisely as written — no spelling corrections, no modernisation.\n"
     "- Preserve the EXACT capitalisation of every letter as it appears in the manuscript "
     "(a capital in the middle of a sentence must stay capital; a lowercase must stay lowercase).\n"
     "- Do NOT expand or interpret abbreviations. Words with superscript letters "
     "(e.g. 'dho', 'dha', 'q̃', 'p̃') must be copied as-is — superscript letters are "
     "abbreviation markers, never digits or numbers.\n"
     "- CRITICAL: In this script, the letters 'd' and 'h' can resemble '3' and '5'. "
     "Never write '35o', '35a', '35os', '35as' — the correct forms are "
     "'dho', 'dha', 'dhos', 'dhas'.\n"
     "- Output only the transcribed text, line by line."),
    # Task 2: Corrected transcription (matches inference Stage 3)
    ("Using the image, produce the final accurate transcription. Fix only genuine misread "
     "characters that are clearly contradicted by the visual evidence in the image.\n\n"
     "IMPORTANT constraints:\n"
     "- Do NOT modernise spelling, vocabulary, or grammar. Keep all historical and "
     "archaic forms exactly as written.\n"
     "- Preserve the EXACT capitalisation from the manuscript — do not change a "
     "capital letter to lowercase or vice versa, even in the middle of a sentence.\n"
     "- Do NOT expand abbreviations. Copy them as they appear: 'dho', 'dha', 'Dho', "
     "'Dha', 'dhos', 'dhas', 'q̃', 'p̃', etc.\n"
     "- CRITICAL misread to fix: In this script, 'd' resembles '3' and 'h' resembles '5'. "
     "If you see '35o', '35a', '35os', or '35as', these are ALWAYS misreads of "
     "'dho', 'dha', 'dhos', 'dhas' respectively. Correct every occurrence.\n"
     "- Preserve the line and paragraph structure of the document.\n"
     "Provide ONLY the final corrected transcript, nothing else."),
]

def prepare_dataset(aligned_files, output_dir="training_data"):
    os.makedirs(output_dir, exist_ok=True)
    dataset = []
    print("Preparing training dataset...")
    for item in tqdm(aligned_files):
        img_path = item["image_path"]
        gt_path  = item["gt_path"]
        ground_truth = extract_text_from_file(gt_path)
        if not ground_truth:
            continue
        try:
            Image.open(img_path)  # verify image opens
        except Exception:
            continue
        # Multi-task: create one sample per prompt per image
        for prompt_idx, prompt in enumerate(PROMPTS):
            dataset.append({
                "image_path": img_path,
                "ground_truth": ground_truth,
                "prompt": prompt,
            })

    jsonl_path = os.path.join(output_dir, "train.jsonl")
    with open(jsonl_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    print(f"Dataset prepared and saved to {jsonl_path}")
    return jsonl_path

# Custom Dataset class to load JSONL into HuggingFace Dataset
from datasets import load_dataset

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL for handwriting OCR")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Folder containing handwriting scan images.")
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="Folder containing ground-truth .docx/.txt files.")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to Qwen2.5-VL-7B-Instruct base model.")
    parser.add_argument("--output-dir", type=str, default="qwen2.5-vl-ocr-lora",
                        help="Directory to save LoRA checkpoints. Default: qwen2.5-vl-ocr-lora")
    args = parser.parse_args()

    image_dir = args.image_dir
    gt_dir    = args.gt_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    print(f"Using model from: {model_dir}")

    print("Aligning files...")
    aligned_files = align_files(image_dir, gt_dir)
    print(f"Found {len(aligned_files)} aligned file pairs.")

    jsonl_path = prepare_dataset(aligned_files)
    
    train_dataset = load_dataset("json", data_files=jsonl_path, split="train")
    
    print(f"Loading processor and model from {model_dir}...")
    processor = AutoProcessor.from_pretrained(model_dir)
    # Cap image tokens at processor level too (matches MAX_IMAGE_DIM=1280)
    processor.image_processor.max_pixels = 1280 * 1280
    processor.image_processor.min_pixels = 224 * 224
    
    # Load model with quantization mapping if running out of memory, or simple fp16/bf16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to("cuda")
    
    # Prepare model for PEFT/LoRA Fine-tuning
    model.enable_input_require_grads()  # crucial for LoRA with gradient checkpointing
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # target linear layers
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,   # 8 samples / 2 = 4 updates per epoch
        optim="adamw_torch",           # safe default — avoids needing bitsandbytes
        save_steps=10,
        logging_steps=5,
        learning_rate=1e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_train_epochs=5,            # 5 epochs × 14 updates = 70 total weight updates
        warmup_steps=10,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,   # parallel data loading on DGX
    )
    
    def qwen_collator(features):
        """Build messages fresh from flat data — avoids HF dataset nested dict issues."""
        all_texts = []
        all_images = []
        for f in features:
            img = Image.open(f["image_path"]).convert("RGB")
            if max(img.size) > MAX_IMAGE_DIM:
                img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)

            # Rebuild messages in Python — never stored as nested JSON
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f["prompt"]}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": f["ground_truth"]}
                ]}
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            all_texts.append(text)
            all_images.append(img)

        inputs = processor(
            text=all_texts,
            images=all_images,
            padding=True,
            return_tensors="pt"
        )
        labels = inputs["input_ids"].clone()
        # Mask padding tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # Mask prompt prefix (image tokens + instruction) — only train on the assistant response
        assist_tokens = processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        for i in range(labels.shape[0]):
            full_ids = inputs["input_ids"][i].tolist()
            for j in range(len(full_ids) - len(assist_tokens)):
                if full_ids[j : j + len(assist_tokens)] == assist_tokens:
                    labels[i, : j + len(assist_tokens)] = -100
                    break
            else:
                print(f"Warning: assistant token boundary not found in sample {i}. "
                      f"Training on full sequence — check if context is being truncated.")
        inputs["labels"] = labels
        return inputs

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=qwen_collator,
        args=training_args,
    )
    


    print("Starting Training...")
    trainer.train()
    
    print("Saving Model...")
    trainer.save_model(os.path.join(output_dir, "final"))
    processor.save_pretrained(os.path.join(output_dir, "final"))
    print("Fine-tuning completed successfully.")

if __name__ == "__main__":
    main()
