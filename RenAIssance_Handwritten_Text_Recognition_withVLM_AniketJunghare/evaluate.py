import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import re
import argparse
import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import docx
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import PeftModel
from jiwer import cer, wer
from tqdm import tqdm

MAX_IMAGE_DIM = 4096


def extract_text_from_file(file_path):
    try:
        if file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            full_text = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
            return "\n".join(full_text)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def sanitize_filename_for_match(name):
    base = os.path.splitext(name)[0]
    base = base.replace('_transcription', '')
    return re.sub(r'[^a-zA-Z0-9]', '', base).lower()


def align_files(image_dir, gt_dir):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.docx') or f.endswith('.txt')]

    img_map = {sanitize_filename_for_match(f): f for f in image_files}
    gt_map = {sanitize_filename_for_match(f): f for f in gt_files}

    pairs = []
    for key in img_map:
        if key in gt_map:
            pairs.append({
                "image_path": os.path.join(image_dir, img_map[key]),
                "gt_path": os.path.join(gt_dir, gt_map[key]),
            })

    for key in gt_map:
        if key not in img_map:
            print(f"Warning: No matching image found for ground truth: {gt_map[key]}")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Evaluation Pipeline (4-Stage VLM)")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to Qwen2.5-VL-7B-Instruct base model.")
    parser.add_argument("--lora", type=str, default=None, metavar="ADAPTER_DIR",
                        help="Path to LoRA adapter.")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Folder containing input images.")
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="Folder containing ground-truth .txt or .docx files.")
    parser.add_argument("--output-csv", type=str, default="evaluation_results.csv",
                        help="Path to save evaluation CSV. Default: evaluation_results.csv")
    args = parser.parse_args()


    if not os.path.isdir(args.gt_dir):
        print(f"Ground-truth folder '{args.gt_dir}' not found. Cannot evaluate.")
        return

    items = align_files(args.image_dir, args.gt_dir)
    print(f"Found {len(items)} aligned image-ground truth pairs to evaluate.")
    if not items:
        print("No pairs found. Exiting.")
        return

    # Load model 
    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    print(f"Using Qwen model from: {model_dir}")

    processor = AutoProcessor.from_pretrained(model_dir)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda")

    if args.lora:
        if not os.path.exists(args.lora):
            raise FileNotFoundError(f"LoRA adapter not found at: {args.lora}")
        print(f"Loading LoRA adapter from: {args.lora}")
        model = PeftModel.from_pretrained(model, args.lora)
        print("Running in FINE-TUNED (LoRA) mode.")
    else:
        print("Running in BASELINE mode (no LoRA adapter).")
    model.eval()

    def run_qwen(img: Image.Image, prompt: str) -> str:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text":  prompt},
        ]}]
        text_prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=img, text=[text_prompt], padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=10,
                repetition_penalty=1.2,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    # Evaluate each image
    results = []
    cer_scores = []
    wer_scores = []

    for item in tqdm(items, desc="Evaluating"):
        image_path = item["image_path"]
        gt_path = item["gt_path"]

        ground_truth = extract_text_from_file(gt_path)
        if not ground_truth:
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            if max(image.size) > MAX_IMAGE_DIM:
                image.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)
        except Exception as e:
            print(f"Failed to load image from {image_path}: {e}")
            continue

        # STAGE 1: VLM Image Analysis 
        analysis_prompt = (
            "Analyze this handwritten document image. Identify:\n"
            "1. The language or script used\n"
            "2. The approximate time period or style of handwriting\n"
            "3. The overall legibility (clear, moderate, poor)\n"
            "4. Any special features (margin notes, crossed-out words, "
            "multiple columns, stamps, drawings)\n"
            "Be concise — respond in 2-3 sentences."
        )
        image_analysis = run_qwen(image, analysis_prompt)

        # STAGE 2b: VLM Literal Read
        pass1_prompt = (
            "Quickly read the handwritten text in this image as literally as possible. "
            "Output only the raw text you can see, line by line, without any spelling "
            "corrections or interpretation. Include every word even if unclear."
        )
        vlm_literal_text = run_qwen(image, pass1_prompt)

        # STAGE 3: VLM Correction 
        correction_prompt = (
            f"Document analysis: {image_analysis}\n\n"
            f"A preliminary reading of this handwritten document produced:\n"
            f"---\n{vlm_literal_text}\n---\n\n"
            "Using BOTH the image above and this preliminary reading as a guide, "
            "produce the final accurate transcription. Correct any character-level "
            "errors using the visual evidence from the image. If you recognize "
            "archaic abbreviations or obvious spelling errors, output the most likely "
            "intended words. Provide ONLY the final corrected transcript, preserving "
            "the structure of the document."
        )
        prediction = run_qwen(image, correction_prompt)

        # Compute metrics
        try:
            curr_cer = cer(ground_truth, prediction)
            curr_wer = wer(ground_truth, prediction)
            cer_scores.append(curr_cer)
            wer_scores.append(curr_wer)
        except Exception as e:
            print(f"Error computing metrics for {image_path}: {e}")
            curr_cer = curr_wer = None

        print(f"\n--- {os.path.basename(image_path)} --- CER: {curr_cer:.4f} | WER: {curr_wer:.4f}" if curr_cer is not None else "")

        results.append({
            "File":            os.path.basename(image_path),
            "Ground_Truth":    ground_truth,
            "Stage1_Analysis": image_analysis,
            "Stage2b_Literal": vlm_literal_text,
            "Final_Prediction": prediction,
            "CER":             f"{curr_cer:.4f}" if curr_cer is not None else "N/A",
            "WER":             f"{curr_wer:.4f}" if curr_wer is not None else "N/A",
        })

    # Save CSV 
    if results:
        import csv
        fieldnames = ["File", "Ground_Truth", "Stage1_Analysis", "Stage2b_Literal",
                      "Final_Prediction", "CER", "WER"]
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output_csv}")

        if cer_scores and wer_scores:
            cer_arr = np.array(cer_scores)
            wer_arr = np.array(wer_scores)

            print("\n\n================ EVALUATION RESULTS ================\n")
            print(f"| {'Metric':<10} | {'Best (Min)':<12} | {'Average':<12} | {'Median':<12} | {'Worst (Max)':<12} |")
            print(f"|{'-'*12}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*14}|")
            print(f"| {'CER':<10} | {np.min(cer_arr):<12.6f} | {np.mean(cer_arr):<12.6f} | {np.median(cer_arr):<12.6f} | {np.max(cer_arr):<12.6f} |")
            print(f"| {'WER':<10} | {np.min(wer_arr):<12.6f} | {np.mean(wer_arr):<12.6f} | {np.median(wer_arr):<12.6f} | {np.max(wer_arr):<12.6f} |")
            print(f"\nTotal images evaluated: {len(cer_scores)}")
            print("\n====================================================\n")


if __name__ == "__main__":
    main()
