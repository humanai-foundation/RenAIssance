import os
# Force transformers and datasets into offline mode so they don't try to resolve online
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import re

# Post-processing: fix systematic misreads of early modern Spanish abbreviations
_ABBREV_FIXES = [
    (re.compile(r'\b35os\b'), 'dhos'),
    (re.compile(r'\b35as\b'), 'dhas'),
    (re.compile(r'\b35o\b'),  'dho'),
    (re.compile(r'\b35a\b'),  'dha'),
    (re.compile(r'\b35Os\b'), 'Dhos'),
    (re.compile(r'\b35As\b'), 'Dhas'),
    (re.compile(r'\b35O\b'),  'Dho'),
    (re.compile(r'\b35A\b'),  'Dha'),
]

def fix_abbreviations(text):
    for pattern, replacement in _ABBREV_FIXES:
        text = pattern.sub(replacement, text)
    return text
import argparse
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import textwrap
# Disable PIL's decompression bomb limit for large scans
Image.MAX_IMAGE_PIXELS = None
import docx
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
from peft import PeftModel
from jiwer import cer, wer
from tqdm import tqdm

# Max dimension to resize large images before VLM inference (prevents OOM)
MAX_IMAGE_DIM = 4096


# ─── Helpers ─────────────────────────────────────────────────────────────────

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
        print(f"Error reading {file_path}: {e}")
        return ""


def sanitize_filename_for_match(name):
    base = os.path.splitext(name)[0]
    base = base.replace('_transcription', '')
    return re.sub(r'[^a-zA-Z0-9]', '', base).lower()


def collect_images(image_dir):
    """Return a list of dicts with image_path (and optional docx_path)."""
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    return [{"image_path": os.path.join(image_dir, f), "docx_path": None}
            for f in image_files]


def align_files(image_dir, docx_dir):
    """Return aligned image+docx pairs where both exist."""
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    docx_files  = [f for f in os.listdir(docx_dir) if f.endswith('.docx') or f.endswith('.txt')]

    img_map  = {sanitize_filename_for_match(f): f for f in image_files}
    docx_map = {sanitize_filename_for_match(f): f for f in docx_files}

    pairs = []
    for key in img_map:
        if key in docx_map:
            pairs.append({
                "image_path": os.path.join(image_dir, img_map[key]),
                "docx_path":  os.path.join(docx_dir,  docx_map[key]),
            })
        else:
            # No matching docx → inference-only for this image
            pairs.append({
                "image_path": os.path.join(image_dir, img_map[key]),
                "docx_path":  None,
            })

    for key in docx_map:
        if key not in img_map:
            print(f"Warning: No matching image found for docx: {docx_map[key]}")

    return pairs


# ─── Visual Transcription ────────────────────────────────────────────────────

def create_transcription_image(original_image_path, transcription, output_folder="Visual_Results"):
    """Render transcription text onto a white image matching the original page dimensions."""
    original_img = Image.open(original_image_path)
    width, height = int(original_img.size[0]), int(original_img.size[1])

    font_paths = [
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf"
    ]

    valid_font_path = None
    for path in font_paths:
        try:
            ImageFont.truetype(path, 10)
            valid_font_path = path
            break
        except Exception:
            pass

    margin = int(width * 0.05)
    usable_width = width - (2 * margin)

    # Natural font size: target ~25 lines per page height, never below 20px
    natural_font_size = max(height // 25, 20)

    if valid_font_path:
        font_size = natural_font_size
        font = ImageFont.truetype(valid_font_path, font_size)
        chars_per_line = max(1, int(usable_width / (font_size * 0.55)))
        wrapped_lines = []
        for line in transcription.split('\n'):
            wrapped_lines.extend(textwrap.wrap(line, width=chars_per_line) or [""])
        total_text_height = len(wrapped_lines) * int(font_size * 1.3)
    else:
        font = ImageFont.load_default()
        font_size = natural_font_size
        chars_per_line = max(1, int(usable_width / 8))
        wrapped_lines = []
        for line in transcription.split('\n'):
            wrapped_lines.extend(textwrap.wrap(line, width=chars_per_line) or [""])
        total_text_height = len(wrapped_lines) * int(font_size * 1.3)

    # Expand canvas height if text overflows rather than shrinking the font
    canvas_height = max(height, total_text_height + 2 * margin)
    transcription_img = Image.new("RGB", (width, canvas_height), "white")
    draw = ImageDraw.Draw(transcription_img)

    y_text = margin
    for line in wrapped_lines:
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_height = bbox[3] - bbox[1]
        except AttributeError:
            line_height = font_size
        draw.text((margin, y_text), line, font=font, fill="black")
        y_text += line_height + int(font_size * 0.3)

    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.basename(original_image_path)
    output_path = os.path.join(output_folder, f"transcribed_{base_name}")
    transcription_img.save(output_path)


# ─── Line Segmentation ──────────────────────────────────────────────────────

def segment_lines(pil_img, min_line_height=20):
    """Segment a full-page handwriting image into individual text lines
    using horizontal projection profiles (classical CV approach).

    Returns a list of PIL Image crops, one per detected text line.
    """
    # Convert PIL → OpenCV grayscale
    img_np = np.array(pil_img.convert("L"))
    # Binarize using Otsu's threshold (invert so text = white)
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection: sum of white pixels per row
    h_proj = np.sum(binary, axis=1)

    # Find line boundaries: rows where projection > threshold
    threshold = h_proj.max() * 0.02  # 2% of max projection
    in_line = h_proj > threshold

    lines = []
    start = None
    for i, val in enumerate(in_line):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_line_height:
                lines.append((start, i))
            start = None
    # Handle last line if image ends mid-line
    if start is not None and len(in_line) - start >= min_line_height:
        lines.append((start, len(in_line)))

    if not lines:
        # Fallback: return the whole image as one "line"
        print("  Warning: line segmentation found no lines, using full image.")
        return [pil_img]

    # Add small vertical padding around each line
    h = img_np.shape[0]
    pad = 5
    crops = []
    for (y1, y2) in lines:
        y1_padded = max(0, y1 - pad)
        y2_padded = min(h, y2 + pad)
        crop = pil_img.crop((0, y1_padded, pil_img.width, y2_padded))
        crops.append(crop)

    return crops


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="4-Stage VLM+OCR Pipeline")
    parser.add_argument("--lora", type=str, default=None, metavar="ADAPTER_DIR",
                        help="Path to LoRA adapter (e.g. qwen2.5-vl-ocr-lora/final).")
    parser.add_argument("--trocr", type=str, default=None, metavar="TROCR_MODEL_DIR",
                        help="Path to fine-tuned TrOCR model. When provided, enables "
                             "the full 4-stage pipeline with line segmentation.")
    parser.add_argument("--image-dir", type=str, default="Handwriting-scans",
                        help="Folder containing input images. Default: Handwriting-scans")
    parser.add_argument("--docx-dir", type=str, default="Handwriting-transcriptions",
                        help="Folder containing ground-truth .docx/.txt files. "
                             "If omitted or folder absent, runs in inference-only mode (no CER/WER).")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to Qwen2.5-VL-7B-Instruct base model.")
    parser.add_argument("--output-visual", type=str, default="Visual_Results",
                        help="Folder to save visual transcription images. Default: Visual_Results")
    args = parser.parse_args()

    image_dir = args.image_dir
    docx_dir  = args.docx_dir

    # ── Determine whether we have ground truth ──────────────────────────────
    has_gt = os.path.isdir(docx_dir)
    if not has_gt:
        print(f"Ground-truth folder '{docx_dir}' not found → INFERENCE-ONLY mode (no CER/WER).")

    # ── Collect images ──────────────────────────────────────────────────────
    if has_gt:
        items = align_files(image_dir, docx_dir)
    else:
        items = collect_images(image_dir)
    print(f"Found {len(items)} image(s) to process.")

    # ── Find Qwen base model ────────────────────────────────────────────────
    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    print(f"Using Qwen model from: {model_dir}")

    # ── Load Qwen processor + model ─────────────────────────────────────────
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

    # ── Optionally load MIM-TrOCR ───────────────────────────────────────────
    trocr_processor = None
    trocr_model     = None
    if args.trocr:
        candidate_trocr = [
            args.trocr,
            "/home/vis-comp/aniketjunghare/GSoC25/model/final_mim_trocr_model",
        ]
        trocr_dir = next((d for d in candidate_trocr if os.path.exists(d)), None)
        if trocr_dir is None:
            raise FileNotFoundError(f"MIM-TrOCR model not found. Looked in: {candidate_trocr}")
        print(f"Loading MIM-TrOCR from: {trocr_dir}")
        trocr_processor = TrOCRProcessor.from_pretrained(trocr_dir, use_fast=True)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_dir).to("cuda").eval()
        print("MIM-TrOCR loaded — will run line-level OCR in Stage 2a.")

    def run_trocr_line(line_img: Image.Image) -> str:
        """Run MIM-TrOCR on a single cropped text line."""
        try:
            pv = trocr_processor(images=line_img, return_tensors="pt").pixel_values.to("cuda")
            with torch.no_grad():
                gen = trocr_model.generate(pv, num_beams=4, early_stopping=True, max_new_tokens=128)
            return trocr_processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        except Exception as e:
            print(f"  Warning: TrOCR failed on a line crop: {e}")
            return ""

    def run_trocr_full_page(pil_img: Image.Image) -> str:
        """Segment page into lines, run TrOCR on each line, combine."""
        line_crops = segment_lines(pil_img)
        line_texts = []
        for i, crop in enumerate(line_crops):
            text = run_trocr_line(crop)
            if text:
                line_texts.append(text)
        result = "\n".join(line_texts)
        # Quality check: if TrOCR output is suspiciously short relative to line count,
        # treat it as garbage and return empty string to fall back to VLM-only mode
        if line_crops and len(result.strip()) < len(line_crops) * 3:
            print(f"  Warning: TrOCR output too short ({len(result.strip())} chars for "
                  f"{len(line_crops)} lines) — falling back to VLM-only mode.")
            return ""
        return result

    def run_qwen(img: Image.Image, prompt: str) -> str:
        """Run a single Qwen inference pass and return the decoded text."""
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

    # ── Process each image ───────────────────────────────────────────────────
    results = []

    for item in tqdm(items, desc="Processing images"):
        image_path = item["image_path"]
        docx_path  = item["docx_path"]

        # Ground truth (optional)
        ground_truth = None
        if docx_path:
            ground_truth = extract_text_from_file(docx_path) or None

        # Load + resize image
        try:
            image = Image.open(image_path).convert("RGB")
            if max(image.size) > MAX_IMAGE_DIM:
                image.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)
                print(f"  Resized {os.path.basename(image_path)} to {image.size}")
        except Exception as e:
            print(f"Failed to load image from {image_path}: {e}")
            continue

        # ── STAGE 1: VLM Image Analysis ─────────────────────────────────────
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
        print(f"  Stage 1 (Analysis): {image_analysis[:150]}{'...' if len(image_analysis) > 150 else ''}")

        # ── STAGE 2a: Line Segmentation + TrOCR (if available) ──────────────
        trocr_text = ""
        if trocr_model is not None:
            trocr_text = run_trocr_full_page(image)

        # ── STAGE 2b: VLM Literal Read ──────────────────────────────────────
        pass1_prompt = (
            "Transcribe every line of handwritten text in this image exactly as it appears.\n"
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
            "- Output only the transcribed text, line by line."
        )
        vlm_literal_text = run_qwen(image, pass1_prompt)
        print(f"  Stage 2 (VLM read): {vlm_literal_text[:120]}{'...' if len(vlm_literal_text) > 120 else ''}")

        # ── STAGE 3: VLM Reconciliation & Correction ────────────────────────
        if trocr_text:
            # 4-stage mode: reconcile TrOCR + VLM readings
            correction_prompt = (
                f"Document analysis: {image_analysis}\n\n"
                f"Two independent OCR readings of this handwritten document were produced:\n\n"
                f"--- Reading A (line-level OCR) ---\n{trocr_text}\n---\n\n"
                f"--- Reading B (VLM full-page read) ---\n{vlm_literal_text}\n---\n\n"
                "Using the image above and BOTH readings as evidence, produce the final "
                "accurate transcription. Where the two readings disagree, use the image "
                "to determine which is correct. Fix only genuine character-level misreads.\n\n"
                "IMPORTANT constraints:\n"
                "- Do NOT modernise spelling, vocabulary, or grammar. Keep all historical and "
                "archaic forms exactly as written.\n"
                "- Preserve the EXACT capitalisation from the manuscript — do not change a "
                "capital letter to lowercase or vice versa, even in the middle of a sentence.\n"
                "- Do NOT expand abbreviations. Copy them as they appear: 'dho', 'dha', 'Dho', "
                "'Dha', 'dhos', 'dhas', 'q̃', 'p̃', etc.\n"
                "- CRITICAL misread to fix: In this script, 'd' resembles '3' and 'h' resembles '5'. "
                "If any reading contains '35o', '35a', '35os', or '35as', these are ALWAYS misreads "
                "of 'dho', 'dha', 'dhos', 'dhas' respectively. Correct every occurrence.\n"
                "Provide ONLY the final corrected transcript, nothing else."
            )
        else:
            # VLM-only mode: correct the VLM literal read
            correction_prompt = (
                f"Document analysis: {image_analysis}\n\n"
                f"A preliminary reading of this handwritten document produced:\n"
                f"---\n{vlm_literal_text}\n---\n\n"
                "Using BOTH the image above and this preliminary reading as a guide, "
                "produce the final accurate transcription. Fix only genuine misread characters "
                "that are clearly contradicted by the visual evidence in the image.\n\n"
                "IMPORTANT constraints:\n"
                "- Do NOT modernise spelling, vocabulary, or grammar. Keep all historical and "
                "archaic forms exactly as written.\n"
                "- Preserve the EXACT capitalisation from the manuscript — do not change a "
                "capital letter to lowercase or vice versa, even in the middle of a sentence.\n"
                "- Do NOT expand abbreviations. Copy them as they appear: 'dho', 'dha', 'Dho', "
                "'Dha', 'dhos', 'dhas', 'q̃', 'p̃', etc.\n"
                "- CRITICAL misread to fix: In this script, 'd' resembles '3' and 'h' resembles '5'. "
                "If the preliminary reading contains '35o', '35a', '35os', or '35as', "
                "these are ALWAYS misreads of 'dho', 'dha', 'dhos', 'dhas' respectively. "
                "Correct every occurrence.\n"
                "Provide ONLY the final corrected transcript, nothing else."
            )

        prediction = run_qwen(image, correction_prompt)
        # Post-processing safety net for any remaining abbreviation misreads
        prediction = fix_abbreviations(prediction)

        # ── Metrics (only when ground truth available) ──────────────────────
        curr_cer = curr_wer = None
        if ground_truth:
            try:
                curr_cer = cer(ground_truth, prediction)
                curr_wer = wer(ground_truth, prediction)
                print(f"\n--- {os.path.basename(image_path)} --- CER: {curr_cer:.4f} | WER: {curr_wer:.4f}")
            except Exception as e:
                print(f"Error computing metrics for {image_path}: {e}")
        else:
            print(f"\n--- {os.path.basename(image_path)} (inference-only, no ground truth) ---")

        print(f"  Predicted: {prediction[:200]}{'...' if len(prediction) > 200 else ''}")

        # ── Create visual transcription image ────────────────────────────────
        create_transcription_image(image_path, prediction, output_folder=args.output_visual)

        # ── Save full stage-by-stage transcript as .txt ──────────────────────
        os.makedirs(args.output_visual, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_path  = os.path.join(args.output_visual, f"{base_name}_transcript.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("STAGE 1 — Document Analysis\n")
            f.write("=" * 80 + "\n")
            f.write(image_analysis + "\n")

            if trocr_text:
                f.write("\n" + "=" * 80 + "\n")
                f.write("STAGE 2a — TrOCR Line-Level Reading\n")
                f.write("=" * 80 + "\n")
                f.write(trocr_text + "\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("STAGE 2b — VLM Literal Reading\n")
            f.write("=" * 80 + "\n")
            f.write(vlm_literal_text + "\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("STAGE 3 — Final Transcription\n")
            f.write("=" * 80 + "\n")
            f.write(prediction + "\n")

            if curr_cer is not None:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"CER: {curr_cer:.4f}  |  WER: {curr_wer:.4f}\n")

        results.append({
            "File":            os.path.basename(image_path),
            "CER":             curr_cer,
            "WER":             curr_wer,
        })

    # ── Summary ─────────────────────────────────────────────────────────────
    if results:
        scored = [r for r in results if r["CER"] is not None]
        print(f"\nVisual transcriptions and transcript .txt files saved to {args.output_visual}/")
        if scored:
            import statistics
            cer_vals = [r["CER"] for r in scored]
            wer_vals = [r["WER"] for r in scored]

            avg_cer = statistics.mean(cer_vals)
            med_cer = statistics.median(cer_vals)
            min_cer = min(cer_vals)
            max_cer = max(cer_vals)

            avg_wer = statistics.mean(wer_vals)
            med_wer = statistics.median(wer_vals)
            min_wer = min(wer_vals)
            max_wer = max(wer_vals)

            print(f"\n## Results Analysis ({len(scored)}/{len(results)} images with ground truth)\n")
            print(f"| {'Metric':<10} | {'Average':<12} | {'Median':<12} | {'Best':<12} | {'Worst':<12} |")
            print(f"|{'-'*12}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*14}|")
            print(f"| {'CER':<10} | {avg_cer:<12.4f} | {med_cer:<12.4f} | {min_cer:<12.4f} | {max_cer:<12.4f} |")
            print(f"| {'WER':<10} | {avg_wer:<12.4f} | {med_wer:<12.4f} | {min_wer:<12.4f} | {max_wer:<12.4f} |")
        else:
            print(f"Inference-only run — no ground truth provided, no CER/WER computed.")


if __name__ == "__main__":
    main()
