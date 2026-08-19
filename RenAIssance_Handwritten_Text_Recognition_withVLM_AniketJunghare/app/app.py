"""
Gradio app for GSoC final submission: Old Spanish handwritten document
transcription using a LoRA fine-tuned Qwen2.5-VL-7B-Instruct model.
Runs on Hugging Face Spaces with the ZeroGPU hardware tier.
This mirrors the 3-stage (or optional 4-stage with TrOCR) pipeline from
the original inference notebook: document analysis -> literal read
(+ optional TrOCR line-level read) -> correction pass -> abbreviation
post-fix -> side-by-side visualization.
"""

import os
import re
import textwrap

import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
from peft import PeftModel

# ---------------------------------------------------------------------------
# Config — set these as Space "Variables" (Settings > Variables and secrets)
# or edit the defaults below before pushing.
# ---------------------------------------------------------------------------
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
LORA_ADAPTER_ID = os.environ.get("LORA_ADAPTER_ID", "your-username/qwen2.5-vl-spanish-ocr-lora")
# Optional: set TROCR_MODEL_ID as a Space variable to enable the 4-stage pipeline.
# Leave unset to run the 3-stage VLM-only pipeline.
TROCR_MODEL_ID = os.environ.get("TROCR_MODEL_ID", "")

MAX_IMAGE_DIM = 4096

# ---------------------------------------------------------------------------
# Model loading — happens once at Space startup, outside any @spaces.GPU
# function. ZeroGPU virtualizes the CUDA calls, so `.to("cuda")` here is fine.
# ---------------------------------------------------------------------------
print(f"Loading base model: {BASE_MODEL_ID}")
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
)

print(f"Loading LoRA adapter: {LORA_ADAPTER_ID}")
# Force the adapter weights to load on CPU. Under ZeroGPU, torch.cuda.is_available()
# reports True even outside an allocated GPU context, which makes peft's default
# device auto-detection try to load the safetensors weights straight to "cuda"
# and fail with "No CUDA GPUs are available". Loading on CPU first avoids that;
# moving the whole (already-loaded) model to "cuda" afterward is a safe, lazy op.
model = PeftModel.from_pretrained(model, LORA_ADAPTER_ID, torch_device="cpu")
model = model.to("cuda")
model.eval()
print("Qwen2.5-VL + LoRA loaded.")

trocr_processor = None
trocr_model = None
if TROCR_MODEL_ID:
    try:
        print(f"Loading TrOCR from: {TROCR_MODEL_ID}")
        trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID, use_fast=True)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID).to("cuda").eval()
        print("TrOCR loaded — pipeline will run in 4-stage mode.")
    except Exception as e:
        print(f"Warning: failed to load TROCR_MODEL_ID ({TROCR_MODEL_ID}): {e}")
        print("Falling back to 3-stage VLM-only mode.")
else:
    print("TROCR_MODEL_ID not set — running in 3-stage VLM-only mode.")

# ---------------------------------------------------------------------------
# Pipeline internals (unchanged from the inference notebook)
# ---------------------------------------------------------------------------
_ABBREV_FIXES = [
    (re.compile(r"\b35os\b"), "dhos"),
    (re.compile(r"\b35as\b"), "dhas"),
    (re.compile(r"\b35o\b"), "dho"),
    (re.compile(r"\b35a\b"), "dha"),
    (re.compile(r"\b35Os\b"), "Dhos"),
    (re.compile(r"\b35As\b"), "Dhas"),
    (re.compile(r"\b35O\b"), "Dho"),
    (re.compile(r"\b35A\b"), "Dha"),
]


def fix_abbreviations(text):
    for pattern, replacement in _ABBREV_FIXES:
        text = pattern.sub(replacement, text)
    return text


def run_qwen(img, prompt):
    """Run a single Qwen VLM inference pass."""
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt},
    ]}]
    text_prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=img, text=[text_prompt], padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=10,
            repetition_penalty=1.2,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def segment_lines(pil_img, min_line_height=20):
    """Segment a full-page image into individual text line crops using horizontal projection."""
    import cv2

    img_np = np.array(pil_img.convert("L"))
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_proj = np.sum(binary, axis=1)
    threshold = h_proj.max() * 0.02
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
    if start is not None and len(in_line) - start >= min_line_height:
        lines.append((start, len(in_line)))

    if not lines:
        return [pil_img]

    h = img_np.shape[0]
    pad = 5
    crops = []
    for (y1, y2) in lines:
        crop = pil_img.crop((0, max(0, y1 - pad), pil_img.width, min(h, y2 + pad)))
        crops.append(crop)
    return crops


def run_trocr_line(line_img):
    """Run TrOCR on a single cropped text line."""
    try:
        pv = trocr_processor(images=line_img, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad():
            gen = trocr_model.generate(pv, num_beams=4, early_stopping=True, max_new_tokens=128)
        return trocr_processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    except Exception:
        return ""


def run_trocr_full_page(pil_img):
    """Segment page into lines, run TrOCR on each, combine into full-page text."""
    line_crops = segment_lines(pil_img)
    line_texts = [t for crop in line_crops if (t := run_trocr_line(crop))]
    result = "\n".join(line_texts)
    if line_crops and len(result.strip()) < len(line_crops) * 3:
        return ""
    return result


def create_transcription_image(original_image, transcription):
    width, height = original_image.size
    transcription_img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(transcription_img)

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
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
    usable_width = width - 2 * margin
    usable_height = height - 2 * margin

    if valid_font_path:
        font_size = int(height * 0.2)
        while font_size > 14:
            font = ImageFont.truetype(valid_font_path, font_size)
            chars_per_line = max(1, int(usable_width / (font_size * 0.55)))
            wrapped_lines = []
            for line in transcription.split("\n"):
                wrapped_lines.extend(textwrap.wrap(line, width=chars_per_line))
            if len(wrapped_lines) * int(font_size * 1.3) <= usable_height:
                break
            font_size -= 2
    else:
        font = ImageFont.load_default()
        font_size = 14
        chars_per_line = max(1, int(usable_width / 8))
        wrapped_lines = []
        for line in transcription.split("\n"):
            wrapped_lines.extend(textwrap.wrap(line, width=chars_per_line))

    y_text = margin
    for line in wrapped_lines:
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_height = bbox[3] - bbox[1]
        except AttributeError:
            line_height = font_size
        draw.text((margin, y_text), line, font=font, fill="black")
        y_text += line_height + int(font_size * 0.3)

    return transcription_img


# ---------------------------------------------------------------------------
# ZeroGPU entry point — only the actual GPU work lives inside this function.
# duration is a generous upper bound for the ZeroGPU scheduler; lower it if
# your typical runs are much faster, to reduce queue wait for other users.
# ---------------------------------------------------------------------------
@spaces.GPU(duration=90)
def transcribe(image_path, progress=gr.Progress()):
    if image_path is None:
        return "Please upload an image.", "", "", None

    image = Image.open(image_path).convert("RGB")
    if max(image.size) > MAX_IMAGE_DIM:
        image.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)

    progress(0.1, desc="Analyzing document...")
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

    trocr_text = ""
    if trocr_model is not None:
        progress(0.3, desc="Running line-level OCR pass...")
        trocr_text = run_trocr_full_page(image)

    progress(0.5, desc="Running literal VLM transcription...")
    literal_prompt = (
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
    vlm_literal_text = run_qwen(image, literal_prompt)

    progress(0.75, desc="Correcting and finalizing transcription...")
    if trocr_text:
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
            "- Preserve the line and paragraph structure of the document.\n"
            "Provide ONLY the final corrected transcript, nothing else."
        )

    final_transcription = run_qwen(image, correction_prompt)
    final_transcription = fix_abbreviations(final_transcription)

    progress(0.95, desc="Rendering output...")
    visual_output = create_transcription_image(image, final_transcription)

    stage2_display = vlm_literal_text
    if trocr_text:
        stage2_display = f"[TrOCR line-level reading]\n{trocr_text}\n\n[VLM literal reading]\n{vlm_literal_text}"

    return final_transcription, image_analysis, stage2_display, visual_output


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Old Spanish Handwriting Transcription") as demo:
    gr.Markdown(
        "# Old Spanish Handwritten Document Transcription\n"
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Manuscript image")
            run_btn = gr.Button("Transcribe", variant="primary")
        with gr.Column(scale=1):
            final_output = gr.Textbox(label="Final transcription", lines=12)
            visual_output = gr.Image(label="Transcription (rendered)")

    with gr.Accordion("Pipeline detail (document analysis + intermediate readings)", open=False):
        analysis_output = gr.Textbox(label="Stage 1 — Document analysis", lines=3)
        stage2_output = gr.Textbox(label="Stage 2 — Preliminary reading(s)", lines=10)

    run_btn.click(
        fn=transcribe,
        inputs=[image_input],
        outputs=[final_output, analysis_output, stage2_output, visual_output],
    )

if __name__ == "__main__":
    demo.queue().launch()
