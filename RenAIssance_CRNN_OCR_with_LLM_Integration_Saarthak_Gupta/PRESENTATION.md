# RenAIssance — Presentation Content & Technical Summary

An end-to-end OCR system for historical (early-modern Spanish) documents.
This document contains a one-page technical summary followed by detailed,
slide-by-slide content for a 7-slide deck. Every metric is drawn from the
project's own training and evaluation notebooks; sources are noted so you can
cite them.

---

## Technical Summary (one page)

RenAIssance is a full-stack web application that turns scanned historical books
into clean, editable, exportable text. It was built and evaluated on early
modern Spanish printed books, whose broken glyphs, bleed-through, warped pages
and archaic typography defeat off-the-shelf OCR.

The product is a five-step wizard: **Upload → Select pages → Preprocess →
Text detection → OCR & export.** Under the hood it combines four ideas:

1. **A configurable image-cleanup pipeline** — ten OpenCV operations with live
   before/after preview, including a new piecewise deskew that corrects pages
   whose top is straight but whose bottom is warped, plus one-click "book type"
   presets of pretested pipelines.
2. **Layout-aware text detection** built on PaddleOCR PP-OCRv5, with automatic
   model-tier selection that reads free VRAM/RAM at runtime and picks server
   models, mobile models, or CPU accordingly.
3. **A choice of recognition engines** behind one interface — cloud vision
   models (Gemini, ChatGPT) and two locally fine-tuned models (a CNN-LSTM/CRNN
   and a TrOCR transformer), swappable via a Strategy + Factory pattern.
4. **An optional AI correction layer** — a `gemma-3-4b-it` model fine-tuned with
   QLoRA into a Spanish OCR corrector that fixes the residual character and word
   errors the recognizer leaves behind.

**Headline results.** The fine-tuned CRNN reaches roughly **2% character error
and 9% word error** at the line level on held-out books. The fine-tuned TrOCR
reaches **6.1% CER / 21.7% WER**, and post-training quantization to fp16 halves
its memory and triples its speed with no accuracy loss. The Gemma corrector cuts
raw OCR word error by nearly half (**WER 20.9% → 11.5%**, **CER 4.65% → 3.2%**).

**Engineering.** The whole system ships as Docker images with a one-command
launcher that auto-detects the GPU and starts the right variant. A single
Dockerfile builds both GPU and CPU images; a CI pipeline runs tests, builds both
variants, smoke-tests the running container, and publishes on every push. Model
weights are fetched from Google Drive at build time so the git repository stays
small.

**Stack.** React 18 + Vite + Tailwind (frontend); FastAPI + Python 3.11
(backend); PaddleOCR 3.0, PyTorch 2.5, Transformers 4.46, OpenCV 4.10 (ML/CV);
Docker + GitHub Actions (delivery).

---

# Slide Deck

---

## Slide 1 — The Problem and the Product

**Title:** RenAIssance: Reading What Machines Cannot

**The problem.**
- Historical books hold centuries of knowledge that is effectively unsearchable.
- Early-modern Spanish print breaks every assumption modern OCR makes: worn and
  broken glyphs, ink bleed-through, non-standard spelling and abbreviations,
  warped and skewed pages, decorative typography, and inconsistent scans.
- General-purpose OCR (Tesseract, stock cloud OCR) produces error rates too high
  to be useful for scholars.

**The product.**
- A full-stack web app that takes a PDF or image and produces clean, editable,
  exportable text.
- Designed for domain experts, not engineers: everything happens in a guided
  five-step wizard in the browser.
- **Upload → Select pages → Preprocess → Text detection → OCR & export.**
- Output formats: TXT, DOCX, PDF.

**Talking point.** The project is not a single model. It is a *pipeline* where
each stage removes a different class of error, and where the user stays in
control at every step.

> [IMAGE: The five-step wizard, one screenshot per step, left to right.]

---

## Slide 2 — System Architecture

**Title:** One Interface, Many Engines

**Frontend (React 18 + Vite + Tailwind).**
- A single-page wizard that manages global state and routes between the five
  steps.
- PDF.js renders and slices uploaded PDFs in the browser.
- Live before/after preview during preprocessing.

**Backend (FastAPI + Python 3.11).**
- Ten routers: health, OCR, preprocess, export, layout detection, dataset,
  recognition, LLM post-process, storage, auth.
- **Strategy + Factory pattern** for OCR engines: one `BaseOCRProvider`
  interface, an `OCRFactory` registry, and independent implementations behind
  it. Adding an engine means implementing the interface and registering it,
  nothing else changes.
- API keys are passed per request from the browser and never stored on disk.

**The provider factory (key design idea).**
- The application talks to *one* interface.
- Behind it sit Gemini, ChatGPT, the local CRNN, and the local TrOCR.
- DeepSeek and Qwen are implemented and registered but hidden in the UI, ready
  to enable.

> [IMAGE: provider-factory.svg — Application → OCRFactory → {Gemini, ChatGPT,
> Local CRNN, Local TrOCR}, with DeepSeek/Qwen greyed behind.]

**Talking point.** This is what makes the system future-proof: new OCR or
correction engines plug in without touching the wizard, the API, or the other
engines.

---

## Slide 3 — Stage 1: Preprocessing

**Title:** Cleaning the Page Before the Machine Reads It

**A configurable OpenCV pipeline.**
- Ten operations, each toggleable and tunable, run in sequence with a live
  before/after preview: normalize, grayscale, deskew, denoise, contrast,
  sharpen, threshold, morphology, remove blobs, remove noise (speckles).

**Highlight 1 — Piecewise deskew.**
- Standard deskew rotates the whole page by one angle. Real books warp: the top
  of a page can be straight while the bottom is skewed.
- The new deskew measures skew in horizontal **bands** using projection
  profiles. If the bands agree (spread below 1°) it rotates once; if they
  disagree it blends per-band rotations seam-free, using triangular weights that
  form a partition of unity so there are no visible discontinuities.
- Modes: auto / global / piecewise, with a bands control.

**Highlight 2 — Book-type presets.**
- A "Recommendation" dropdown offers pretested pipelines per book type.
- Selecting a type *loads* the pipeline into the editor without running it, so
  the user can inspect and tune it before applying.
- First shipped preset: **PORCONES → Remove Ink Speckles (20px).**

**Talking point.** Preprocessing is where most OCR wins or loses. Giving experts
a tuned starting point plus live feedback is worth more than any single clever
filter.

> [IMAGE: The three-panel preprocess editor with a warped page corrected by
> piecewise deskew, before on the left, after on the right.]

---

## Slide 4 — Stage 2: Text Detection

**Title:** Finding the Text, Adapting to the Machine

**Layout-aware detection on PaddleOCR PP-OCRv5.**
- Detects text regions per page before recognition, so recognition runs on clean
  line/word crops rather than the whole page.
- The detector was fine-tuned on hand-labelled page bounding boxes from the
  target books (PP-OCRv5 server detector), improving region proposals on
  dense historical layouts.

**Adaptive model-tier selection (engineering highlight).**
- At runtime the system reads free VRAM/RAM and picks a tier automatically:
  - Enough free VRAM → **server** models on GPU (best accuracy).
  - Less → **mobile** models on GPU (smaller footprint).
  - No usable GPU → **CPU**, still fully functional.
- The same code runs on a workstation with an RTX GPU and on a Mac laptop with
  no GPU at all.

**Talking point.** The system does not assume a fixed machine. It measures the
hardware it wakes up on and scales the models to fit, which is what lets one
Docker image serve both a lab GPU box and a reviewer's laptop.

> [IMAGE: A detected page with bounding boxes overlaid on each text line.]

---

## Slide 5 — Stage 3: Text Recognition and Results

**Title:** Two Local Models, Measured Honestly

**Two locally fine-tuned recognizers (plus cloud options).**
- **CRNN (CNN-LSTM):** a lightweight ResNet CNN backbone → BiLSTM → CTC decoder,
  fine-tuned on a Spanish line dataset. Fast, small, strong on clean printed
  lines.
- **TrOCR:** a vision-encoder / text-decoder transformer fine-tuned from
  `microsoft/trocr-base-printed`. Heavier but more robust on degraded text.
- Cloud vision models (Gemini, ChatGPT) available through the same interface.

**Line-level accuracy on held-out books** (document-level 10% split, same books
held out across all experiments):

| Model | CER | WER |
|---|---|---|
| CRNN (fine-tuned CNN-LSTM) | ~2.0% | ~9.0% |
| TrOCR (fine-tuned, fp32) | 6.1% | 21.7% |

*Source: training/validation logs, notebooks 01 and 03; TrOCR fp32 from the
quantization study, notebook 04.*

**TrOCR quantization study** (post-training, notebook 04) — accuracy vs memory
vs latency:

| Precision | CER | WER | Size | vs fp32 size | ms/line |
|---|---|---|---|---|---|
| fp32 | 6.11% | 21.65% | 1274 MB | 100% | 197 |
| **fp16** | **6.08%** | **21.59%** | **637 MB** | **50%** | **63** |
| int8 | 6.29% | 21.62% | 471 MB | 37% | 246 |
| 4-bit NF4 | 7.46% | 24.62% | 337 MB | 26% | 127 |
| 4-bit FP4 | 9.25% | 29.64% | 337 MB | 26% | 126 |

**Takeaway.** fp16 is a free win: half the memory, roughly 3× faster, and CER
actually a hair *better* than fp32. 4-bit shrinks the model to a quarter but
costs real accuracy, so it is only worth it under hard memory limits.

> [IMAGE: Bar chart of the quantization table — CER and size per precision.]

---

## Slide 6 — Stage 4: AI Post-Processing

**Title:** Teaching a Language Model to Fix OCR

**The idea.**
- Even the best recognizer leaves residual errors: a confused character, a split
  or merged word, a wrong diacritic.
- A language model that knows Spanish and knows *how OCR fails* can correct these
  from context.

**What was built.**
- `gemma-3-4b-it` fine-tuned with **QLoRA** (4-bit NF4 base, LoRA rank 16) into a
  line-by-line Spanish OCR corrector.
- Trained on OCR-vs-ground-truth pairs generated from **two** engines (the
  pretrained Paddle recognizer and the fine-tuned CRNN), so it learns to fix real
  model mistakes rather than synthetic noise.
- Ships as a small LoRA adapter on top of the base model; runs locally on an
  NVIDIA GPU.

**Results — the corrector nearly halves word error:**

| Stage | CER | WER |
|---|---|---|
| Raw OCR | 4.65% | 20.9% |
| **+ Fine-tuned Gemma corrector** | **3.2%** | **11.5%** |

- Relative reduction: **CER −31%, WER −45%.**
- **Critical finding:** the *off-the-shelf* (zero-shot) model made results
  *worse* — it rewrote text it did not understand. Fine-tuning on real OCR error
  pairs is what turned it into a corrector instead of a paraphraser.

**Also available as cloud correctors:** Gemini, OpenAI, DeepSeek, Qwen, through
the same post-processing interface.

> [IMAGE: Before/after text sample — raw OCR line above, corrected line below,
> with the fixes highlighted.]

**Talking point.** This is the difference between "a big model" and "the right
model." A general LLM hurt accuracy; the same model, fine-tuned on the actual
failure modes, delivered the single largest error reduction in the pipeline.

---

## Slide 7 — Delivery, Impact, and Takeaways

**Title:** From Research Notebooks to a Product Anyone Can Run

**Packaged for real use.**
- One-command launcher (`./run.sh` / `run.ps1`) auto-detects the GPU and starts
  the correct image variant.
- A single Dockerfile builds **both** GPU and CPU images; the app picks
  CUDA / Apple MPS / CPU at runtime, so it runs on a lab GPU box, a Windows PC,
  or a Mac laptop unchanged.
- CI on every push: run tests → build GPU + CPU images → smoke-test the live
  container's health endpoint → publish images and tag only on full success.
- Model weights are fetched from Google Drive at build time, keeping the git
  repo small.

**What the numbers add up to.**
- Preprocessing removes skew and noise before the model ever sees the page.
- A fine-tuned detector + adaptive PP-OCRv5 finds the text on any hardware.
- The fine-tuned CRNN reaches ~2% CER on clean lines; TrOCR is the robust
  fallback, and fp16 makes it cheap to run.
- The Gemma corrector cuts word error nearly in half on top of that.
- Each stage attacks a different error source, and they compound.

**Engineering takeaways.**
- Design for pluggability (Factory pattern) so new engines cost nothing to add.
- Measure the hardware at runtime instead of assuming it.
- Bigger models are not automatically better — fine-tuning on the real failure
  distribution is what wins.
- Reproducibility and one-command deployment are features, not afterthoughts.

**Closing line.** RenAIssance turns pages that machines could not read into
searchable, editable text, and it does so as a product that a historian can run
on a laptop, not a script that only works in the author's notebook.

> [IMAGE: The final OCR & Export screen with a finished transcription and the
> TXT / DOCX / PDF export buttons.]

---

## Appendix — Metric Provenance (for Q&A)

- **CRNN line-level (~2.0% CER / ~9.0% WER):** best validation checkpoint,
  notebook 01 pipeline run and notebook 03 recognition fine-tuning (fine-tunes
  `latin_PP-OCRv5_mobile_rec`). Training CER reached ~1.3%.
- **TrOCR (6.1% CER / 21.7% WER):** fp32 baseline in the quantization study,
  notebook 04; consistent with notebook 01 best validation CER of ~7%.
- **Quantization table:** notebook 04 stored outputs, verbatim (fp16 / int8 /
  4-bit FP4 / 4-bit NF4 vs fp32, on the same validation lines).
- **Gemma corrector (CER 4.65% → 3.2%, WER 20.9% → 11.5%):** post-processing
  fine-tuning experiment, notebook 05 (QLoRA `gemma-3-4b-it`), evaluated raw vs
  zero-shot vs fine-tuned per engine.
- **Evaluation protocol:** document-level `GroupShuffleSplit(test_size=0.10,
  random_state=42)` so the same books are held out across every experiment; CER
  and WER computed by `src/evals/metrics.py`.

*Note: the CRNN and TrOCR error rates are line-level on held-out lines. A single
end-to-end full-page number is much higher for any engine because it compounds
detection, reading-order, and recognition errors across a whole page; quote the
line-level figures for model quality and describe the full pipeline
qualitatively.*
