"""The `local_es` corrector: gemma-3-4b-it + our shipped Spanish LoRA adapter.

Runs in 4-bit QLoRA (same config it was trained with, ~3.5 GB VRAM) and corrects
one line at a time, matching training. GPU only — 4-bit needs CUDA.

gemma-3 is gated on HuggingFace, so the first base-weights download needs
HF_TOKEN set and the licence accepted at huggingface.co/google/gemma-3-4b-it.
"""

import os
from pathlib import Path
from threading import Lock

# AutoModelForCausalLM resolves this to the multimodal Gemma3 class; text-only
# generation works fine and matches how the adapter was trained.
FINETUNED_BASE_MODEL = "google/gemma-3-4b-it"
FINETUNED_ADAPTER_SUBDIR = "postprocess/gemma-3-4b-it-lora"

# Must stay byte-identical to the prompt the adapter was fine-tuned with.
_FINETUNED_SYSTEM_PROMPT = (
    "You are an expert in early-modern Spanish (15th-19th century) palaeography "
    "and OCR correction. The user gives you one raw OCR line from a historical "
    "Spanish document. Return ONLY the corrected line, nothing else. "
    "Fix glyph confusions (rn→m, 0→o, 1/I→l, ſ→s), restore diacritics "
    "(á é í ó ú ñ ç), and merge or split wrongly tokenised words. "
    "Preserve the original early-modern spelling: never modernise, translate, "
    "or add content. If the line is already correct, return it unchanged."
)

# (tokenizer, model, device) cache. The lock stops two requests from kicking off
# simultaneous multi-GB loads.
_CACHE: dict = {}
_LOAD_LOCK = Lock()


def _adapter_dir() -> str:
    """Shipped LoRA adapter dir; same path in dev and in the container."""
    backend_root = Path(__file__).resolve().parents[3]
    return str(backend_root / "models" / "weights" / FINETUNED_ADAPTER_SUBDIR)


def base_model_cached() -> bool:
    """True when the gated base weights are already on disk.

    Mirrors layout_detection.models_cached: the UI probes this so it can warn
    about the one-time ~8 GB download instead of looking hung. Filesystem only
    — never imports torch/transformers, so it stays cheap to call.

    huggingface_hub only creates a snapshot symlink once a blob has finished
    downloading, so a half-done shard leaves a dangling/absent entry and
    .exists() correctly reports False.
    """
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    repo = (
        Path(hf_home) / "hub"
        / ("models--" + FINETUNED_BASE_MODEL.replace("/", "--"))
    )
    snapshots = repo / "snapshots"
    if not snapshots.is_dir():
        return False

    for snap in snapshots.iterdir():
        # Sharded checkpoints (gemma-3-4b is 2 shards): every file named in the
        # index must be present, or we would flash "ready" mid-download.
        index = snap / "model.safetensors.index.json"
        if index.is_file():
            try:
                import json

                shards = set(json.loads(index.read_text())["weight_map"].values())
            except (OSError, ValueError, KeyError):
                continue
            if shards and all((snap / s).exists() for s in shards):
                return True
        elif (snap / "model.safetensors").exists():
            return True
    return False


def _load(model_name: str, *, adapter_dir: str | None = None, quantize: bool = False):
    """Load and cache a tokenizer + causal LM.

    adapter_dir applies a local PEFT LoRA on top of the base; quantize loads
    4-bit nf4 when CUDA is present, which keeps the 4B model near 3 GB VRAM.
    """
    key = f"{model_name}::{adapter_dir or ''}::{'q4' if quantize else 'full'}"
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    with _LOAD_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from app.utils.torch_device import select_torch_device

        device = select_torch_device()

        # Tokenizer comes from the BASE model, never the adapter dir. LoRA
        # doesn't touch the vocabulary, and the adapter's tokenizer.json was
        # serialized by a newer `tokenizers` than the one pinned in the image.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # NEVER fp16 here. gemma-3 is bf16-trained and its activations overflow
        # fp16: the logits go NaN, generate() emits nothing but <pad>, and the
        # caller below keeps every original line — a silent no-op that looks
        # like "the model had no corrections to make" rather than a failure.
        # bf16 needs Ampere or newer, so older cards take the fp32 path (slower,
        # but correct).
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            dtype = torch.float32

        load_kwargs: dict = {}
        if quantize and device == "cuda":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            load_kwargs["device_map"] = "auto"
        # torch_dtype, not dtype — the image pins transformers 4.53. Still a
        # valid alias in 5.x, so this works both places.
        load_kwargs["torch_dtype"] = dtype

        # One local LLM resident at a time, evicted BEFORE the new load —
        # otherwise switching providers holds both on an 8 GB GPU and OOMs.
        # Safe because post-processing runs one request at a time.
        _evict_all_except(key)

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        if adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)

        # device_map="auto" already placed the quantized model.
        if "device_map" not in load_kwargs:
            model.to(device)
        model.eval()

        _CACHE[key] = (tokenizer, model, device)
        return _CACHE[key]


def _evict_all_except(keep_key: str) -> None:
    """Free every cached local LLM except `keep_key` (frees VRAM/RAM)."""
    import gc

    import torch

    for k in [k for k in _CACHE if k != keep_key]:
        _, model, device = _CACHE.pop(k)
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def post_process_text_finetuned(
    text: str,
    model: str | None = None,          # ignored; adapter is fixed
    template_name: str | None = None,  # ignored; per-line system prompt is fixed
    batch_size: int = 16,
) -> str:
    """Correct OCR text with the Spanish-finetuned gemma LoRA.

    Line by line, because that's how the adapter was trained — a whole page is
    out of distribution. Blank lines are kept so paragraphs survive. Raises on
    CPU-only hosts; the API providers still work there.
    """
    if not text or not text.strip():
        return text

    import torch

    from app.utils.torch_device import select_torch_device

    if select_torch_device() != "cuda":
        raise ValueError(
            "The local fine-tuned Spanish model requires a GPU (4-bit QLoRA). "
            "Use a cloud provider (Gemini/OpenAI) on CPU-only hosts."
        )

    adapter_dir = _adapter_dir()
    if not os.path.isdir(adapter_dir):
        raise ValueError(f"Fine-tuned adapter not found at {adapter_dir}")

    try:
        tokenizer, llm, device = _load(
            FINETUNED_BASE_MODEL, adapter_dir=adapter_dir, quantize=True
        )
    except Exception as exc:
        # Turn huggingface_hub's GatedRepoError/401 into something actionable.
        msg = str(exc).lower()
        if "gated" in msg or "401" in msg or "authoriz" in msg or "token" in msg:
            raise ValueError(
                f"Could not download the gated base model '{FINETUNED_BASE_MODEL}'. "
                "Set HF_TOKEN in the environment and accept the licence at "
                "huggingface.co/google/gemma-3-4b-it."
            ) from exc
        raise
    tokenizer.padding_side = "left"  # decoder-only batched generation

    # generate() stops a batch only once EVERY row has emitted a stop token, so
    # both knobs below are about not letting one long row bill the other 15.
    stop_ids = llm.generation_config.eos_token_id
    stop_ids = set(stop_ids) if isinstance(stop_ids, (list, tuple)) else {stop_ids}

    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.strip()]
    out = list(lines)
    answered = 0

    # Batch lines of similar length together: rows are left-padded to the
    # longest in their batch, and a batch runs as many steps as its slowest row
    # needs. Mixing a 5-token line with a 60-token one makes the short ones pay
    # for the long one at both ends. Results are written back to out[i] by
    # original index, so the visual order the frontend relies on is untouched.
    line_lens = dict(zip(
        idx,
        (len(t) for t in tokenizer(
            [lines[i] for i in idx], add_special_tokens=False
        ).input_ids),
    ))
    idx.sort(key=line_lens.__getitem__)

    for start in range(0, len(idx), batch_size):
        chunk_idx = idx[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": _FINETUNED_SYSTEM_PROMPT},
                    {"role": "user", "content": lines[i]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for i in chunk_idx
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        # A corrected line runs about as long as the line that went in — this is
        # glyph repair, not translation. Sizing the ceiling to the batch's
        # longest input (+ slack for restored diacritics and split words) beats
        # a flat 160, which one rambling row would otherwise drag all 16 rows to.
        # ponytail: 1.5x + 16 is generous rather than tuned; the guard below
        # catches anything that does overrun, so a too-tight cap can only cost a
        # correction, never truncate a line.
        cap = min(160, int(max(line_lens[i] for i in chunk_idx) * 1.5) + 16)
        with torch.no_grad():
            gen = llm.generate(
                **enc,
                max_new_tokens=cap,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new = gen[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new, skip_special_tokens=True)
        for i, corrected, row in zip(chunk_idx, decoded, new):
            # No stop token means generation was cut off at the cap, so what we
            # have is half a line. Keep the original rather than write a
            # truncated one — a silently chopped line is worse than an uncorrected
            # one, and the frontend has no way to tell them apart.
            if stop_ids.isdisjoint(row.tolist()):
                continue
            # One input line must stay one output line — the frontend maps
            # results back onto boxes by index. Never blank a line out either.
            corrected = " ".join(corrected.split()).strip()
            if corrected:
                out[i] = corrected
                answered += 1

    # Keeping the original line when the model returns nothing is right per
    # line, but if it answered NOTHING at all it is broken (NaN logits from a
    # bad dtype do exactly this) — and silently echoing the input back looks
    # like a clean run to the user. Fail loudly instead.
    if idx and answered == 0:
        raise ValueError(
            "Fine-tuned model returned no output for any line — it may have "
            "failed to load correctly."
        )

    result = "\n".join(out).strip()
    if not result:
        raise ValueError("Fine-tuned model returned an empty response")
    return result
