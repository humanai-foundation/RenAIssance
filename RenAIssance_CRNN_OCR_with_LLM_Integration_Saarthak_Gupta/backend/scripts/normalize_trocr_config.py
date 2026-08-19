"""Drop a known-good processor next to the fine-tuned TrOCR weights.

We fine-tuned from microsoft/trocr-base-printed, so its tokenizer and image
stats match exactly. Copying them in at build time makes the checkpoint load
regardless of which transformers version serialized the original processor.
No-op when the weights aren't present, so dev builds still work.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from transformers import TrOCRProcessor

# The base model the user fine-tuned from (see
# RenAIssanceExperimental/experimentation.ipynb).
BASE_MODEL_ID = "microsoft/trocr-base-printed"

TROCR_DIR = Path("/app/models/weights/trocr")

# Files that belong to the fine-tuned weights — we must not overwrite these
# with the base model's copies.
FINETUNED_ARTIFACTS = {
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "pytorch_model.bin",
}


def main() -> None:
    if not TROCR_DIR.is_dir():
        print(f"[normalize_trocr] {TROCR_DIR} does not exist — skipping")
        return

    weights = TROCR_DIR / "model.safetensors"
    if not weights.is_file():
        print(f"[normalize_trocr] no model.safetensors in {TROCR_DIR} — skipping")
        return

    print(f"[normalize_trocr] pulling processor assets from {BASE_MODEL_ID}")
    processor = TrOCRProcessor.from_pretrained(BASE_MODEL_ID)

    # Save all processor assets (preprocessor_config.json + vocab.json +
    # merges.txt + tokenizer_config.json + special_tokens_map.json +
    # tokenizer.json) into a staging directory, then copy only the files we
    # don't already have — this prevents clobbering the fine-tuned weights.
    staging = TROCR_DIR / ".processor_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    processor.save_pretrained(staging)

    copied: list[str] = []
    replaced: list[str] = []
    for src in staging.iterdir():
        if src.name in FINETUNED_ARTIFACTS:
            continue
        dst = TROCR_DIR / src.name
        if dst.exists():
            replaced.append(src.name)
        else:
            copied.append(src.name)
        shutil.copy2(src, dst)

    shutil.rmtree(staging)

    # The newer-format bundled processor_config.json confuses 4.44.x's loader
    # once preprocessor_config.json is in place ("multiple values for
    # image_processor"). Remove it — the dir is self-consistent without it.
    stale = TROCR_DIR / "processor_config.json"
    if stale.exists():
        stale.unlink()
        replaced.append("processor_config.json (removed)")

    if copied:
        print(f"[normalize_trocr] copied: {sorted(copied)}")
    if replaced:
        print(f"[normalize_trocr] replaced: {sorted(replaced)}")
    print("[normalize_trocr] done")


if __name__ == "__main__":
    main()
