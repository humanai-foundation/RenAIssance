"""Prompt templates for LLM cleanup of OCR text.

Line count is sacred: one line per detected text region, mapped back by
position. Every template gets _LINE_PRESERVATION appended to enforce that.
(local_es skips these — it corrects one line at a time anyway.)
"""

_LINE_PRESERVATION = (
    "\n\nCRITICAL — DO NOT CHANGE THE NUMBER OF LINES:\n"
    "The transcript below has exactly one line per text region detected on the "
    "page. That line count is fixed and correct — it comes from the layout "
    "detection step, not from you. Rules:\n"
    "- Return EXACTLY the same number of lines, in the same order.\n"
    "- Each input line maps to one output line. Correct text only WITHIN a line.\n"
    "- NEVER merge two lines into one, split one line into two, reorder, add, or "
    "delete a line — not even blank lines.\n"
    "- If a line seems to end mid-word or mid-sentence, leave it as-is on its "
    "own line; the next line continues it.\n"
    "- Read the whole page for context (to disambiguate glyphs and words), but "
    "edit each line independently.\n"
    "- Output ONLY the corrected lines, separated by single newlines. No "
    "numbering, no commentary, no code fences.\n\n"
    "Transcript (one line per detected region):\n\n"
)

TEMPLATES = {
    "full_cleanup": {
        "name": "Full Cleanup",
        "description": "Fix spelling, formatting, and OCR artifacts in one pass",
        "prompt": (
            "You are an OCR post-processing assistant for historical Spanish "
            "documents. Clean up the OCR-extracted transcript:\n"
            "- Fix obvious OCR errors and misspellings\n"
            "- Fix incorrect character substitutions (e.g., 'rn' misread as 'm', '1' as 'l', long-s 'ſ' as 's')\n"
            "- Restore diacritics that OCR dropped (á é í ó ú ñ ç)\n"
            "- Normalize inconsistent spacing within a line\n"
            "- Preserve the original early-modern spelling and meaning\n"
            "- Do NOT translate, modernise, rephrase, add or remove content"
        ),
    },
    "spelling_correction": {
        "name": "Spelling Correction",
        "description": "Fix only spelling errors and character misrecognitions",
        "prompt": (
            "You are a spelling correction assistant for OCR output of "
            "historical Spanish documents. Fix ONLY spelling errors and "
            "character misrecognitions. Do not change wording, do not "
            "translate or modernise, do not add or remove content."
        ),
    },
    "formatting": {
        "name": "Format & Structure",
        "description": "Normalize spacing within lines (line count stays fixed)",
        "prompt": (
            "You are a text formatting assistant for OCR output.\n"
            "- Normalize spacing within each line (remove extra spaces, fix missing spaces)\n"
            "- Do NOT change any words or fix spelling\n"
            "- Do NOT add or remove content"
        ),
    },
    "historical_normalization": {
        "name": "Historical Text Normalization",
        "description": "Normalize archaic spellings and historical typography",
        "prompt": (
            "You are a historical text normalization assistant.\n"
            "- Normalize long-s (ſ) to modern 's'\n"
            "- Expand common ligatures (ﬀ→ff, ﬁ→fi, ﬂ→fl, ﬃ→ffi, ﬄ→ffl)\n"
            "- Normalize archaic letter forms while preserving meaning\n"
            "- Fix OCR errors specific to historical typefaces\n"
            "- Do NOT modernise vocabulary or grammar"
        ),
    },
}


def get_template(name: str) -> str:
    """Template prompt + the line contract. Callers append the transcript after."""
    entry = TEMPLATES.get(name, TEMPLATES["full_cleanup"])
    return entry["prompt"] + _LINE_PRESERVATION


def list_templates() -> list:
    """Template metadata for the UI dropdown."""
    return [
        {"id": key, "name": val["name"], "description": val["description"]}
        for key, val in TEMPLATES.items()
    ]
