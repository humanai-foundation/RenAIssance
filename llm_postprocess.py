"""
RenAIssance OCR — LLM Post-Processing Module
Issue #51 : Add LLM post-processing for better OCR accuracy
Author   : Abhiram G (abhiram123467)
GSoC 2026 | HumanAI Foundation
"""

import os
import re
import google.generativeai as genai


# ── Configuration ──────────────────────────────────────────────
GEMINI_MODEL = "gemini-1.5-flash"
API_KEY      = os.environ.get("GEMINI_API_KEY", "")


# ── Prompt Template ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert in 17th-century Spanish historical documents.
You will receive raw OCR output from a CRNN model that may contain:
- Misread archaic letterforms (e.g. long-s mistaken for f)
- Incorrect diacritics
- Garbled rare characters
- Minor spacing errors

Your task:
1. Correct ONLY clear OCR errors
2. Preserve original archaic Spanish spelling (do NOT modernize)
3. Preserve original punctuation and line breaks
4. Return ONLY the corrected text — no explanations

Raw OCR text to correct:
"""


# ── CER Calculation ─────────────────────────────────────────────
def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate (CER).
    CER = (Substitutions + Deletions + Insertions) / len(reference)
    """
    ref = list(reference)
    hyp = list(hypothesis)

    # Build DP matrix
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,       # deletion
                d[i][j - 1] + 1,       # insertion
                d[i - 1][j - 1] + cost # substitution
            )

    return d[len(ref)][len(hyp)] / max(len(ref), 1)


# ── Clean Raw OCR Output ────────────────────────────────────────
def clean_raw_ocr(raw_text: str) -> str:
    """
    Basic cleaning of raw CRNN output before sending to LLM.
    - Remove repeated spaces
    - Remove non-printable characters
    """
    text = re.sub(r'[^\x20-\x7E\xA0-\xFF]', '', raw_text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


# ── Main Post-Processing Function ───────────────────────────────
def postprocess_ocr(raw_ocr_text: str, api_key: str = None) -> dict:
    """
    Post-process raw OCR output using Gemini LLM.

    Args:
        raw_ocr_text : Raw text from CRNN model
        api_key      : Gemini API key (or set GEMINI_API_KEY env var)

    Returns:
        dict with keys:
            - raw_text       : original CRNN output
            - cleaned_text   : after basic cleaning
            - corrected_text : after LLM correction
            - cer_before     : CER before LLM (vs cleaned)
            - cer_after      : CER after LLM (vs cleaned)
    """
    # Setup API
    key = api_key or API_KEY
    if not key:
        raise ValueError(
            "Gemini API key not found. "
            "Set GEMINI_API_KEY environment variable or pass api_key argument."
        )
    genai.configure(api_key=key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    # Step 1 — Clean raw OCR
    cleaned = clean_raw_ocr(raw_ocr_text)

    # Step 2 — Send to Gemini
    prompt   = SYSTEM_PROMPT + cleaned
    response = model.generate_content(prompt)
    corrected = response.text.strip()

    # Step 3 — Compute CER improvement
    cer_before = compute_cer(cleaned, cleaned)          # baseline = 0
    cer_after  = compute_cer(cleaned, corrected)        # how much LLM changed

    return {
        "raw_text"       : raw_ocr_text,
        "cleaned_text"   : cleaned,
        "corrected_text" : corrected,
        "cer_before"     : round(cer_before, 4),
        "cer_after"      : round(cer_after, 4),
    }


# ── Batch Processing ────────────────────────────────────────────
def batch_postprocess(ocr_texts: list, api_key: str = None) -> list:
    """
    Post-process a list of raw OCR strings.

    Args:
        ocr_texts : list of raw OCR strings
        api_key   : Gemini API key

    Returns:
        list of result dicts (same as postprocess_ocr)
    """
    results = []
    for i, text in enumerate(ocr_texts):
        print(f"Processing {i + 1}/{len(ocr_texts)}...")
        try:
            result = postprocess_ocr(text, api_key)
            results.append(result)
        except Exception as e:
            print(f"Error on text {i + 1}: {e}")
            results.append({"raw_text": text, "error": str(e)})
    return results


# ── Demo ────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Example raw CRNN output from historical Spanish document
    sample_raw_ocr = """
    Efta es la inftruccion que fe ha de dar
    a los que van a las Indias para que
    guarden y cumplan lo que les efta
    mandado por fu Mageftad.
    """

    print("=" * 55)
    print("  RenAIssance OCR — LLM Post-Processing Demo")
    print("=" * 55)
    print(f"\n RAW OCR OUTPUT:\n{sample_raw_ocr}")

    # Run post-processing
    # Replace with your actual API key or set env var
    result = postprocess_ocr(
        raw_ocr_text=sample_raw_ocr,
        api_key="YOUR_GEMINI_API_KEY_HERE"
    )

    print(f"\n CLEANED TEXT:\n{result['cleaned_text']}")
    print(f"\n CORRECTED TEXT:\n{result['corrected_text']}")
    print(f"\n CER Before LLM : {result['cer_before']}")
    print(f" CER After LLM  : {result['cer_after']}")
    print("=" * 55)
