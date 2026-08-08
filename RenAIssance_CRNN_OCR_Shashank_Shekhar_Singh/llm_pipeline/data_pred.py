"""
Data Preparation Module for LLM-Enhanced OCR Pipeline

This module handles:
1. Extraction of ground truth text from DOCX transcription files
2. Generation of synthetic OCR noise to simulate realistic errors
3. Preparation of data for LLM correction evaluation
"""

import os
import random
import re

# Handle docx import
try:
    from docx import Document
except ImportError:
    Document = None
    print("Warning: python-docx not installed. Run: pip install python-docx")


def extract_text_from_docx(docx_path):
    """
    Extract plain text from a DOCX file.
    """
    if Document is None:
        raise ImportError("python-docx is required. Install with: pip install python-docx")
    
    doc = Document(docx_path)
    text_lines = []
    
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        # Skip lines starting with "PDF p" (page markers)
        if text and not text.startswith("PDF p"):
            text_lines.append(text)
    
    return "\n".join(text_lines)


def load_ground_truth(data_dir="../data"):
    """
    Load ground truth transcriptions from DOCX files.
    """
    virtuosa_path = os.path.join(data_dir, "Padilla_Nobleza_virtuosa_testTranscription.docx")
    perfecto_path = os.path.join(data_dir, "Padilla - 2 Noble perfecto_Transcription.docx")
    
    virtuosa_text = ""
    perfecto_text = ""
    
    if os.path.exists(virtuosa_path):
        virtuosa_text = extract_text_from_docx(virtuosa_path)
    if os.path.exists(perfecto_path):
        perfecto_text = extract_text_from_docx(perfecto_path)
    
    return virtuosa_text, perfecto_text


def inject_s_f_swap(text, swap_probability=0.15):
    """
    Swap 's' and 'f' characters based on context (17th-century Spanish pattern).
    """
    result = []
    words = text.split()
    
    for word in words:
        new_word = list(word)
        for i, char in enumerate(word):
            # Only swap if not at beginning or end of word
            if i > 0 and i < len(word) - 1:
                if random.random() < swap_probability:
                    if char.lower() == 's':
                        new_word[i] = 'f' if char.islower() else 'F'
                    elif char.lower() == 'f':
                        new_word[i] = 's' if char.islower() else 'S'
        result.append(''.join(new_word))
    
    return ' '.join(result)


def inject_u_v_swap(text, swap_probability=0.1):
    """
    Swap 'u' and 'v' characters (common in historical texts).
    """
    result = []
    for char in text:
        if random.random() < swap_probability:
            if char.lower() == 'u':
                result.append('v' if char.islower() else 'V')
            elif char.lower() == 'v':
                result.append('u' if char.islower() else 'U')
            else:
                result.append(char)
        else:
            result.append(char)
    
    return ''.join(result)


def inject_old_spelling(text, swap_probability=0.2):
    """
    Replace modern 'z' with old spelling 'ç'.
    """
    result = []
    for char in text:
        if char.lower() == 'z' and random.random() < swap_probability:
            result.append('ç' if char.islower() else 'Ç')
        else:
            result.append(char)
    
    return ''.join(result)


def inject_tilde_errors(text, error_probability=0.08):
    """
    Simulate tilde misreading errors.
    """
    # ñ → n
    def replace_n_tilde(match):
        return 'n' if random.random() < error_probability else 'ñ'
    
    def replace_N_tilde(match):
        return 'N' if random.random() < error_probability else 'Ñ'
    
    text = re.sub(r'ñ', replace_n_tilde, text)
    text = re.sub(r'Ñ', replace_N_tilde, text)
    
    # Randomly drop 'n' after vowels
    vowels = 'aeiouAEIOU'
    result = []
    i = 0
    while i < len(text):
        if i < len(text) - 1 and text[i] in vowels and text[i+1] in 'nN':
            if random.random() < error_probability:
                result.append(text[i])
                i += 2
            else:
                result.append(text[i])
                result.append(text[i+1])
                i += 2
        else:
            result.append(text[i])
            i += 1
    
    return ''.join(result)


def generate_noisy_text(ground_truth, s_f_prob=0.15, u_v_prob=0.1, 
                        old_spelling_prob=0.2, tilde_prob=0.08):
    """
    Generate synthetic OCR noise by applying all error patterns.
    """
    noisy = ground_truth
    noisy = inject_s_f_swap(noisy, s_f_prob)
    noisy = inject_u_v_swap(noisy, u_v_prob)
    noisy = inject_old_spelling(noisy, old_spelling_prob)
    noisy = inject_tilde_errors(noisy, tilde_prob)
    
    return noisy


def prepare_dataset(data_dir="../data"):
    """
    Main function to prepare ground truth and noisy data.
    """
    print("Loading ground truth transcriptions...")
    virtuosa_text, perfecto_text = load_ground_truth(data_dir)
    
    # Combine both texts
    combined_gt = virtuosa_text + "\n\n" + perfecto_text
    
    print(f"Loaded {len(combined_gt)} characters of ground truth")
    print("\nGenerating synthetic OCR noise...")
    
    noisy_text = generate_noisy_text(combined_gt)
    
    print(f"Generated {len(noisy_text)} characters of noisy text")
    
    # Calculate how many characters changed
    differences = sum(1 for a, b in zip(combined_gt, noisy_text) if a != b)
    noise_rate = (differences / len(combined_gt)) * 100 if combined_gt else 0
    
    print(f"Synthetic noise rate: {noise_rate:.2f}%")
    
    return combined_gt, noisy_text


if __name__ == "__main__":
    # Test the data preparation
    gt, noisy = prepare_dataset()
    
    print("\n" + "="*60)
    print("SAMPLE OUTPUT:")
    print("="*60)
    
    if gt:
        print("\nGround Truth (first 200 chars):")
        print(gt[0:200])
        print("\nNoisy Text (first 200 chars):")
        print(noisy[0:200])
    else:
        print("No data loaded - check that .docx files exist in data/ folder")
    
    print("\n" + "="*60)
