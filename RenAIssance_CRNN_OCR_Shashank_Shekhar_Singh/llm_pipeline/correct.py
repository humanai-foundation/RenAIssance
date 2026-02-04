"""
LLM Correction Module using OpenAI ChatGPT API

This module handles the correction of OCR errors in 17th-century Spanish texts
using OpenAI's ChatGPT with carefully crafted prompts.
"""

import os

# Handle OpenAI import
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("Warning: openai not installed. Run: pip install openai")


def get_openai_client(api_key=None):
    """
    Get OpenAI client with the provided API key.
    """
    if OpenAI is None:
        raise ImportError("openai is required. Install with: pip install openai")
    
    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not provided. Either pass it as argument or set OPENAI_API_KEY environment variable."
        )
    
    return OpenAI(api_key=api_key)


def create_correction_prompt(noisy_text):
    """
    Create a detailed system and user prompt for ChatGPT to correct OCR errors.
    """
    system_prompt = """You are an expert in 17th-century Spanish paleography and OCR error correction.

Your task is to correct OCR errors in historical Spanish texts while:
- Preserving the historical Spanish language style
- Fixing only the OCR errors, not the original 17th-century spelling variations
- Maintaining the original formatting and structure
- Being conservative: only change what is clearly an error

KNOWN ERROR PATTERNS:
1. s/f confusion: 's' and 'f' are often confused, especially within words
2. u/v confusion: 'u' and 'v' were used interchangeably in historical texts
3. Tilde errors: Missing or incorrect tildes over n, or missing 'n' after vowels
4. Old spellings: 'c cedilla' used instead of modern 'z'

OUTPUT ONLY THE CORRECTED TEXT, WITHOUT ANY EXPLANATIONS."""

    user_prompt = f"""Please correct the OCR errors in this 17th-century Spanish text:

{noisy_text}"""
    
    return system_prompt, user_prompt


def correct_text_with_chatgpt(noisy_text, api_key=None, model_name="gpt-3.5-turbo", chunk_size=2000):
    """
    Correct OCR errors using OpenAI ChatGPT API.
    """
    client = get_openai_client(api_key)
    
    # Split text into chunks if needed
    chunks = []
    text_lines = noisy_text.split('\n')
    current_chunk = []
    current_length = 0
    
    for line in text_lines:
        line_length = len(line) + 1
        if current_length + line_length > chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_length = line_length
        else:
            current_chunk.append(line)
            current_length += line_length
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    print(f"Processing {len(chunks)} chunk(s) with ChatGPT...")
    
    # Process each chunk
    corrected_chunks = []
    
    for i, chunk in enumerate(chunks, 1):
        print(f"   Processing chunk {i}/{len(chunks)}... ", end='', flush=True)
        
        system_prompt, user_prompt = create_correction_prompt(chunk)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3  # Lower temperature for more consistent corrections
        )
        
        corrected_text = response.choices[0].message.content.strip()
        corrected_chunks.append(corrected_text)
        
        print("Done")
    
    # Combine all chunks
    final_corrected = '\n\n'.join(corrected_chunks)
    
    return final_corrected


# Alias for backward compatibility
correct_text_with_gemini = correct_text_with_chatgpt


def correct_text_batch(text_list, api_key=None, model_name="gpt-3.5-turbo"):
    """
    Correct multiple texts in batch.
    """
    corrected_list = []
    for i, text in enumerate(text_list, 1):
        print(f"Processing text {i}/{len(text_list)}...")
        corrected = correct_text_with_chatgpt(text, api_key, model_name)
        corrected_list.append(corrected)
    
    return corrected_list


if __name__ == "__main__":
    # Test with a sample noisy text
    sample_noisy = """En efte libro fe trata de la noblefa uirtvofa, 
y de los medios para alcançarla. Es obra de gran vtilidad 
para todos los que defean fer nobles y uirtvolos."""
    
    print("Testing ChatGPT Correction Module")
    print("="*60)
    print("\nInput (noisy):")
    print(sample_noisy)
    print("\n" + "="*60)
    
    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("\nError: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
    else:
        print("\nSending to ChatGPT for correction...")
        try:
            corrected = correct_text_with_chatgpt(sample_noisy, api_key)
            print("\nCorrected Output:")
            print(corrected)
            print("\n" + "="*60)
        except Exception as e:
            print(f"\nError: {e}")
