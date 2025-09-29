# spell_correction_eval.py

"""
A complete Python script to evaluate the 'google/gemma-2-9b-it' model 
on a spelling correction task.

This script performs the following steps:
1.  Sets up the environment and generates a sample dataset if not present.
2.  Loads the specified Hugging Face model and tokenizer, optimizing for GPU.
3.  Reads a TSV dataset containing misspelled words ('input') and their correct versions ('target').
4.  Iterates through each word, prompting the model for a spelling correction.
5.  Cleans and parses the model's output to extract the predicted word.
6.  Saves the 'input', 'predicted', and 'target' triplets to a new TSV file.
7.  Calculates and prints the final Accuracy and average Character Error Rate (CER).

Prerequisites:
pip install torch transformers pandas tqdm editdistance accelerate bitsandbytes
"""

import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import editdistance
import warnings
# Suppress a specific warning from the transformers library for cleaner output
warnings.filterwarnings(
    "ignore", 
    message=".*A new version of the model is available.*", 
    category=UserWarning
)
from huggingface_hub import login
login(token="HF_TOKEN")
# --- Configuration ---
MODEL_ID = "google/gemma-2-9b-it"
INPUT_TSV = r"C:\Users\prana\Downloads\Synthetic_Noise\Synthetic_Noisy_Data_GSoC25\train.tsv"
OUTPUT_TSV = "predictions.tsv"
PROMPT_TEMPLATE = "Correct the spelling of the word: {word}"

# --- Helper Function to Create Dummy Data ---
def create_dummy_dataset_if_not_exists():
    """Creates a sample 'spell_data.tsv' if it doesn't exist."""
    if not os.path.exists(INPUT_TSV):
        print(f"'{INPUT_TSV}' not found. Creating a dummy dataset for demonstration.")
        data = {
            'input': ['accomodate', 'definately', 'wierd', 'goverment', 'seperate', 'untill', 'publically', 'suprise', 'procede', 'existance'],
            'target': ['accommodate', 'definitely', 'weird', 'government', 'separate', 'until', 'publicly', 'surprise', 'proceed', 'existence'],
            'source_rule': ['dummy'] * 10,
            'freq': [1] * 10,
            'edit_count': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'seed': [42] * 10,
            'freq_bucket': ['low'] * 10
        }
        df = pd.DataFrame(data)
        df.to_csv(INPUT_TSV, sep='\t', index=False)
        print("Dummy dataset created successfully.")

# --- Core Functions ---

def load_model_and_tokenizer():
    """Loads the model and tokenizer with GPU and bfloat16 optimization."""
    print(f"Loading model: {MODEL_ID}")
    
    # 1. Setup device and data type
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.bfloat16
        print("✅ GPU detected. Using bfloat16 for performance.")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print("⚠️ No GPU detected. Running on CPU (this will be very slow).")

    # 2. Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map="auto" # Automatically maps model layers to available devices
        )
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please ensure you have accepted the model's license on Hugging Face and are logged in via `huggingface-cli login`.")
        exit(1)
        
    return model, tokenizer, device

def get_spelling_correction(model, tokenizer, device, word: str) -> str:
    """
    Generates a spelling correction for a single word.

    Args:
        model: The loaded transformer model.
        tokenizer: The loaded tokenizer.
        device: The device to run inference on ('cuda' or 'cpu').
        word: The input word with a potential spelling error.

    Returns:
        The cleaned, predicted corrected word.
    """
    prompt = PROMPT_TEMPLATE.format(word=word)
    
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate a response from the model
    # We limit the generation to a few tokens to get just the corrected word.
    outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the full output
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # --- Response Cleaning ---
    # The model's output will contain the original prompt. We need to remove it.
    # Example full_response: "Correct the spelling of the word: accomodate\naccommodate"
    # We only want the part *after* the prompt.
    response_only = full_response[len(prompt):].strip()
    
    # Take only the first word of the response to avoid any explanations.
    predicted_word = response_only.split()[0] if response_only.split() else ""
    
    return predicted_word

def calculate_cer(predicted: str, target: str) -> float:
    """
    Calculates the Character Error Rate (CER).
    CER is defined as the Levenshtein distance divided by the length of the target string.
    """
    if not target:  # Avoid division by zero for empty target strings
        return 1.0 if predicted else 0.0
    
    distance = editdistance.eval(predicted, target)
    return distance / len(target)

# --- Main Execution ---
def main():
    """Main function to run the entire evaluation pipeline."""
    
    # Ensure the dataset exists
    create_dummy_dataset_if_not_exists()
    
    # 1. Load Model and Tokenizer
    model, tokenizer, device = load_model_and_tokenizer()
    
    # 2. Load Dataset
    print(f"Loading dataset from '{INPUT_TSV}'...")
    try:
        df = pd.read_csv(INPUT_TSV, sep='\t')
        # Ensure required columns exist
        if 'input' not in df.columns or 'target' not in df.columns:
            print("❌ Error: Dataset must contain 'input' and 'target' columns.")
            exit(1)
        print(f"Dataset loaded with {len(df)} rows.")
    except FileNotFoundError:
        print(f"❌ Error: The file '{INPUT_TSV}' was not found.")
        exit(1)

    # 3. Process each row and get predictions
    results = []
    
    # Use tqdm for a nice progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc="🤖 Correcting spellings"):
        input_word = str(row['input'])
        target_word = str(row['target'])
        
        predicted_word = get_spelling_correction(model, tokenizer, device, input_word)
        
        results.append({
            "input": input_word,
            "predicted": predicted_word,
            "target": target_word
        })
        
    # 4. Save predictions to a new TSV
    print(f"Saving predictions to '{OUTPUT_TSV}'...")
    predictions_df = pd.DataFrame(results)
    predictions_df.to_csv(OUTPUT_TSV, sep='\t', index=False)
    print("Predictions saved successfully.")

    # 5. Compute Evaluation Metrics
    print("\n--- Evaluation Metrics ---")
    
    # Accuracy
    correct_predictions = (predictions_df['predicted'] == predictions_df['target']).sum()
    total_predictions = len(predictions_df)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Character Error Rate (CER)
    predictions_df['cer'] = predictions_df.apply(
        lambda row: calculate_cer(row['predicted'], row['target']), 
        axis=1
    )
    average_cer = predictions_df['cer'].mean() if total_predictions > 0 else 0.0
    
    # 6. Print final results
    print(f"Total words evaluated: {total_predictions}")
    print(f"Correct predictions:   {correct_predictions}")
    print(f"Accuracy:              {accuracy:.4f}")
    print(f"Average CER:           {average_cer:.4f}")
    print("\nEvaluation complete. ✨")


if __name__ == "__main__":
    main()