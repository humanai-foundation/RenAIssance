"""
Main Pipeline for LLM-Enhanced OCR Correction

This script orchestrates the complete workflow:
1. Load ground truth from DOCX files
2. Generate synthetic OCR noise
3. Correct errors using Gemini LLM
4. Evaluate and report results
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Now import from llm_pipeline
from llm_pipeline.data_pred import prepare_dataset
from llm_pipeline.correct import correct_text_with_gemini
from llm_pipeline.evaluate import evaluate_correction, print_evaluation_report


def save_results(ground_truth, noisy, corrected, results, output_dir="llm_output"):
    """
    Save all results to files for later inspection.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save texts
    with open(f"{output_dir}/1_ground_truth.txt", 'w', encoding='utf-8') as f:
        f.write(ground_truth)
    
    with open(f"{output_dir}/2_noisy_text.txt", 'w', encoding='utf-8') as f:
        f.write(noisy)
    
    with open(f"{output_dir}/3_corrected_text.txt", 'w', encoding='utf-8') as f:
        f.write(corrected)
    
    # Save evaluation report
    with open(f"{output_dir}/4_evaluation_report.txt", 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(" "*20 + "EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("BASELINE (Noisy Text vs Ground Truth):\n")
        f.write(f"  Character Error Rate (CER):  {results['baseline']['cer']:.2f}%\n")
        f.write(f"  Word Error Rate (WER):       {results['baseline']['wer']:.2f}%\n")
        f.write(f"  Character Accuracy:          {results['baseline']['accuracy']:.2f}%\n\n")
        
        f.write("AFTER LLM CORRECTION:\n")
        f.write(f"  Character Error Rate (CER):  {results['corrected']['cer']:.2f}%\n")
        f.write(f"  Word Error Rate (WER):       {results['corrected']['wer']:.2f}%\n")
        f.write(f"  Character Accuracy:          {results['corrected']['accuracy']:.2f}%\n\n")
        
        f.write("IMPROVEMENT:\n")
        f.write(f"  CER Reduction:        {results['improvement']['cer']:+.2f}%\n")
        f.write(f"  WER Reduction:        {results['improvement']['wer']:+.2f}%\n")
        f.write(f"  Accuracy Gain:        {results['improvement']['accuracy']:+.2f}%\n\n")
        
        f.write("="*70 + "\n")
        
        if results['improvement']['cer'] > 0:
            f.write("SUCCESS: LLM correction improved the OCR output!\n")
        else:
            f.write("WARNING: LLM correction did not improve results.\n")
    
    print(f"\n Results saved to '{output_dir}/' directory")


def main():
    """
    Main pipeline execution.
    """
    print("\n" + "="*70)
    print(" "*15 + "LLM-ENHANCED OCR PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Prepare data
    print("STEP 1: Data Preparation")
    print("-" * 70)
    
    data_dir = "data"
    ground_truth, noisy_text = prepare_dataset(data_dir)
    
    # Limit to first 3000 chars for faster testing
    MAX_CHARS = 3000
    if len(ground_truth) > MAX_CHARS:
        print(f"\n Limiting to first {MAX_CHARS} characters for testing...")
        ground_truth = ground_truth[:MAX_CHARS]
        noisy_text = noisy_text[:MAX_CHARS]
    
    print("\n" + "="*70 + "\n")
    
    # Step 2: LLM Correction
    print("STEP 2: LLM Correction with Gemini")
    print("-" * 70)
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("\n ERROR: OPENAI_API_KEY environment variable not set!")
        print("\nPlease set it with:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nGet your API key at: https://platform.openai.com/api-keys")
        return
    
    try:
        corrected_text = correct_text_with_gemini(noisy_text, api_key)
    except Exception as e:
        print(f"\n Error during correction: {e}")
        return
    
    print("\n" + "="*70 + "\n")
    
    # Step 3: Evaluation
    print("STEP 3: Evaluation")
    print("-" * 70)
    
    results = evaluate_correction(ground_truth, noisy_text, corrected_text)
    print_evaluation_report(results)
    
    # Step 4: Save results
    print("STEP 4: Saving Results")
    print("-" * 70)
    save_results(ground_truth, noisy_text, corrected_text, results)
    
    print("\n" + "="*70)
    print("Pipeline completed successfully!")
    print("="*70 + "\n")
    
    # Print project goal achievement
    print("PROJECT GOAL STATUS:")
    target_accuracy = 90.0
    achieved_accuracy = results['corrected']['accuracy']
    
    if achieved_accuracy >= target_accuracy:
        print(f"   Target accuracy of {target_accuracy}% ACHIEVED!")
        print(f"   Actual accuracy: {achieved_accuracy:.2f}%")
    else:
        print(f"   Target accuracy: {target_accuracy}%")
        print(f"   Actual accuracy: {achieved_accuracy:.2f}%")
        print(f"   Gap: {target_accuracy - achieved_accuracy:.2f}%")
    
    print("\n")


if __name__ == "__main__":
    main()
