"""
Evaluation Module for OCR Correction

Calculates Character Error Rate (CER) and Word Error Rate (WER)
to measure the effectiveness of LLM-based correction.
"""

from difflib import SequenceMatcher
from typing import Tuple


def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance (edit distance) between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Minimum number of edits needed to transform s1 into s2
    """
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_cer(predicted: str, ground_truth: str) -> float:
    """
    Calculate Character Error Rate (CER).
    
    CER = (Levenshtein Distance) / (Length of Ground Truth)
    
    Args:
        predicted: Predicted/corrected text
        ground_truth: Reference ground truth text
        
    Returns:
        CER as a percentage (0-100)
    """
    if not ground_truth:
        return 0.0
    
    distance = calculate_levenshtein_distance(predicted, ground_truth)
    cer = (distance / len(ground_truth)) * 100
    
    return cer


def calculate_wer(predicted: str, ground_truth: str) -> float:
    """
    Calculate Word Error Rate (WER).
    
    WER = (Substitutions + Insertions + Deletions) / (Total Words in Ground Truth)
    
    Args:
        predicted: Predicted/corrected text
        ground_truth: Reference ground truth text
        
    Returns:
        WER as a percentage (0-100)
    """
    pred_words = predicted.split()
    gt_words = ground_truth.split()
    
    if not gt_words:
        return 0.0
    
    distance = calculate_levenshtein_distance(' '.join(pred_words), ' '.join(gt_words))
    wer = (distance / len(gt_words)) * 100
    
    return wer


def calculate_accuracy(predicted: str, ground_truth: str) -> float:
    """
    Calculate character-level accuracy.
    
    Accuracy = (Matching Characters) / (Total Characters) * 100
    
    Args:
        predicted: Predicted/corrected text
        ground_truth: Reference ground truth text
        
    Returns:
        Accuracy as a percentage (0-100)
    """
    if not ground_truth:
        return 0.0
    
    matcher = SequenceMatcher(None, ground_truth, predicted)
    similarity = matcher.ratio()
    accuracy = similarity * 100
    
    return accuracy


def evaluate_correction(
    ground_truth: str,
    noisy_text: str,
    corrected_text: str
) -> dict:
    """
    Comprehensive evaluation of OCR correction.
    
    Compares:
    1. Noisy text vs Ground truth (baseline)
    2. Corrected text vs Ground truth (after LLM)
    
    Args:
        ground_truth: Clean reference text
        noisy_text: Text with OCR errors
        corrected_text: LLM-corrected text
        
    Returns:
        Dictionary with all metrics and improvements
    """
    # Baseline (noisy vs GT)
    baseline_cer = calculate_cer(noisy_text, ground_truth)
    baseline_wer = calculate_wer(noisy_text, ground_truth)
    baseline_accuracy = calculate_accuracy(noisy_text, ground_truth)
    
    # After correction (corrected vs GT)
    corrected_cer = calculate_cer(corrected_text, ground_truth)
    corrected_wer = calculate_wer(corrected_text, ground_truth)
    corrected_accuracy = calculate_accuracy(corrected_text, ground_truth)
    
    # Improvements
    cer_improvement = baseline_cer - corrected_cer
    wer_improvement = baseline_wer - corrected_wer
    accuracy_improvement = corrected_accuracy - baseline_accuracy
    
    results = {
        'baseline': {
            'cer': baseline_cer,
            'wer': baseline_wer,
            'accuracy': baseline_accuracy
        },
        'corrected': {
            'cer': corrected_cer,
            'wer': corrected_wer,
            'accuracy': corrected_accuracy
        },
        'improvement': {
            'cer': cer_improvement,
            'wer': wer_improvement,
            'accuracy': accuracy_improvement
        }
    }
    
    return results


def print_evaluation_report(results: dict) -> None:
    """
    Print a formatted evaluation report.
    
    Args:
        results: Dictionary from evaluate_correction()
    """
    print("\n" + "="*70)
    print(" "*20 + "📊 EVALUATION REPORT")
    print("="*70)
    
    print("\n📉 BASELINE (Noisy Text vs Ground Truth):")
    print(f"   Character Error Rate (CER):  {results['baseline']['cer']:.2f}%")
    print(f"   Word Error Rate (WER):       {results['baseline']['wer']:.2f}%")
    print(f"   Character Accuracy:          {results['baseline']['accuracy']:.2f}%")
    
    print("\n✨ AFTER LLM CORRECTION (Corrected vs Ground Truth):")
    print(f"   Character Error Rate (CER):  {results['corrected']['cer']:.2f}%")
    print(f"   Word Error Rate (WER):       {results['corrected']['wer']:.2f}%")
    print(f"   Character Accuracy:          {results['corrected']['accuracy']:.2f}%")
    
    print("\n📈 IMPROVEMENT:")
    cer_symbol = "✅" if results['improvement']['cer'] > 0 else "⚠️"
    wer_symbol = "✅" if results['improvement']['wer'] > 0 else "⚠️"
    acc_symbol = "✅" if results['improvement']['accuracy'] > 0 else "⚠️"
    
    print(f"   {cer_symbol} CER Reduction:        {results['improvement']['cer']:+.2f}%")
    print(f"   {wer_symbol} WER Reduction:        {results['improvement']['wer']:+.2f}%")
    print(f"   {acc_symbol} Accuracy Gain:        {results['improvement']['accuracy']:+.2f}%")
    
    print("\n" + "="*70)
    
    # Overall assessment
    if results['improvement']['cer'] > 0:
        print("✅ SUCCESS: LLM correction improved the OCR output!")
    else:
        print("⚠️  WARNING: LLM correction did not improve the results.")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test with sample data
    gt = "En este libro se trata de la nobleza virtuosa"
    noisy = "En efte libro fe trata de la noblefa uirtvofa"
    corrected = "En este libro se trata de la nobleza virtuosa"
    
    print("🧪 Testing Evaluation Module")
    results = evaluate_correction(gt, noisy, corrected)
    print_evaluation_report(results)
