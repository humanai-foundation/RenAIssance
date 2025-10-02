import os
import time
import glob
import random
import pandas as pd
import numpy as np
from PIL import Image
import jiwer
import Levenshtein
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from optimum.onnxruntime import ORTModelForVision2Seq

# Set random seed for reproducibility
random.seed(27)

plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11
})

# Paths configuration
model_path = "../../models"  # Original model path (for processor)
quantized_model_path = "../../models/trocr_quantized"  # Quantized model path
image_folder = "../../data/test/line_segments"  # Folder with line images
gt_folder = "../../data/test/line_texts"  # Folder with ground truth text files
num_samples = 100  # Number of random samples to evaluate
results_dir = "comparison_results"
os.makedirs(results_dir, exist_ok=True)

def count_parameters(model):
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
# Function to calculate CER
def calculate_cer(reference, hypothesis):
    """Calculate character error rate using Levenshtein distance."""
    if len(reference) == 0:
        return 0 if len(hypothesis) == 0 else 1
        
    # Calculate edit distance at character level
    edit_distance = Levenshtein.distance(reference, hypothesis)
    return edit_distance / len(reference)

# Function to perform OCR with PyTorch model
def perform_ocr_pytorch(model, processor, image_path):
    """Perform OCR on an image using PyTorch model and measure inference time."""
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Measure inference time
        start_time = time.time()
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(inputs.pixel_values)
        end_time = time.time()
        
        # Decode output
        predicted_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return {
            "text": predicted_text,
            "inference_time": end_time - start_time
        }
    except Exception as e:
        print(f"Error processing {image_path} with PyTorch model: {e}")
        return {
            "text": "",
            "inference_time": 0
        }

# Function to perform OCR with quantized model
def perform_ocr_quantized(model, processor, image_path):
    """Perform OCR on an image using quantized model and measure inference time."""
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Measure inference time
        start_time = time.time()
        inputs = processor(image, return_tensors="pt")
        outputs = model.generate(inputs.pixel_values)
        end_time = time.time()
        
        # Decode output
        predicted_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return {
            "text": predicted_text,
            "inference_time": end_time - start_time
        }
    except Exception as e:
        print(f"Error processing {image_path} with quantized model: {e}")
        return {
            "text": "",
            "inference_time": 0
        }

# Load ground truth texts
def load_ground_truth(gt_folder):
    """Load ground truth texts from text files."""
    gt_dict = {}
    gt_files = glob.glob(os.path.join(gt_folder, "*.txt"))
    
    for gt_file in gt_files:
        basename = os.path.basename(gt_file).split(".")[0]
        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                gt_dict[basename] = f.read().strip()
        except Exception as e:
            print(f"Error reading {gt_file}: {e}")
            gt_dict[basename] = ""
    
    return gt_dict

# Create visualizations in research paper style
def create_visualizations(results_df, save_dir):
    """Create comparison visualizations between models in research paper style."""
    
    # 1. Inference Time Comparison
    plt.figure(figsize=(6, 4))
    model_labels = ['PyTorch Model', 'Quantized Model']
    times = [results_df['pytorch_time'].mean(), results_df['quantized_time'].mean()]
    
    bars = plt.bar(model_labels, times, color=['#4472C4', '#70AD47'], width=0.6)
    plt.title('Average Inference Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text annotations on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}s', ha='center', fontsize=9)
    
    # Add speedup as text
    speedup = results_df['pytorch_time'].mean() / results_df['quantized_time'].mean()
    plt.text(0.5, 0.9, f'Speedup: {speedup:.2f}x', 
             horizontalalignment='center',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'inference_time_comparison.png'), dpi=300)
    
    # 2. Error Rates Comparison
    plt.figure(figsize=(6, 4))
    
    x = np.arange(2)
    width = 0.35
    
    cer_vals = [results_df['pytorch_cer'].mean(), results_df['quantized_cer'].mean()]
    wer_vals = [results_df['pytorch_wer'].mean(), results_df['quantized_wer'].mean()]
    
    plt.bar(x - width/2, cer_vals, width, label='CER', color='#4472C4')
    plt.bar(x + width/2, wer_vals, width, label='WER', color='#70AD47')
    
    plt.xlabel('Model Type')
    plt.ylabel('Error Rate')
    plt.title('CER and WER Comparison')
    plt.xticks(x, model_labels)
    plt.legend()
    
    # Add text annotations
    for i, v in enumerate(cer_vals):
        plt.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)
    for i, v in enumerate(wer_vals):
        plt.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_rates_comparison.png'), dpi=300)
    
    # 3. CER Distribution Comparison
    plt.figure(figsize=(7, 4))
    
    # Create CER ranges
    bins = [0, 0.05, 0.1, 0.2, 0.3, 1.0]
    cer_range_labels = ['0-5%', '5-10%', '10-20%', '20-30%', '30-100%']
    
    results_df['pytorch_cer_range'] = pd.cut(results_df['pytorch_cer'], bins=bins, labels=cer_range_labels)
    results_df['quantized_cer_range'] = pd.cut(results_df['quantized_cer'], bins=bins, labels=cer_range_labels)
    
    pytorch_dist = results_df['pytorch_cer_range'].value_counts().sort_index()
    quant_dist = results_df['quantized_cer_range'].value_counts().sort_index()
    
    x = np.arange(len(cer_range_labels))
    width = 0.35
    
    pytorch_vals = [pytorch_dist.get(label, 0) for label in cer_range_labels]
    quant_vals = [quant_dist.get(label, 0) for label in cer_range_labels]
    
    plt.bar(x - width/2, pytorch_vals, width, label='PyTorch Model', color='#4472C4')
    plt.bar(x + width/2, quant_vals, width, label='Quantized Model', color='#70AD47')
    
    plt.xlabel('CER Range')
    plt.ylabel('Number of Images')
    plt.title('CER Distribution Comparison')
    plt.xticks(x, cer_range_labels)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cer_distribution_comparison.png'), dpi=300)
    
    # 4. Inference Time Distribution
    plt.figure(figsize=(7, 4))
    
    # Create boxplot of inference times
    data = [results_df['pytorch_time'], results_df['quantized_time']]
    
    # Use tick_labels instead of labels (updated parameter name)
    bp = plt.boxplot(data, tick_labels=model_labels, patch_artist=True)
    
    # Set colors
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=['#4472C4', '#70AD47'][i])
    
    plt.ylabel('Inference Time (seconds)')
    plt.title('Distribution of Inference Times')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'inference_time_distribution.png'), dpi=300)

def main():
    # Check if folders exist
    if not os.path.exists(image_folder):
        print(f"Error: Image folder '{image_folder}' does not exist.")
        return
    if not os.path.exists(gt_folder):
        print(f"Error: Ground truth folder '{gt_folder}' does not exist.")
        return
    
    # Load processor
    print("Loading processor...")
    processor = TrOCRProcessor.from_pretrained(model_path)
    
    # Load ground truth
    gt_dict = load_ground_truth(gt_folder)
    if not gt_dict:
        print("No ground truth files found or all files are empty.")
        return
    
    # Find images
    image_files = glob.glob(os.path.join(image_folder, "*.jpg")) + \
                  glob.glob(os.path.join(image_folder, "*.png")) + \
                  glob.glob(os.path.join(image_folder, "*.jpeg"))
    
    if not image_files:
        print("No image files found.")
        return
    
    # Filter images with ground truth
    valid_images = []
    for img_path in image_files:
        basename = os.path.basename(img_path).split(".")[0]
        if basename in gt_dict and gt_dict[basename]:
            valid_images.append((img_path, basename))
    
    if not valid_images:
        print("No images with corresponding ground truth found.")
        return
    
    # Select random samples
    if len(valid_images) > num_samples:
        print(f"Selecting {num_samples} random images from {len(valid_images)} valid images...")
        random_samples = random.sample(valid_images, num_samples)
    else:
        print(f"Using all {len(valid_images)} valid images (fewer than requested {num_samples})...")
        random_samples = valid_images
    
    # Initialize results structure
    results = []
    
    # STEP 1: Process all images with quantized model first
    print("\n===== PROCESSING WITH QUANTIZED MODEL =====")
    
    # Load quantized model
    print("Loading quantized model...")
    try:
        quantized_model = ORTModelForVision2Seq.from_pretrained(quantized_model_path)
        print("Successfully loaded quantized model")
    except Exception as e:
        print(f"Error loading quantized model: {e}")
        return
    
    # Process all images with quantized model
    quantized_results = {}
    total = len(random_samples)
    
    for i, (img_path, basename) in enumerate(random_samples):
        print(f"[{i+1}/{total}] Processing {basename} with quantized model...")
        
        # Process with quantized model
        result = perform_ocr_quantized(quantized_model, processor, img_path)
        
        # Store result
        quantized_results[basename] = {
            "text": result["text"],
            "time": result["inference_time"]
        }
    
    # Calculate average inference time for quantized model
    avg_quant_time = sum(item["time"] for item in quantized_results.values()) / len(quantized_results)
    print(f"\nAverage quantized model inference time: {avg_quant_time:.4f} seconds")
    
    # STEP 2: Process all images with PyTorch model
    print("\n===== PROCESSING WITH PYTORCH MODEL =====")
    
    # Load PyTorch model
    print("Loading PyTorch model...")
    try:
        pytorch_model = VisionEncoderDecoderModel.from_pretrained(model_path)
        pytorch_model.eval()
        print("Successfully loaded PyTorch model")
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        return
    
    # Process all images with PyTorch model
    pytorch_results = {}
    
    for i, (img_path, basename) in enumerate(random_samples):
        print(f"[{i+1}/{total}] Processing {basename} with PyTorch model...")
        
        # Process with PyTorch model
        result = perform_ocr_pytorch(pytorch_model, processor, img_path)
        
        # Store result
        pytorch_results[basename] = {
            "text": result["text"],
            "time": result["inference_time"]
        }
    
    # Calculate average inference time for PyTorch model
    avg_pytorch_time = sum(item["time"] for item in pytorch_results.values()) / len(pytorch_results)
    print(f"\nAverage PyTorch model inference time: {avg_pytorch_time:.4f} seconds")
    
    # STEP 3: Compile results and calculate metrics
    print("\n===== CALCULATING METRICS =====")
    
    for img_path, basename in random_samples:
        ground_truth = gt_dict[basename]
        
        # Get results for both models
        pytorch_result = pytorch_results[basename]
        quantized_result = quantized_results[basename]
        
        # Calculate metrics
        pytorch_cer = calculate_cer(ground_truth, pytorch_result["text"])
        pytorch_wer = jiwer.wer(ground_truth, pytorch_result["text"])
        
        quantized_cer = calculate_cer(ground_truth, quantized_result["text"])
        quantized_wer = jiwer.wer(ground_truth, quantized_result["text"])
        
        # Check if outputs match
        outputs_match = pytorch_result["text"] == quantized_result["text"]
        
        # Compile result
        result = {
            "image": basename,
            "ground_truth": ground_truth,
            "pytorch_text": pytorch_result["text"],
            "quantized_text": quantized_result["text"],
            "pytorch_time": pytorch_result["time"],
            "quantized_time": quantized_result["time"],
            "pytorch_cer": pytorch_cer,
            "quantized_cer": quantized_cer,
            "pytorch_wer": pytorch_wer,
            "quantized_wer": quantized_wer,
            "outputs_match": outputs_match
        }
        
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate statistics
    num_matching = df["outputs_match"].sum()
    num_total = len(df)
    percent_matching = (num_matching / num_total) * 100 if num_total > 0 else 0
    
    # Time comparison
    time_speedup = avg_pytorch_time / avg_quant_time if avg_quant_time > 0 else 0
    
    # Accuracy comparison
    avg_pytorch_cer = df["pytorch_cer"].mean()
    avg_quantized_cer = df["quantized_cer"].mean()
    avg_pytorch_wer = df["pytorch_wer"].mean()
    avg_quantized_wer = df["quantized_wer"].mean()
    
    cer_diff = avg_quantized_cer - avg_pytorch_cer
    wer_diff = avg_quantized_wer - avg_pytorch_wer
    
    # Print summary
    print("\n===== MODEL COMPARISON RESULTS =====")
    print(f"Number of samples: {len(df)}")
    
    print("\n----- INFERENCE TIME -----")
    print(f"PyTorch model average:   {avg_pytorch_time:.4f} seconds")
    print(f"Quantized model average: {avg_quant_time:.4f} seconds")
    print(f"Speedup:                 {time_speedup:.2f}x")
    
    print("\n----- ACCURACY METRICS -----")
    print(f"PyTorch model CER:      {avg_pytorch_cer:.4f}")
    print(f"Quantized model CER:    {avg_quantized_cer:.4f}")
    print(f"CER difference:         {cer_diff:.4f} ({'+' if cer_diff > 0 else ''}{cer_diff*100:.2f}%)")
    
    print(f"PyTorch model WER:      {avg_pytorch_wer:.4f}")
    print(f"Quantized model WER:    {avg_quantized_wer:.4f}")
    print(f"WER difference:         {wer_diff:.4f} ({'+' if wer_diff > 0 else ''}{wer_diff*100:.2f}%)")
    
    # Output similarity
    print("\n----- OUTPUT SIMILARITY -----")
    print(f"Identical outputs: {num_matching} out of {num_total} ({percent_matching:.1f}%)")
    
    # Memory usage estimate (model size on disk)
    pytorch_size = sum(os.path.getsize(os.path.join(model_path, f)) for f in os.listdir(model_path) if f.endswith('.bin'))
    quantized_size = sum(os.path.getsize(os.path.join(quantized_model_path, f)) for f in os.listdir(quantized_model_path) if f.endswith('.onnx'))
    
    size_reduction = (1 - (quantized_size / pytorch_size)) * 100 if pytorch_size > 0 else 0
    
    print("\n----- MODEL SIZE -----")
    print(f"PyTorch model size:     {pytorch_size / (1024*1024):.2f} MB")
    print(f"Quantized model size:   {quantized_size / (1024*1024):.2f} MB")
    print(f"Size reduction:         {size_reduction:.1f}%")
    
    # Overall assessment
    print("\n----- OVERALL ASSESSMENT -----")
    if time_speedup >= 3.0 and abs(cer_diff) < 0.02:
        print("EXCELLENT: Quantization achieved dramatic speedup with minimal accuracy impact")
    elif time_speedup >= 2.0 and abs(cer_diff) < 0.05:
        print("VERY GOOD: Quantization provides substantial speedup with acceptable accuracy impact")
    elif time_speedup >= 1.5 and abs(cer_diff) < 0.1:
        print("GOOD: Quantization provides good speedup with reasonable accuracy trade-off")
    elif time_speedup >= 1.2:
        print("FAIR: Quantization provides some speedup but may affect accuracy")
    else:
        print("POOR: Quantization did not provide significant benefits")
    
    # Save results to CSV
    csv_path = os.path.join(results_dir, "pytorch_vs_quantized_sequential.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to {csv_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df, results_dir)
    print(f"Visualizations saved to {results_dir} directory")

    # Add this to your print summary section:
    print("\n----- MODEL COMPLEXITY -----")
    print(f"PyTorch model parameters: {count_parameters(pytorch_model):,}")

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import jiwer
    except ImportError:
        print("Installing jiwer package for WER calculation...")
        import subprocess
        subprocess.check_call(["pip", "install", "jiwer"])
        import jiwer
    
    try:
        import Levenshtein
    except ImportError:
        print("Installing Levenshtein package for CER calculation...")
        import subprocess
        subprocess.check_call(["pip", "install", "python-Levenshtein"])
        import Levenshtein
    
    try:
        import matplotlib
    except ImportError:
        print("Installing matplotlib for visualizations...")
        import subprocess
        subprocess.check_call(["pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt
    
    main()