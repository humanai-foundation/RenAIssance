import time
import torch
from PIL import Image
import os
import glob
import shutil
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from optimum.onnxruntime import ORTModelForVision2Seq, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Path to your fine-tuned model
model_path = "../../models"  # Path to your model
onnx_path = "../../models/trocr_onnx_temp"  # Temporary directory for ONNX conversion
quantized_path = "../../models/trocr_quantized"  # Final directory for quantized model

# Function to measure inference time
def measure_inference_time(model, processor, image, num_runs=5):
    # Preprocess the image
    inputs = processor(image, return_tensors="pt")
    pixel_values = inputs.pixel_values
    
    # Warmup
    for _ in range(2):
        _ = model.generate(pixel_values)
    
    # Measure time
    start_time = time.time()
    for _ in range(num_runs):
        _ = model.generate(pixel_values)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time

# Create temporary directory and ensure quantized directory exists
os.makedirs(onnx_path, exist_ok=True)
os.makedirs(quantized_path, exist_ok=True)

# Load processor and PyTorch model
print("Loading PyTorch model...")
processor = TrOCRProcessor.from_pretrained(model_path)
pytorch_model = VisionEncoderDecoderModel.from_pretrained(model_path)
pytorch_model.eval()

# Load a test image
test_image_path = "40.jpg"
image = Image.open(test_image_path).convert("RGB")

# Measure PyTorch model inference time
print("Measuring PyTorch model performance...")
pytorch_time = measure_inference_time(pytorch_model, processor, image)
print(f"PyTorch model inference time: {pytorch_time:.4f} seconds")

# Get PyTorch output
inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    pytorch_output = pytorch_model.generate(inputs.pixel_values)
pytorch_text = processor.batch_decode(pytorch_output, skip_special_tokens=True)[0]
print(f"PyTorch output: {pytorch_text}")

# Export to ONNX using optimum (to temporary directory)
print("\nExporting to ONNX format (temporary)...")
ort_model = ORTModelForVision2Seq.from_pretrained(
    model_path,
    export=True,
    provider="CPUExecutionProvider"
)
ort_model.save_pretrained(onnx_path)

# Measure ONNX model inference time
print("Measuring ONNX model performance...")
onnx_time = measure_inference_time(ort_model, processor, image)
print(f"ONNX model inference time: {onnx_time:.4f} seconds")
print(f"ONNX speedup: {pytorch_time/onnx_time:.2f}x")

# Get ONNX output for comparison
inputs = processor(image, return_tensors="pt")
onnx_output = ort_model.generate(inputs.pixel_values)
onnx_text = processor.batch_decode(onnx_output, skip_special_tokens=True)[0]
print(f"ONNX output: {onnx_text}")

# Quantize the model directly to final directory
print("\nQuantizing the ONNX model components...")

# Find all ONNX files in the temporary directory
onnx_files = glob.glob(os.path.join(onnx_path, "*.onnx"))
print(f"Found {len(onnx_files)} ONNX files to quantize: {[os.path.basename(f) for f in onnx_files]}")

for onnx_file in onnx_files:
    file_name = os.path.basename(onnx_file)
    print(f"\nQuantizing {file_name}...")
    
    try:
        quantizer = ORTQuantizer.from_pretrained(
            onnx_path, 
            file_name=file_name
        )
        
        qconfig = AutoQuantizationConfig.dynamic(
            per_channel=False,
            operators_to_quantize=['MatMul'],
            exclude_nodes=[]
        )
        
        # Apply quantization directly to final directory
        quantizer.quantize(
            save_dir=quantized_path,
            quantization_config=qconfig
        )
        print(f"Successfully quantized {file_name}")
    except Exception as e:
        print(f"Error quantizing {file_name}: {e}")
        # Only copy the original file if quantization fails
        shutil.copy(onnx_file, os.path.join(quantized_path, file_name))
        print(f"Copied original file {file_name} to quantized directory")

# Copy only necessary configuration files from temp directory to final directory
required_config_files = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json"
]

for config_file in required_config_files:
    src_path = os.path.join(onnx_path, config_file)
    if os.path.exists(src_path):
        shutil.copy(src_path, os.path.join(quantized_path, config_file))
        print(f"Copied essential config file {config_file} to quantized directory")

# Load and test the quantized model
print("\nLoading quantized model...")
try:
    quantized_model = ORTModelForVision2Seq.from_pretrained(quantized_path)

    # Measure quantized model inference time
    print("Measuring quantized model performance...")
    quantized_time = measure_inference_time(quantized_model, processor, image)
    print(f"Quantized model inference time: {quantized_time:.4f} seconds")
    print(f"Quantized speedup vs PyTorch: {pytorch_time/quantized_time:.2f}x")
    print(f"Quantized speedup vs ONNX: {onnx_time/quantized_time:.2f}x")

    # Get quantized output for comparison
    inputs = processor(image, return_tensors="pt")
    quantized_output = quantized_model.generate(inputs.pixel_values)
    quantized_text = processor.batch_decode(quantized_output, skip_special_tokens=True)[0]
    print(f"Quantized output: {quantized_text}")

    print("\nSummary:")
    print(f"PyTorch model: {pytorch_time:.4f} seconds")
    print(f"ONNX model: {onnx_time:.4f} seconds (speedup: {pytorch_time/onnx_time:.2f}x)")
    print(f"Quantized model: {quantized_time:.4f} seconds (speedup: {pytorch_time/quantized_time:.2f}x)")

    # Clean up the temporary directory
    print("\nCleaning up temporary files...")
    shutil.rmtree(onnx_path)
    print(f"Removed temporary directory: {onnx_path}")

except Exception as e:
    print(f"Error loading quantized model: {e}")
    print("\nSummary:")
    print(f"PyTorch model: {pytorch_time:.4f} seconds")
    print(f"ONNX model: {onnx_time:.4f} seconds (speedup: {pytorch_time/onnx_time:.2f}x)")
    print("Quantized model: Failed to load")

# Calculate and print saved storage space
pytorch_size = sum(os.path.getsize(os.path.join(model_path, f)) for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f)))
quantized_size = sum(os.path.getsize(os.path.join(quantized_path, f)) for f in os.listdir(quantized_path) if os.path.isfile(os.path.join(quantized_path, f)))

print(f"\nStorage comparison:")
print(f"PyTorch model size: {pytorch_size / (1024*1024):.2f} MB")
print(f"Quantized model size: {quantized_size / (1024*1024):.2f} MB")
print(f"Storage reduction: {(1 - quantized_size/pytorch_size) * 100:.2f}%")

print("\nThe quantized model is saved at:", quantized_path)