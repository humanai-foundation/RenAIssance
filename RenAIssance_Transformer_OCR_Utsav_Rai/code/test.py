import re
import os
import yaml
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from utils import sort_key, generate_text_from_image_segment

# Load configuration
try:
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    raise Exception("Configuration file not found.")
except yaml.YAMLError as e:
    raise Exception(f"Error reading configuration file: {e}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Paths to model and processor
model_path = config["model_dir"]
processor_path = config["model_dir"]

# Load the fine-tuned model and processor
try:
    processor = TrOCRProcessor.from_pretrained(processor_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
except Exception as e:
    raise Exception(f"Failed to load model or processor: {e}")

# Base directory where the page folders are located
base_dir = config["base_dir"]

# Iterate through each page's folder
try:
    for page_folder in sorted(os.listdir(base_dir)):
        page_path = os.path.join(base_dir, page_folder)
        if os.path.isdir(page_path):
            print(f"Processing {page_folder}:")
            page_texts = []

            # Sort the line segment images numerically based on the segment number
            line_segment_images = sorted([f for f in os.listdir(page_path) if f.endswith('.jpg')], key=sort_key)

            # Iterate through each sorted line segment in the page folder
            for line_segment_image in line_segment_images:
                line_segment_path = os.path.join(page_path, line_segment_image)
                line_text = generate_text_from_image_segment(line_segment_path, processor, model)
                page_texts.append((line_segment_image, line_text))
            
            # Sort the generated texts based on the filenames
            page_texts.sort(key=lambda x: sort_key(x[0]))

            # Print the texts in sorted order
            for line_segment_image, line_text in page_texts:
                print(f"  {line_segment_image}: {line_text}")

            # Compile and display the full page's text
            full_page_text = "\n".join([text for _, text in page_texts])
            print(f"\nFull text for {page_folder}:")
            print(full_page_text)
            print("\n" + "="*50 + "\n")

            # Save the full page's text to a file
            output_file = os.path.join(page_path, "output.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(full_page_text)
except Exception as e:
    raise Exception(f"Failed during inference or saving results: {e}")
