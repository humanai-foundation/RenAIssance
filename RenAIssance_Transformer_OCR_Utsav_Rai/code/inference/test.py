import re
import os
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_path = "../../weights/trocr_finetuned_printed_optuna"
processor_path = "../../wights/trocr_finetuned_printed_optuna"

# Load the fine-tuned model and processor
processor = TrOCRProcessor.from_pretrained(processor_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)

# Function to generate text for a single image segment
def generate_text_from_image_segment(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)  # Move pixel_values to the correct device
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Base directory where the page folders are located
base_dir = "../../data/test" 

def sort_key(filename):
    """
    Custom sort function to extract the segment number from the filename
    and use it as the key for sorting.
    """
    match = re.search(r"(\d+)\.jpg", filename)
    if match:
        return int(match.group(1))
    return -1  # Return -1 if the pattern doesn't match

# Iterate through each page's folder
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
            line_text = generate_text_from_image_segment(line_segment_path)
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