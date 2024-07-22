import subprocess
import os

book_name = "book2"
# Define the paths and parameters
book_path = f"../../data/train/pdf/{book_name}.pdf"
output_dir = f"../../data/train/processed_book/{book_name}/pages"
trained_model = "../CRAFT/weights/craft_mlt_25k.pth"
refiner_model = "../CRAFT/weights/craft_refiner_CTW1500.pth"
text_threshold = 0.9
test_folder = output_dir  # Images folder created by process_main_utils.py
result_folder = "./result"
contour_line_splitter_script = "contour_line_splitter.py"
line_segments_folder = f"../../data/train/processed_book/{book_name}/line_segments"
padding = 20
min_width = 0
threshold = 0.6
margin = 0.1

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)
os.makedirs(line_segments_folder, exist_ok=True)

# Check if the output directory is empty
if not os.listdir(output_dir):
    print("Output directory is empty. Performing Step 1: Processing PDF into individual images.")
    # Step 1: Process the PDF into individual images
    subprocess.run([
        "python", "process_main_utils.py",
        book_path, output_dir,
        "--dpi", "300",
        "--remove_borders"
    ])
else:
    print("Output directory is not empty. Skipping Step 1.")

# Step 2: Run the text detection model on the processed images
print("Performing Step 2: Running the text detection model.")
subprocess.run([
    "python", "../CRAFT/test.py",
    "--trained_model", trained_model,
    "--text_threshold", str(text_threshold),
    "--test_folder", output_dir,
    "--refine",
    "--refiner_model", refiner_model
])

# Step 3: Run the contour line splitter
print("Performing Step 3: Running the contour line splitter.")
subprocess.run([
    "python", contour_line_splitter_script,
    output_dir,  # Image directory from the first process
    result_folder,  # Contour directory from the second process
    line_segments_folder,  # Output directory for line segments
    "--padding", str(padding),
    "--min_width", str(min_width),
    "--threshold", str(threshold),
    "--margin", str(margin)
])

print("Processing completed successfully.")