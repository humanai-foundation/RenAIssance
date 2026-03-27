import fitz  # PyMuPDF
from PIL import Image, ImageTk, ImageEnhance
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Canvas, ttk, Scrollbar, IntVar, BooleanVar, DoubleVar
import platform
import os
import glob
import subprocess
import tkinter as tk
from tkinter import Toplevel, Text
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
import os
from shared_app_utils import (
    deskew_image,
    draw_bounding_boxes,
    filter_and_adjust_bounding_boxes,
    get_bounding_boxes,
    preprocess_image,
    read_contour_points,
    remove_borders,
    rotate,
    split_bounding_boxes,
)

# Set up the OCR model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OCR_MODEL_PATH = os.path.abspath(os.path.join(APP_DIR, "..", "..", "models"))
model_path = os.getenv("RENAISSANCE_OCR_MODEL_DIR", DEFAULT_OCR_MODEL_PATH)
processor_path = model_path

processor = TrOCRProcessor.from_pretrained(processor_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)


# Global variables
current_page = 0
total_pages = 0
processed_pages = []  # List to store processed pages of the current PDF page
pdf_document = None   # Global variable to keep the PDF document accessible
zoom_factor = 1.0  # Default zoom factor
left_page_image = None
right_page_image = None
raw_left_half = None
raw_right_half = None
temp_dir = "temp_images"  # Temporary directory to store images
contour_dir = "result"  # Directory where test.py outputs contour files

# Ensure temp_dir exists
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Variables to store contour data
contour_data_left = None
contour_data_right = None
need_to_run_craft = False  # Flag to control when to run the craft script

def create_custom_slider(root, label_text, variable, from_, to, resolution, command, fill_color="#8A2BE2"):
    frame = tk.Frame(root, bg='#2e2e2e')
    frame.pack(pady=10, padx=10, fill=tk.X)

    # Label on the left
    label = tk.Label(frame, text=label_text, fg='white', bg='#2e2e2e')
    label.pack(side=tk.LEFT, padx=(0, 10))

    # Canvas to draw the slider components
    slider_canvas = tk.Canvas(frame, height=60, bg='#2e2e2e', bd=0, highlightthickness=0, relief='ridge')
    slider_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)

    # Value label on the right
    value_label = tk.Label(frame, text=f"{variable.get():.2f}", fg='white', bg='#2e2e2e')
    value_label.pack(side=tk.RIGHT, padx=(10, 0))

    # **Flag to indicate initialization**
    initializing = True

    # Function to update the slider visuals
    def update_slider(*args):
        slider_canvas.delete('all')  # Clear canvas
        track_start = 20
        track_end = slider_canvas.winfo_width() - 20  # Dynamic width based on canvas size
        track_length = track_end - track_start
        track_y = 30  # Center of the track

        # Draw the track line
        slider_canvas.create_line(track_start, track_y, track_end, track_y, width=15, fill='#ccc', capstyle='round')

        # Calculate the position of the handle
        slider_value = (variable.get() - from_) / (to - from_) * track_length + track_start

        # Draw the filled part of the track
        slider_canvas.create_line(track_start, track_y, slider_value, track_y, width=15, fill=fill_color, capstyle='round')

        # Draw the handle (thumb) as a perfect circle centered on the track
        handle_size = 15  # Adjust this value to increase the size of the handle
        slider_canvas.create_oval(slider_value - handle_size, track_y - handle_size, 
                                  slider_value + handle_size, track_y + handle_size, 
                                  fill=fill_color, outline='')

        # Update the value label
        value_label.config(text=f"{variable.get():.2f}")

        # **Call command only if not initializing**
        if not initializing:
            command()

    # Function to handle mouse click and movement to adjust the slider value
    def move_slider(event):
        track_start = 20
        track_end = slider_canvas.winfo_width() - 20
        track_length = track_end - track_start

        # Calculate the new value based on the click position
        new_value = (event.x - track_start) / track_length * (to - from_) + from_
        # Set the slider value within the bounds
        variable.set(min(max(new_value, from_), to))

    # Bind updates
    def variable_changed(*args):
        update_slider()
    variable.trace_add('write', variable_changed)
    slider_canvas.bind('<B1-Motion>', move_slider)
    slider_canvas.bind('<Button-1>', move_slider)

    # Adjust the slider whenever the window resizes
    slider_canvas.bind('<Configure>', update_slider)

    # **Set initializing to False after initial setup**
    initializing = False
    update_slider()



# Global variables for storing original processed images (without bounding boxes)
original_left_page_image = None
original_right_page_image = None

def process_page():
    global processed_pages, raw_left_half, raw_right_half, left_page_image, right_page_image
    global contour_data_left, contour_data_right, need_to_run_craft
    global original_left_page_image, original_right_page_image  # Declare the original images

    processed_pages.clear()  # Clear previous images
    right_page_image = None  # Reset right page image when processing a new page
    contour_data_left = None
    contour_data_right = None
    need_to_run_craft = line_segmentation_var.get()  # Run CRAFT if line segmentation is enabled
    
    if not pdf_document or current_page >= total_pages:
        return

    try:
        page = pdf_document.load_page(current_page)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi_var.get() / 72, dpi_var.get() / 72), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image = ImageEnhance.Sharpness(image).enhance(2)  # Enhance sharpness
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Split if the page appears to be a double-page spread
        width, height = image.shape[1], image.shape[0]
        if width > height:
            left_half = image[:, :width // 2]
            right_half = image[:, width // 2:]

            raw_left_half = left_half.copy()
            raw_right_half = right_half.copy()

            # Process only if show_raw_var is False
            if not show_raw_var.get():
                if deskew_var.get():
                    left_half = deskew_image(left_half)
                    right_half = deskew_image(right_half)

                if remove_borders_var.get():
                    left_half = remove_borders(left_half)
                    right_half = remove_borders(right_half)

                # Apply noise threshold and intensity (See Through) adjustments
                left_half = preprocess_image(left_half, noise_threshold_var.get(), intensity_threshold_var.get())
                right_half = preprocess_image(right_half, noise_threshold_var.get(), intensity_threshold_var.get())

            left_page_image = left_half if not show_raw_var.get() else raw_left_half
            right_page_image = right_half if not show_raw_var.get() else raw_right_half

            # Store the original processed images without bounding boxes
            original_left_page_image = left_page_image.copy()
            original_right_page_image = right_page_image.copy()

            processed_pages.append(left_page_image)
            processed_pages.append(right_page_image)
        else:
            raw_left_half = image.copy()

            if not show_raw_var.get():
                if deskew_var.get():
                    image = deskew_image(image)

                if remove_borders_var.get():
                    image = remove_borders(image)

                # Apply noise threshold and intensity (See Through) adjustments
                image = preprocess_image(image, noise_threshold_var.get(), intensity_threshold_var.get())

            left_page_image = image if not show_raw_var.get() else raw_left_half
            right_page_image = None  # Ensure right page image is set to None

            # Store the original processed image without bounding boxes
            original_left_page_image = left_page_image.copy()

            processed_pages.append(left_page_image)

        # If line segmentation is enabled
        if line_segmentation_var.get():
            perform_line_segmentation()
        else:
            update_image()  # Display the processed images

    except Exception as e:
        print(f"Error processing page {current_page + 1}: {e}")


# Function to perform line segmentation on the processed images
def perform_line_segmentation():
    global left_page_image, right_page_image, contour_data_left, contour_data_right, need_to_run_craft

    images_to_process = []
    image_names = []

    # Prepare images and names
    if left_page_image is not None:
        images_to_process.append(left_page_image)
        image_names.append(f"page_{current_page + 1}_left.jpg")
    if right_page_image is not None:
        images_to_process.append(right_page_image)
        image_names.append(f"page_{current_page + 1}_right.jpg")

    if need_to_run_craft:
        # Clear temp_dir and contour_dir
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        for filename in os.listdir(contour_dir):
            file_path = os.path.join(contour_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        # Save images to temp_dir
        for img, name in zip(images_to_process, image_names):
            cv2.imwrite(os.path.join(temp_dir, name), img)

        # Run the test.py script
        command = [
            "python", "../CRAFT/test.py",
            "--trained_model", "../CRAFT/weights/craft_mlt_25k.pth",
            "--text_threshold", "0.9",
            "--test_folder", "temp_images",
            "--refine",
            "--refiner_model", "../CRAFT/weights/craft_refiner_CTW1500.pth"
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running test.py: {e}")

        need_to_run_craft = False  # Reset the flag after running craft

    # Read contour files and store data
    contour_data_left = None
    contour_data_right = None

    for img_name in image_names:
        contour_path = os.path.join(contour_dir, f"res_{os.path.splitext(img_name)[0]}.txt")
        if os.path.exists(contour_path):
            contours = read_contour_points(contour_path)
            if "left" in img_name:
                contour_data_left = contours
            elif "right" in img_name:
                contour_data_right = contours

    apply_line_segmentation()  # Apply line segmentation with current slider values

# Add this line to store final bounding boxes
ocr_bounding_boxes_left = []
ocr_bounding_boxes_right = []

def apply_line_segmentation():
    global left_page_image, right_page_image, contour_data_left, contour_data_right
    global original_left_page_image, original_right_page_image  # Use the original images
    global ocr_bounding_boxes_left, ocr_bounding_boxes_right  # Store bounding boxes for OCR

    ocr_bounding_boxes_left = []  # Initialize OCR bounding boxes
    ocr_bounding_boxes_right = []

    images_processed = False  # Flag to check if any image was processed

    # Process left image
    if original_left_page_image is not None and contour_data_left is not None:
        # Reset to the original processed image before drawing new bounding boxes
        left_page_image = original_left_page_image.copy()

        # Convert image to BGR color format if needed
        if len(left_page_image.shape) == 2 or left_page_image.shape[2] == 1:
            left_page_image = cv2.cvtColor(left_page_image, cv2.COLOR_GRAY2BGR)

        img_height, img_width = left_page_image.shape[:2]
        # Get bounding boxes
        bounding_boxes = get_bounding_boxes(
            contour_data_left,
            img_width,
            img_height,
            padding=padding_var.get(),
            min_width=min_width_var.get(),
            margin=margin_var.get()
        )

        # Split bounding boxes
        new_bounding_boxes = split_bounding_boxes(left_page_image, bounding_boxes, threshold=threshold_var.get())

        # Filter and adjust bounding boxes
        adjusted_bounding_boxes = filter_and_adjust_bounding_boxes(new_bounding_boxes)

        # Sort bounding boxes by their top (y1) coordinate to maintain order
        sorted_bounding_boxes = sorted(adjusted_bounding_boxes, key=lambda box: box[1])

        # Store the sorted bounding boxes for OCR
        ocr_bounding_boxes_left = sorted_bounding_boxes

        # Draw bounding boxes on the image
        draw_bounding_boxes(left_page_image, sorted_bounding_boxes)

        images_processed = True

    # Process right image
    if original_right_page_image is not None and contour_data_right is not None:
        # Reset to the original processed image before drawing new bounding boxes
        right_page_image = original_right_page_image.copy()

        # Convert image to BGR color format if needed
        if len(right_page_image.shape) == 2 or right_page_image.shape[2] == 1:
            right_page_image = cv2.cvtColor(right_page_image, cv2.COLOR_GRAY2BGR)

        img_height, img_width = right_page_image.shape[:2]
        # Get bounding boxes
        bounding_boxes = get_bounding_boxes(
            contour_data_right,
            img_width,
            img_height,
            padding=padding_var.get(),
            min_width=min_width_var.get(),
            margin=margin_var.get()
        )

        # Split bounding boxes
        new_bounding_boxes = split_bounding_boxes(right_page_image, bounding_boxes, threshold=threshold_var.get())

        # Filter and adjust bounding boxes
        adjusted_bounding_boxes = filter_and_adjust_bounding_boxes(new_bounding_boxes)

        # Sort bounding boxes by their top (y1) coordinate to maintain order
        sorted_bounding_boxes = sorted(adjusted_bounding_boxes, key=lambda box: box[1])

        # Store the sorted bounding boxes for OCR
        ocr_bounding_boxes_right = sorted_bounding_boxes

        # Draw bounding boxes on the image
        draw_bounding_boxes(right_page_image, sorted_bounding_boxes)

        images_processed = True

    if images_processed:
        update_image()  # Update the display with the new images



# Function to generate text for a single image segment
# Function to generate text from a single line image segment
def generate_text_from_image_segment(image_segment):
    image_pil = Image.fromarray(cv2.cvtColor(image_segment, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def crop_and_transcribe(image, bounding_boxes):
    texts = []
    for (x1, y1, x2, y2) in bounding_boxes:
        # Crop the image segment using bounding box coordinates
        line_segment = image[y1:y2, x1:x2]
        # Generate text from the image segment
        text = generate_text_from_image_segment(line_segment)
        texts.append(text)
    return texts




def transcribe_and_display_ocr():
    global ocr_bounding_boxes_left, ocr_bounding_boxes_right  # Use the stored bounding boxes

    # Create a new window for displaying the OCR output
    ocr_window = Toplevel(root)
    ocr_window.title("OCR Output")
    ocr_window.geometry("600x400")
    
    # Text widget to display the OCR output
    text_display = Text(ocr_window, wrap='word', font=('Arial', 12))
    text_display.pack(expand=True, fill='both')

    def display_text(text):
        text_display.insert(tk.END, text + "\n")
        text_display.update()

    # Transcribe lines from the left page if available, updating the display line by line
    if left_page_image is not None and ocr_bounding_boxes_left is not None:
        display_text("Transcribing Left Page:\n")
        for i, (x1, y1, x2, y2) in enumerate(ocr_bounding_boxes_left):
            # Crop each line segment from the left page image
            line_segment = left_page_image[y1:y2, x1:x2]
            # Generate text from the image segment
            text = generate_text_from_image_segment(line_segment)
            # display_text(f"Line {i+1}: {text}")
            display_text(f"{text}")
    
    # Transcribe lines from the right page if available, updating the display line by line
    if right_page_image is not None and ocr_bounding_boxes_right is not None:
        display_text("\nTranscribing Right Page:\n")
        for i, (x1, y1, x2, y2) in enumerate(ocr_bounding_boxes_right):
            # Crop each line segment from the right page image
            line_segment = right_page_image[y1:y2, x1:x2]
            # Generate text from the image segment
            text = generate_text_from_image_segment(line_segment)
            # display_text(f"Line {i+1}: {text}")
            display_text(f"{text}")

    # Add closing text when transcription is complete
    display_text("\nOCR Transcription Complete.")






# Function to update the display based on the current zoom level
def update_image():
    # If only the left page is available (single page mode)
    if right_page_image is None:
        # Display only the left canvas and hide the right canvas
        canvas_left.grid(row=0, column=0, sticky='nsew')  # Place left canvas in the first column
        canvas_right.grid_forget()  # Hide the right canvas completely
        main_frame.grid_columnconfigure(0, weight=1)  # Make left canvas expand to fill space
        main_frame.grid_columnconfigure(1, weight=0)  # Reset the right canvas column
        main_frame.grid_columnconfigure(2, weight=0)  # No space reserved for the right canvas

        # Resize and display the left page image
        resized_left = cv2.resize(
            left_page_image,
            (int(left_page_image.shape[1] * zoom_factor), int(left_page_image.shape[0] * zoom_factor))
        )
        display_image(resized_left, "Single Page", is_right=False)

    else:
        # Display both canvases when both pages are available
        canvas_left.grid(row=0, column=0, sticky='nsew')
        canvas_right.grid(row=0, column=1, sticky='nsew')  # Adjust the column placement
        main_frame.grid_columnconfigure(0, weight=1)  # Expand the left canvas space
        main_frame.grid_columnconfigure(1, weight=1)  # Expand the right canvas space
        main_frame.grid_columnconfigure(2, weight=0)  # No space reserved for the control panel

        # Resize and display the left and right page images
        resized_left = cv2.resize(
            left_page_image,
            (int(left_page_image.shape[1] * zoom_factor), int(left_page_image.shape[0] * zoom_factor))
        )
        resized_right = cv2.resize(
            right_page_image,
            (int(right_page_image.shape[1] * zoom_factor), int(right_page_image.shape[0] * zoom_factor))
        )

        display_image(resized_left, "Left Page", is_right=False)
        display_image(resized_right, "Right Page", is_right=True)

# Function to display an image on the canvas
def display_image(image, title, is_right=False):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)

    if is_right:
        canvas_right.create_image(0, 0, anchor=tk.NW, image=image_tk)
        canvas_right.image = image_tk
        canvas_right.config(scrollregion=canvas_right.bbox(tk.ALL))  # Update scroll region
    else:
        canvas_left.create_image(0, 0, anchor=tk.NW, image=image_tk)
        canvas_left.image = image_tk
        canvas_left.config(scrollregion=canvas_left.bbox(tk.ALL))  # Update scroll region

# Function to move to the next PDF page
def next_pdf_page():
    global current_page
    if current_page < total_pages - 1:
        current_page += 1
        process_page()

# Function to move to the previous PDF page
def previous_pdf_page():
    global current_page
    if current_page > 0:
        current_page -= 1
        process_page()

# Function to zoom in and out using scroll
def zoom(event):
    global zoom_factor
    zoom_step = 0.1
    if event.num == 4 or event.delta > 0:  # Scroll up to zoom in
        zoom_factor *= (1 + zoom_step)
    elif event.num == 5 or event.delta < 0:  # Scroll down to zoom out
        zoom_factor *= (1 - zoom_step)
    zoom_factor = max(0.5, min(zoom_factor, 3.0))  # Keep zoom between 0.5x and 3.0x
    update_image()

# Pan function for dragging
def start_pan(event):
    canvas_left.scan_mark(event.x, event.y)
    canvas_right.scan_mark(event.x, event.y)

def pan_image(event):
    canvas_left.scan_dragto(event.x, event.y, gain=1)
    canvas_right.scan_dragto(event.x, event.y, gain=1)

# Function to select PDF file
def select_pdf():
    global pdf_document, current_page, total_pages
    file_path = filedialog.askopenfilename(title="Select PDF File", filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        try:
            pdf_document = fitz.open(file_path)
            total_pages = pdf_document.page_count
            current_page = 0
            process_page()  # Process the first page
        except Exception as e:
            print(f"Error opening PDF: {e}")

# Function to save left or right page
def save_page(is_right=False):
    if not pdf_document:
        return
    page_number = current_page + 1
    image = right_page_image if is_right else left_page_image
    if image is not None:
        side = 'right' if is_right else 'left'
        filename = f"page_{page_number}_{side}.jpg"
        cv2.imwrite(filename, image)
        print(f"Saved {filename}")

# Function to process all pages automatically
def process_all_pages():
    global current_page
    for i in range(total_pages):
        current_page = i
        process_page()

# Initialize Tkinter GUI
root = tk.Tk()
root.title("PDF Processor with Line Segmentation")
root.geometry("1600x900")  # Set a larger window size
root.configure(bg='#2e2e2e')

# Create main frame with scrollbars
main_frame = tk.Frame(root, bg='#2e2e2e')
main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Scrollable canvas for displaying left image
canvas_left = Canvas(main_frame, bg='#1e1e1e', highlightthickness=0)
h_scroll_left = Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas_left.xview)
v_scroll_left = Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas_left.yview)
canvas_left.configure(xscrollcommand=h_scroll_left.set, yscrollcommand=v_scroll_left.set)

# Scrollable canvas for displaying right image
canvas_right = Canvas(main_frame, bg='#1e1e1e', highlightthickness=0)
h_scroll_right = Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas_right.xview)
v_scroll_right = Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas_right.yview)
canvas_right.configure(xscrollcommand=h_scroll_right.set, yscrollcommand=v_scroll_right.set)

# Grid arrangement for canvases and scrollbars
# Arrange canvases within the main frame
canvas_left.grid(row=0, column=0, sticky='nsew')
h_scroll_left.grid(row=1, column=0, sticky='ew')
v_scroll_left.grid(row=0, column=1, sticky='ns')

canvas_right.grid(row=0, column=2, sticky='nsew')
h_scroll_right.grid(row=1, column=2, sticky='ew')
v_scroll_right.grid(row=0, column=3, sticky='ns')

# Resize configuration for canvases in the main frame
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(2, weight=1)

# Bind zoom and pan events
if platform.system() == 'Linux':
    canvas_left.bind("<Button-4>", zoom)
    canvas_left.bind("<Button-5>", zoom)
    canvas_right.bind("<Button-4>", zoom)
    canvas_right.bind("<Button-5>", zoom)
else:
    canvas_left.bind("<MouseWheel>", zoom)
    canvas_right.bind("<MouseWheel>", zoom)

canvas_left.bind("<ButtonPress-1>", start_pan)
canvas_left.bind("<B1-Motion>", pan_image)
canvas_right.bind("<ButtonPress-1>", start_pan)
canvas_right.bind("<B1-Motion>", pan_image)

# Variables for the options
dpi_var = IntVar(value=100)
noise_threshold_var = IntVar(value=10)
intensity_threshold_var = IntVar(value=128)
remove_borders_var = BooleanVar(value=False)
deskew_var = BooleanVar(value=False)
show_raw_var = BooleanVar(value=False)
line_segmentation_var = BooleanVar(value=False)  # Set to False initially

# Variables for line segmentation parameters
padding_var = IntVar(value=20)
min_width_var = IntVar(value=100)
threshold_var = DoubleVar(value=0.2)
margin_var = DoubleVar(value=0.1)



# Add the OCR transcription button in the line segmentation mode
def toggle_line_segmentation_sliders():
    global need_to_run_craft
    if line_segmentation_var.get():
        line_segmentation_frame.pack(fill=tk.X, pady=5)
        ocr_button.pack(fill=tk.X, pady=5)
        need_to_run_craft = True
        perform_line_segmentation()
    else:
        line_segmentation_frame.pack_forget()
        ocr_button.pack_forget()
        process_page()


# Control panel frame on the right side of the right canvas
control_panel = tk.Frame(root, bg='#2e2e2e')
control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
style = ttk.Style()
style.configure('Modern.TButton', 
                font=('Arial', 14), 
                padding=10, 
                background='#4e4e4e', 
                foreground='white',
                relief='flat')
style.map('Modern.TButton', 
          background=[('active', '#8A2BE2'), ('pressed', '#5B2C6F')])

# Modern checkbutton styling with enlarged indicators
style.configure('Modern.TCheckbutton',
                font=('Arial', 12),
                background='#2e2e2e',
                foreground='white',
                indicatorcolor='#8A2BE2',  # Custom color for indicator
                indicatorsize=20,  # Increase size of checkbox
                padding=(10, 5))  # Adjust padding to make the checkbox more prominent
style.map('Modern.TCheckbutton',
          background=[('active', '#4e4e4e'), ('selected', '#8A2BE2')])

# Button to select PDF
select_pdf_button = ttk.Button(control_panel, text="Select PDF", command=select_pdf, style='Modern.TButton')
select_pdf_button.pack(fill=tk.X, pady=10)

create_custom_slider(control_panel, "DPI:", dpi_var, 50, 300, 1, command=process_page, fill_color="#dbff33")
# Reintroduce Noise and Intensity (See Through) sliders
create_custom_slider(control_panel, "Noise Threshold:", noise_threshold_var, 0, 100, 1, command=process_page, fill_color="#ff33bc")
create_custom_slider(control_panel, "See Through:", intensity_threshold_var, 0, 255, 1, command=process_page, fill_color="#FFD700")

# Toggle Buttons
remove_borders_check = ttk.Checkbutton(control_panel, text="Remove Borders", variable=remove_borders_var, 
                                       command=process_page, style='Modern.TCheckbutton')
remove_borders_check.pack(fill=tk.X, pady=5)

deskew_check = ttk.Checkbutton(control_panel, text="Deskew", variable=deskew_var, 
                               command=process_page, style='Modern.TCheckbutton')
deskew_check.pack(fill=tk.X, pady=5)

show_raw_button = ttk.Checkbutton(control_panel, text="Show Raw Pages", variable=show_raw_var, 
                                  command=process_page, style='Modern.TCheckbutton')
show_raw_button.pack(fill=tk.X, pady=5)

# Line Segmentation Toggle
line_segmentation_check = ttk.Checkbutton(control_panel, text="Line Segmentation", variable=line_segmentation_var, 
                                          command=toggle_line_segmentation_sliders, style='Modern.TCheckbutton')
line_segmentation_check.pack(fill=tk.X, pady=5)

# Frame for Line Segmentation Sliders (initially not packed)
line_segmentation_frame = tk.Frame(control_panel, bg='#2e2e2e')

# Line Segmentation Parameters (inside the frame)
# Line Segmentation Parameters (inside the frame)
# Line Segmentation Parameters (inside the frame)
create_custom_slider(line_segmentation_frame, "Padding", padding_var, 0, 500, 1, command=apply_line_segmentation, fill_color="#FFB6C1")
create_custom_slider(line_segmentation_frame, "Min Width", min_width_var, 0, 500, 1, command=apply_line_segmentation, fill_color="#87CEFA")
create_custom_slider(line_segmentation_frame, "Margin", margin_var, 0, 2, 0.01, command=apply_line_segmentation, fill_color="#98FB98")
create_custom_slider(line_segmentation_frame, "Threshold", threshold_var, 0, 2, 0.01, command=apply_line_segmentation, fill_color="#FFD700")


# Navigation buttons
previous_page_button = ttk.Button(control_panel, text="Previous PDF Page", command=previous_pdf_page, style='Modern.TButton')
previous_page_button.pack(fill=tk.X, pady=5)

next_page_button = ttk.Button(control_panel, text="Next PDF Page", command=next_pdf_page, style='Modern.TButton')
next_page_button.pack(fill=tk.X, pady=5)

# Save buttons for left and right pages
save_left_button = ttk.Button(control_panel, text="Save Left Page", command=lambda: save_page(is_right=False), style='Modern.TButton')
save_left_button.pack(fill=tk.X, pady=5)

save_right_button = ttk.Button(control_panel, text="Save Right Page", command=lambda: save_page(is_right=True), style='Modern.TButton')
save_right_button.pack(fill=tk.X, pady=5)

# Button to process all pages
process_all_button = ttk.Button(control_panel, text="Process All Pages", command=process_all_pages, style='Modern.TButton')
process_all_button.pack(fill=tk.X, pady=5)


# Create the OCR button
# Create the OCR button to appear when line segmentation is enabled
ocr_button = ttk.Button(control_panel, text="Run OCR", command=transcribe_and_display_ocr, style='Modern.TButton')



# Ensure initial visibility of line segmentation sliders
toggle_line_segmentation_sliders()

# Start the GUI loop
root.mainloop()
