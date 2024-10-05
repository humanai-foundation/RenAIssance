import fitz  # PyMuPDF
from PIL import Image, ImageTk, ImageEnhance
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Canvas, ttk, Scrollbar, IntVar, BooleanVar, HORIZONTAL, Scale, Spinbox
from deskew import determine_skew
import platform
import math
import os

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

        # Trigger the command to update the main application
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
        update_slider()

    # Bind updates
    variable.trace_add('write', update_slider)
    slider_canvas.bind('<B1-Motion>', move_slider)
    slider_canvas.bind('<Button-1>', move_slider)

    # Adjust the slider whenever the window resizes
    slider_canvas.bind('<Configure>', update_slider)

    # Initial update
    update_slider()

# Function to rotate an image by a given angle
def rotate(image: np.ndarray, angle: float, background: tuple) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

# Function to deskew an image
def deskew_image(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    if angle is not None:
        rotated = rotate(image, angle, (0, 0, 0))
        return rotated
    else:
        return image

# Function to preprocess an image by removing noise and adjusting intensity
def preprocess_image(image: np.ndarray, noise_removal_area_threshold: int, intensity_threshold: int) -> np.ndarray:
    # Ensure the image is in the correct format
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # If already grayscale

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 25)

    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    sizes = stats[1:, -1]
    new_image = np.zeros((labels.shape), np.uint8)

    # Apply component filtering based on size and intensity thresholds
    for i in range(1, num_labels):
        component_mask = (labels == i)
        component_intensity = np.mean(gray[component_mask])
        if sizes[i - 1] >= noise_removal_area_threshold and component_intensity <= intensity_threshold:
            new_image[component_mask] = 255

    inverted_image = cv2.bitwise_not(new_image)
    bordered = cv2.copyMakeBorder(inverted_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return bordered

# Function to remove borders from an image
def remove_borders(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # If already grayscale

    img_inverted = cv2.bitwise_not(gray)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    detected_horizontal = cv2.morphologyEx(img_inverted, cv2.MORPH_OPEN, horizontal_kernel)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    detected_vertical = cv2.morphologyEx(img_inverted, cv2.MORPH_OPEN, vertical_kernel)

    detected_lines = cv2.addWeighted(detected_horizontal, 1.0, detected_vertical, 1.0, 0.0)
    dilated_lines = cv2.dilate(detected_lines, np.ones((1, 1), np.uint8), iterations=2)
    closed_lines = cv2.morphologyEx(dilated_lines, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))

    _, binary_lines = cv2.threshold(closed_lines, 127, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(binary_lines, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    mask = np.zeros_like(gray)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(mask, (x1, y1), (x2, y2), 255, 10)

    dilated_mask = cv2.dilate(mask, np.ones((1, 1), np.uint8), iterations=2)
    img_result = gray.copy()
    img_result[dilated_mask == 255] = 255

    return img_result

# Function to process a single page of the PDF
def process_page():
    global processed_pages, raw_left_half, raw_right_half, left_page_image, right_page_image
    processed_pages.clear()  # Clear previous images
    right_page_image = None  # Reset right page image when processing a new page

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

                left_half = preprocess_image(left_half, noise_threshold_var.get(), intensity_threshold_var.get())
                right_half = preprocess_image(right_half, noise_threshold_var.get(), intensity_threshold_var.get())

            left_page_image = left_half if not show_raw_var.get() else raw_left_half
            right_page_image = right_half if not show_raw_var.get() else raw_right_half

            processed_pages.append(left_page_image)
            processed_pages.append(right_page_image)
        else:
            raw_left_half = image.copy()

            if not show_raw_var.get():
                if deskew_var.get():
                    image = deskew_image(image)

                if remove_borders_var.get():
                    image = remove_borders(image)

                image = preprocess_image(image, noise_threshold_var.get(), intensity_threshold_var.get())

            left_page_image = image if not show_raw_var.get() else raw_left_half
            right_page_image = None  # Ensure right page image is set to None
            processed_pages.append(left_page_image)

        update_image()  # Display the processed images
    except Exception as e:
        print(f"Error processing page {current_page + 1}: {e}")


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
root.title("PDF Processor")
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

create_custom_slider(control_panel, "DPI:", dpi_var, 50, 300, 1, command=lambda: process_page(), fill_color="#FFB6C1")


# Toggle Buttons
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


create_custom_slider(control_panel, "Noise", noise_threshold_var, 0, 100, 1, command=lambda: process_page(), fill_color="#87CEFA")

# Custom Intensity Threshold slider
create_custom_slider(control_panel, "See Through", intensity_threshold_var, 0, 255, 1, command=lambda: process_page())

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
# Start the GUI loop
root.mainloop()