import cv2
import numpy as np
import os
import glob
import tkinter as tk
from tkinter import Scale, Label, Button, IntVar, DoubleVar, Canvas, HORIZONTAL
from tkinter.ttk import Style
from PIL import Image, ImageTk
import platform
from tkinter import HORIZONTAL, DoubleVar, IntVar
from PIL import Image, ImageTk
import platform
from tkinter import ttk

# Function to create a custom slider with text on the left, slider in the middle, and value on the right
# Function to create a custom slider with text on the left, slider in the middle, and value on the right
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


def configure_button_style():
    style = ttk.Style()
    style.theme_use("clam")  # Use a modern-looking theme

    # Configure the style for the TButton
    style.configure("Modern.TButton",
                    font=("Helvetica", 10, "bold"),  # Font style
                    foreground="white",              # Text color
                    background="#4e4e4e",            # Background color
                    borderwidth=1,
                    relief="flat",
                    padding=(10, 5),                 # Padding inside the button
                    anchor="center")                 # Center the text

    # Add hover effect for buttons
    style.map("Modern.TButton",
              background=[("active", "#646464")],  # Hover color
              relief=[("pressed", "groove"), ("!pressed", "flat")])

# Read contour points from a file
def read_contour_points(file_path):
    contour_points = []
    with open(file_path, 'r') as file:
        for line in file:
            points = list(map(int, line.strip().split(',')))
            contour_points.append(points)
    return contour_points

# Function to generate bounding boxes around contours
def get_bounding_boxes(contours, img_width, img_height, padding=10, min_width=20, margin=0.1):
    bounding_boxes = []
    top_margin = img_height * margin
    bottom_margin = img_height * (1 - margin)

    for contour in contours:
        points = np.array(contour).reshape((-1, 2))
        x, y, w, h = cv2.boundingRect(points)
        if w > min_width and (y > top_margin and y + h < bottom_margin):
            x = max(x - padding, 0)
            w = min(w + 2 * padding, img_width - x)
            bounding_boxes.append((x, y, x + w, y + h))

    centers = [(x1 + (x2 - x1) // 2) for (x1, y1, x2, y2) in bounding_boxes]
    median_center = np.median(centers)
    
    filtered_boxes = []
    for (x1, y1, x2, y2) in bounding_boxes:
        center = x1 + (x2 - x1) // 2
        if abs(center - median_center) < 800:
            filtered_boxes.append((x1, y1, x2, y2))
    return filtered_boxes

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, bounding_boxes, color=(0, 255, 0), thickness=6):
    for (x1, y1, x2, y2) in bounding_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

# Function to split tall bounding boxes into smaller ones based on a threshold and visualize them in red
def split_bounding_boxes(image, bounding_boxes, threshold=0.8):
    heights = [y2 - y1 for (x1, y1, x2, y2) in bounding_boxes]
    median_height = np.median(heights)
    new_bounding_boxes = []
    split_bounding_boxes = []

    for (x1, y1, x2, y2) in bounding_boxes:
        height = y2 - y1
        ratio = height / median_height
        if ratio > 1 + threshold:  # Identify tall bounding boxes
            split_number = round(ratio)
            split_height = height // split_number
            for i in range(split_number):
                new_y1 = y1 + i * split_height
                new_y2 = new_y1 + split_height if i < split_number - 1 else y2
                split_bounding_boxes.append((x1, new_y1, x2, new_y2))
        else:
            new_bounding_boxes.append((x1, y1, x2, y2))
    
    # Visualize the split bounding boxes in red
    draw_bounding_boxes(image, split_bounding_boxes, color=(255, 0, 0))
    
    return new_bounding_boxes + split_bounding_boxes

# Function to save settings and apply them to all images
def apply_to_all():
    for img_path, contour_path in zip(image_paths, contour_paths):
        image = cv2.imread(img_path)
        img_height, img_width = image.shape[:2]
        contours = read_contour_points(contour_path)

        bounding_boxes = get_bounding_boxes(
            contours,
            img_width,
            img_height,
            padding=padding.get(),
            min_width=min_width.get(),
            margin=margin.get()
        )
        draw_bounding_boxes(image, bounding_boxes)

        # Save processed image
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, image)
    print("Applied settings to all images and saved.")

# Navigation functions for image indexing
def navigate(step):
    global img_index
    img_index = (img_index + step) % len(image_paths)
    update_image()

def filter_and_adjust_bounding_boxes(bounding_boxes):
    x1s = [x1 for (x1, y1, x2, y2) in bounding_boxes]
    x2s = [x2 for (x1, y1, x2, y2) in bounding_boxes]
    median_x1 = int(np.median(x1s)) - 30  # Calculate median x1
    median_x2 = int(np.median(x2s)) + 20  # Calculate median x2 with a small adjustment

    adjusted_boxes = []
    for (x1, y1, x2, y2) in bounding_boxes:
        adjusted_boxes.append((median_x1, y1, median_x2, y2))  # Adjust bounding boxes to median x1 and x2

    # Remove overlapping bounding boxes, keeping only the one with the greater width
    non_overlapping_boxes = []
    for box in adjusted_boxes:
        overlap = False
        for other_box in non_overlapping_boxes:
            x1, y1, x2, y2 = box
            ox1, oy1, ox2, oy2 = other_box
            
            # Calculate intersection area
            inter_x1 = max(x1, ox1)
            inter_y1 = max(y1, oy1)
            inter_x2 = min(x2, ox2)
            inter_y2 = min(y2, oy2)
            
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            box_area = (x2 - x1) * (y2 - y1)
            other_box_area = (ox2 - ox1) * (oy2 - oy1)
            
            if inter_area > 0.9 * min(box_area, other_box_area):  # Check for 90% overlap
                overlap = True
                if box_area > other_box_area:
                    non_overlapping_boxes.remove(other_box)
                    non_overlapping_boxes.append(box)
                break
        if not overlap:
            non_overlapping_boxes.append(box)
    
    return non_overlapping_boxes


# Zoom function triggered by mouse scroll
def zoom(event):
    global zoom_factor
    zoom_step = 0.1
    # Adjust zoom factor based on scroll direction
    if event.num == 4 or event.delta > 0:  # Scroll up to zoom in
        zoom_factor *= (1 + zoom_step)
    elif event.num == 5 or event.delta < 0:  # Scroll down to zoom out
        zoom_factor *= (1 - zoom_step)
    zoom_factor = max(0.5, min(zoom_factor, 3.0))  # Keep zoom between 0.5x and 3.0x
    update_image()

# Pan function for dragging
def start_pan(event):
    canvas.scan_mark(event.x, event.y)

def pan_image(event):
    canvas.scan_dragto(event.x, event.y, gain=1)

# Function to update and visualize bounding boxes based on parameters
def update_image():
    global img_index, offset_x, offset_y
    img_path = image_paths[img_index]
    contour_path = contour_paths[img_index]
    original_image = cv2.imread(img_path)
    img_height, img_width = original_image.shape[:2]
    contours = read_contour_points(contour_path)

    # Get bounding boxes
    bounding_boxes = get_bounding_boxes(
        contours,
        img_width,
        img_height,
        padding=padding.get(),
        min_width=min_width.get(),
        margin=margin.get()
    )

    # Split bounding boxes with threshold and mark them with red color
    new_bounding_boxes = split_bounding_boxes(original_image, bounding_boxes, threshold=threshold.get())

    # Filter and adjust bounding boxes
    adjusted_bounding_boxes = filter_and_adjust_bounding_boxes(new_bounding_boxes)

    # Draw bounding boxes
    display_image = original_image.copy()
    draw_bounding_boxes(display_image, bounding_boxes, color=(0, 0, 255))  # Draw regular boxes in blue
    draw_bounding_boxes(display_image, adjusted_bounding_boxes, color=(0, 255, 0))  # Draw split boxes in green

    # Draw median lines
    x1s = [x1 for (x1, y1, x2, y2) in adjusted_bounding_boxes]
    x2s = [x2 for (x1, y1, x2, y2) in adjusted_bounding_boxes]
    median_x1 = int(np.median(x1s)) - 30  # Left median x with adjustment
    median_x2 = int(np.median(x2s)) + 20  # Right median x with adjustment

    # Draw left and right median x lines in cyan color
    cv2.line(display_image, (median_x1, 0), (median_x1, img_height), (0, 215, 255), 15)
    cv2.line(display_image, (median_x2, 0), (median_x2, img_height), (0, 215, 255), 15)

    # Resize image based on the zoom factor
    zoomed_width = int(img_width * zoom_factor)
    zoomed_height = int(img_height * zoom_factor)
    resized_image = cv2.resize(display_image, (zoomed_width, zoomed_height))

    # Convert image for Tkinter display
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)

    # Update canvas with the zoomed and panned image
    canvas.config(scrollregion=(0, 0, image_tk.width(), image_tk.height()))
    canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=image_tk)
    canvas.image = image_tk

# Initialize Tkinter GUI
root = tk.Tk()
root.title("Bounding Box Visualizer")
root.geometry("1024x768")  # Set the initial size of the window
root.configure(bg='#2e2e2e')  # Set dark theme background

# Initialize variables
zoom_factor = 1.0
offset_x = 0
offset_y = 0

# Variables for parameters
padding = IntVar(value=20)
min_width = IntVar(value=100)
threshold = DoubleVar(value=0.2)
margin = DoubleVar(value=0.1)

# Directories and paths
image_dir = "/home/utsav/Documents/GSoC/Data/datasetB/test/b2"
contour_dir = "/home/utsav/mygit/CRAFT-pytorch/result"
output_dir = "/home/utsav/Documents/GSoC/Data/datasetB/test/b2"

image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
contour_paths = [os.path.join(contour_dir, f"res_{os.path.splitext(os.path.basename(p))[0]}.txt") for p in image_paths]
img_index = 0

# Create a scrollable canvas to display images
canvas = tk.Canvas(root, bg='#1e1e1e', highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)

# Create custom sliders with different pastel colors
# Create custom sliders with different pastel colors and correct command integration
# Create custom sliders with different pastel colors and full width
create_custom_slider(root, "Padding", padding, 0, 500, 1, lambda: update_image(), fill_color="#FFB6C1")
create_custom_slider(root, "Min Width", min_width, 0, 500, 1, lambda: update_image(), fill_color="#87CEFA")
create_custom_slider(root, "Margin", margin, 0, 2, 0.01, lambda: update_image(), fill_color="#98FB98")
create_custom_slider(root, "Threshold", threshold, 0, 2, 0.01, lambda: update_image(), fill_color="#FFD700")



# Buttons for navigation and application with dark theme styling
button_frame = tk.Frame(root, bg='#2e2e2e')
button_frame.pack(pady=10)
configure_button_style()

# Create modern buttons using ttk with the new style
ttk.Button(button_frame, text="Previous Image", command=lambda: navigate(-1), style="Modern.TButton").pack(side="left", padx=5)
ttk.Button(button_frame, text="Next Image", command=lambda: navigate(1), style="Modern.TButton").pack(side="left", padx=5)
ttk.Button(button_frame, text="Apply to All", command=apply_to_all, style="Modern.TButton").pack(side="right", padx=5)


# Bind events for zooming and panning based on platform
if platform.system() == 'Linux':
    canvas.bind("<Button-4>", zoom)
    canvas.bind("<Button-5>", zoom)
else:
    canvas.bind("<MouseWheel>", zoom)

canvas.bind("<ButtonPress-1>", start_pan)
canvas.bind("<B1-Motion>", pan_image)

# Start the GUI loop
update_image()
root.mainloop()
