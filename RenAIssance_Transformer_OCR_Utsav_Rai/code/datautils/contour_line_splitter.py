import cv2
import numpy as np
import os
import argparse
import glob

# Function to read contour points from a file
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
    top_margin = img_height * margin  # Define top margin
    bottom_margin = img_height * (1 - margin)  # Define bottom margin

    for contour in contours:
        points = np.array(contour).reshape((-1, 2))
        x, y, w, h = cv2.boundingRect(points)
        if w > min_width and (y > top_margin and y + h < bottom_margin):
            x = max(x - padding, 0)  # Add padding to x-coordinate
            w = min(w + 2 * padding, img_width - x)  # Add padding to width
            bounding_boxes.append((x, y, x + w, y + h))
    
    # Calculate the median center of the bounding boxes
    centers = [(x1 + (x2 - x1) // 2) for (x1, y1, x2, y2) in bounding_boxes]
    median_center = np.median(centers)
    
    filtered_boxes = []
    for (x1, y1, x2, y2) in bounding_boxes:
        center = x1 + (x2 - x1) // 2
            
        if abs(center - median_center) < 800:  # Preserve short lines near the center
            print(f"center diff: {abs(center - median_center)}")
            filtered_boxes.append((x1, y1, x2, y2))
        else:
            print(f"center diff for skipped: {abs(center - median_center)}")
        
    
    return filtered_boxes

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, bounding_boxes, color=(0, 255, 0), thickness=10):
    for i, (x1, y1, x2, y2) in enumerate(bounding_boxes):
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, str(i + 1), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Function to split tall bounding boxes into smaller ones based on a threshold
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
    
    return new_bounding_boxes + split_bounding_boxes

# Function to adjust bounding boxes to have consistent x-coordinates
# Function to filter and adjust bounding boxes
def filter_and_adjust_bounding_boxes(bounding_boxes):
    x1s = [x1 for (x1, y1, x2, y2) in bounding_boxes]
    x2s = [x2 for (x1, y1, x2, y2) in bounding_boxes]
    median_x1 = int(np.median(x1s))-30  # Calculate median x1
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

# Function to save individual line segment images
def save_line_segments(image, bounding_boxes, output_dir, base_name):
    for i, (x1, y1, x2, y2) in enumerate(bounding_boxes):
        segment = image[y1:y2, x1:x2]  # Crop the image to the bounding box
        segment_path = os.path.join(output_dir, f"{i+1}.jpg")
        cv2.imwrite(segment_path, segment)  # Save the cropped image
        print(f"Saved line segment to {segment_path}")

# Main function to process directories and apply bounding box logic
def process_directory(image_dir, contour_dir, output_dir, padding=20, min_width=100, threshold=0.2, margin=0.1, visualize=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it doesn't exist
        print(f"Created output directory: {output_dir}")

    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        base_index = os.path.splitext(base_name)[0]
        contour_file_path = os.path.join(contour_dir, f"res_{base_index}.txt")

        if os.path.exists(contour_file_path):
            print(f"Processing {image_path} with contours {contour_file_path}")

            contour_points = read_contour_points(contour_file_path)  # Read contour points
            image = cv2.imread(image_path)  # Read image
            img_height, img_width = image.shape[:2]

            # Generate initial bounding boxes, preserving middle short lines
            bounding_boxes = get_bounding_boxes(contour_points, img_width, img_height, padding=padding, min_width=min_width, margin=margin)
            # Split tall bounding boxes if needed
            new_bounding_boxes = split_bounding_boxes(image, bounding_boxes, threshold=threshold)
            # Adjust bounding boxes to have consistent x-coordinates
            adjusted_bounding_boxes = filter_and_adjust_bounding_boxes(new_bounding_boxes)

            # Sort bounding boxes from top to bottom based on y-coordinate
            adjusted_bounding_boxes = sorted(adjusted_bounding_boxes, key=lambda box: box[1])

            if visualize:
                draw_bounding_boxes(image, adjusted_bounding_boxes, color=(0, 255, 0))  # Draw bounding boxes if visualization is enabled
                output_image_path = os.path.join(output_dir, base_name)
                cv2.imwrite(output_image_path, image)  # Save the image with bounding boxes
                print(f"Saved visualized image to {output_image_path}")

            # Save individual line segment images
            line_segments_dir = os.path.join(output_dir, base_index)
            if not os.path.exists(line_segments_dir):
                os.makedirs(line_segments_dir)  # Create directory for line segments
            save_line_segments(image, adjusted_bounding_boxes, line_segments_dir, base_index)  # Save line segments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and contour files to draw bounding boxes and save the processed images.")
    parser.add_argument("image_dir", type=str, help="Directory containing the input images")
    parser.add_argument("contour_dir", type=str, help="Directory containing the contour text files")
    parser.add_argument("output_dir", type=str, help="Directory to save the processed images")
    parser.add_argument("--padding", type=int, default=50, help="Padding around bounding boxes (default: 20)")
    parser.add_argument("--min_width", type=int, default=100, help="Minimum width of bounding boxes (default: 100)")
    parser.add_argument("--threshold", type=float, default=0.2, help="Threshold for splitting bounding boxes (default: 0.2)")
    parser.add_argument("--margin", type=float, default=0.1, help="Margin to exclude short lines at the top and bottom (default: 0.1)")
    parser.add_argument("--visualize", action='store_true', help="Option to visualize bounding boxes on images")

    args = parser.parse_args()
    process_directory(args.image_dir, args.contour_dir, args.output_dir, padding=args.padding, min_width=args.min_width, threshold=args.threshold, margin=args.margin, visualize=args.visualize)
