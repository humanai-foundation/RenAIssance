import fitz
import cv2
import csv
import os
from docx import Document
import string
import numpy as np
from PIL import Image, ImageOps

def count_files_in_folder(folder_path, extensions_list):
    # Initialize counter for files
    file_count = 0

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file ends with the given file extension
        for extension in extensions_list:
            if filename.lower().endswith(extension):
                file_count += 1

    return file_count

def pdf_to_images(pdf_path, output_folder):
    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Iterate over each page in the PDF
    for page_number in range(len(pdf_document)):
        # Get the page
        page = pdf_document.load_page(page_number)

        # Render the page as a Pixmap
        pixmap = page.get_pixmap()

        # Save the Pixmap as a PNG image
        image_path = os.path.join(output_folder, f'page_{page_number + 1}.png')
        pixmap.save(image_path)

    # Close the PDF
    pdf_document.close()

def split_and_save_image(image_path, output_folder, last_image_number):
    # Read the image
    img = cv2.imread(image_path)

    # Get image width
    _, width, _ = img.shape

    # You can alter these, to get the optimum values for both
    width_for_single_page, width_for_dual_pages = 350, 450

    # Determine filename based on width and last image number
    if width < width_for_single_page:
        filename = f"image_{last_image_number}.png"
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)
        last_image_number += 1

    elif width > width_for_dual_pages:
        left_half = img[:, :width // 2]
        right_half = img[:, width // 2:]
        filename = f"image_{last_image_number}.png"
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, left_half)
        last_image_number += 1
        filename = f"image_{last_image_number}.png"
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, right_half)
        last_image_number += 1

    return last_image_number

def process_images(image_folder, output_folder):
    # Initialize image counter, starting from 1
    last_image_number = 1

    # Iterate through all files in the image folder
    for indx in range(count_files_in_folder(image_folder, [".png", ".jpg", ".jpeg"])):
        # Check for image files only
        filename = 'page_' + str(indx+1) + '.png'
        image_path = os.path.join(image_folder, filename)
        last_image_number = split_and_save_image(image_path, output_folder, last_image_number)

# Function to process bounding boxes
def process_bounding_boxes(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Parse bounding box coordinates
    bounding_boxes = []
    for line in lines:
        coords = list(map(int, line.strip().split(',')))
        bounding_boxes.append(coords)

    # Sort bounding boxes based on y_min value
    bounding_boxes.sort(key=lambda box: box[1])

    vertical_distance_between_lines = 10   #Change it according to the dataset, you are using
    # Group bounding boxes based on difference between max and min y_min values
    grouped_boxes = []
    current_group = []
    for box in bounding_boxes:
        if not current_group:
            current_group.append(box)
        else:
            min_y = min(current_group, key=lambda x: x[1])[1]
            max_y = max(current_group, key=lambda x: x[1])[1]
            if box[1] - min_y <= vertical_distance_between_lines:
                current_group.append(box)
            else:
                grouped_boxes.append(current_group)
                current_group = [box]

    # Append the last group
    if current_group:
        grouped_boxes.append(current_group)

    # Sort each group based on x_min value
    for group in grouped_boxes:
        group.sort(key=lambda box: box[0])

    return grouped_boxes

def remove_punctuation(text):
    # Define a translation table that maps punctuations to None
    translation_table = str.maketrans("", "", string.punctuation)
    # Remove punctuations using translate() method
    return text.translate(translation_table)

def save_pages_to_text(docx_file, output_file):
    document = Document(docx_file)
    all_text = ""
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        # Remove punctuations from the text before appending to the all_text variable
        # text_without_punctuation = remove_punctuation(text)
        # Check if the line starts with "PDF p"
        if not text.startswith("PDF p"):
            all_text += text + "\n"

    # Write all_text to the output file
    with open(output_file, "w") as file:
        file.write(all_text)

def count_occurrences_of_semicolon(filename):
    with open(filename, 'r') as file:
        content = file.read()
        return content.count(';')

def read_nth_line(file_path, n):
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i == n - 1:
                return line.rstrip('\n')  # Remove newline character from the end of the line
    return None

def process_textfiles(textfile, sorted_BoundBox_folder, output_folder):
  # Initialize textfile counter, starting from 1
  last_textfile_number = 7
  print(str(count_files_in_folder(sorted_BoundBox_folder,['.txt'])) + ' files in the folder')
  # Iterate through all files in the textfile folder
  for indx in range(count_files_in_folder(sorted_BoundBox_folder, ['.txt'])-6):   # 6 pages left for testing
    # Check for textfile files only
    filename = 'res_image_' + str(indx+1) + '_sorted.txt'
    output_file = os.path.join(output_folder, 'res_image_' + str(indx+1) + '_actual.txt')
    res_image_sorted_path = os.path.join(sorted_BoundBox_folder, filename)
    occurrences = count_occurrences_of_semicolon(res_image_sorted_path) - 2
    with open(output_file, 'w') as output:
        for eachline in range(occurrences):
            content = read_nth_line(textfile, eachline+last_textfile_number)
            output.write(content + '\n')
    last_textfile_number = last_textfile_number + occurrences

def extract_bounding_boxes(image_path, bounding_boxes_file, output_folder, word):
    # Read the main image
    main_image = cv2.imread(image_path)
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read bounding box coordinates from the text file
    with open(bounding_boxes_file, 'r') as f:
        bounding_boxes_data = f.read().split(';')
    bounding_boxes_data = bounding_boxes_data[1:]

    for indx in range(len(bounding_boxes_data)-1):
        bounding_box_coords = bounding_boxes_data[indx].strip().split('\n')
        for cnt in range(len(bounding_box_coords)):
            coordinates_list = [int(coord) for coord in bounding_box_coords[cnt].split(',')]
            x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max = coordinates_list

            # Extract the bounding box from the main image
            bounding_box = main_image[y_min:y_max, x_min:x_max]

            # Save the bounding box as a separate image
            output_path = os.path.join(output_folder, f'{word}.png')
            cv2.imwrite(output_path, bounding_box)

            word += 1

    return word

def extract_bounding_boxes_train(image_path, bounding_boxes_file, text_file, output_folder):
    # Read the main image
    main_image = cv2.imread(image_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read bounding box coordinates from the text file
    with open(bounding_boxes_file, 'r') as f:
        bounding_boxes_data = f.read().split(';')
    bounding_boxes_data = bounding_boxes_data[1:]
    #first value is for page number so skip it
    # Read text data from the text file
    with open(text_file, 'r') as f:
        text_data = f.read().split('\n')

    for indx in range(len(text_data)):
        words = text_data[indx].split(' ')
        bounding_box_coords = bounding_boxes_data[indx].strip().split('\n')
        for cnt in range(min(len(words),len(bounding_box_coords))):
            coordinates_list = [int(coord) for coord in bounding_box_coords[cnt].split(',')]
            x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max = coordinates_list

            # Extract the bounding box from the main image
            bounding_box = main_image[y_min:y_max, x_min:x_max]

            # Save the bounding box as a separate image
            output_path = os.path.join(output_folder, f'{words[cnt]}.png')
            cv2.imwrite(output_path, bounding_box)

def apply_extraction_to_folder_for_train(image_folder, bounding_box_folder, text_folder, output_folder):
    # Iterate over each image in the image folder
    # for image_filename in os.listdir(image_folder):
    for image in range(count_files_in_folder(image_folder, ['.png', '.jpeg', '.jpg'])- 6): #last 6 pages are for testing
        image_filename = 'image_' + str(image+1) + '.png'
        if image_filename.endswith('.png') or image_filename.endswith('.jpg'):
            # Get the shared number before the extension
            image_base_name = os.path.splitext(image_filename)[0]
            bounding_box_filename = "res_" + image_base_name + '_sorted.txt'
            text_filename = "res_" + image_base_name + '_actual.txt'
            # Check if the corresponding bounding box file exists
            bounding_box_path = os.path.join(bounding_box_folder, bounding_box_filename)
            actual_text_path = os.path.join(text_folder, text_filename)
            if os.path.exists(bounding_box_path):
                image_path = os.path.join(image_folder, image_filename)
                # Apply bounding box extraction to each image
                extract_bounding_boxes_train(image_path, bounding_box_path, actual_text_path, output_folder)
            else:
                print(f'Bounding box file for {image_filename} does not exist.')

def apply_extraction_to_folder_for_test(image_folder, bounding_box_folder, output_folder, word):
    for image in range(count_files_in_folder(image_folder, ['.png', '.jpeg', '.jpg']) - 25):
        image_filename = 'image_' + str(image+26) + '.png'
        if image_filename.endswith('.png') or image_filename.endswith('.jpg'):
            image_base_name = os.path.splitext(image_filename)[0]
            print(image_base_name)
            bounding_box_filename = "res_" + image_base_name + '_sorted.txt'
            bounding_box_path = os.path.join(bounding_box_folder, bounding_box_filename)
            if os.path.exists(bounding_box_path):
                image_path = os.path.join(image_folder, image_filename)
                word = extract_bounding_boxes(image_path, bounding_box_path, output_folder, word)
            else:
                print(f'Bounding box file for {image_filename} does not exist.')

                
def create_csv_from_folder(folder_path, csv_file_path):
    # Get a list of all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Create or overwrite the CSV file
    with open(csv_file_path, 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the header row
        csv_writer.writerow(['FILENAME', 'IDENTITY'])

        # Write data rows, excluding files with the name ".png"
        for file_name in files:
            if file_name.lower() == ".png":
                continue  # Skip files with the name ".png"

            # file_path = os.path.join(folder_path, file_name)

            # Remove the file extension (assuming it's three characters long, like '.png')
            file_name_without_extension = os.path.splitext(file_name)[0]

            csv_writer.writerow([file_name, file_name_without_extension])

    print(f'CSV file "{csv_file_path}" created successfully.')

def pad_and_resize_images(folder_path):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        raise ValueError(f"The folder {folder_path} does not exist")

    # Define the target aspect ratio and size
    target_aspect_ratio = 4  # 1:4 aspect ratio
    target_width = 200
    target_height = 40

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                # Open the image
                with Image.open(file_path) as img:
                    img = img.convert('L')
                    width, height = img.size
                    aspect_ratio = width / height

                    if aspect_ratio < target_aspect_ratio:
                        # Calculate padding to make aspect ratio 1:4
                        new_width = height * 4
                        padding = (new_width - width) // 2
                        padded_img = ImageOps.expand(img, border=(padding, 0, padding, 0), fill='white')
                    else:
                        padded_img = img

                    # Resize the image to 200x40
                    resized_img = padded_img.resize((target_width, target_height))

                    # Save the processed image back to the original path
                    resized_img.save(file_path)

                    print(f"Processed and replaced: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")