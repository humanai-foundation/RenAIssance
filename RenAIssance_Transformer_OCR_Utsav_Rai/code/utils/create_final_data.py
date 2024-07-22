import os
import shutil
import argparse

def create_folders(line_segments_path, line_texts_path):
    if not os.path.exists(line_segments_path):
        os.makedirs(line_segments_path)
    if not os.path.exists(line_texts_path):
        os.makedirs(line_texts_path)

def get_next_index(line_segments_path):
    existing_files = os.listdir(line_segments_path)
    if not existing_files:
        return 0
    existing_indices = [int(f.split('.')[0]) for f in existing_files]
    return max(existing_indices) + 1

def process_single_folder(processed_book_path, transcription_path, line_segments_path, line_texts_path):
    line_image_count = get_next_index(line_segments_path)
    
    for page_num in sorted(os.listdir(processed_book_path), key=int):
        segment_page_path = os.path.join(processed_book_path, page_num)
        text_page_path = os.path.join(transcription_path, page_num)
        
        # Read the corresponding text file
        text_file_path = os.path.join(text_page_path, f"{page_num}.txt")
        with open(text_file_path, "r") as text_file:
            lines = text_file.readlines()
        
        for idx, line_image in enumerate(sorted(os.listdir(segment_page_path), key=lambda x: int(x.split('.')[0]))):
            image_src = os.path.join(segment_page_path, line_image)
            text_line = lines[idx].strip()
            
            # Create new image and text file names
            new_image_name = f"{line_image_count}.jpg"
            new_text_name = f"{line_image_count}.txt"
            
            # Copy the image to the new folder
            shutil.copy(image_src, os.path.join(line_segments_path, new_image_name))
            
            # Create a new text file with the corresponding line text
            with open(os.path.join(line_texts_path, new_text_name), "w") as new_text_file:
                new_text_file.write(text_line)
            
            line_image_count += 1

def process_folders(base_dir_processed, base_dir_transcription, line_segments_path, line_texts_path):
    for book_name in os.listdir(base_dir_processed):
        processed_book_path = os.path.join(base_dir_processed, book_name, "line_segments")
        transcription_book_path = os.path.join(base_dir_transcription, book_name)
        
        if os.path.isdir(processed_book_path) and os.path.isdir(transcription_book_path):
            process_single_folder(processed_book_path, transcription_book_path, line_segments_path, line_texts_path)

def main():
    parser = argparse.ArgumentParser(description='Process directory paths.')
    parser.add_argument('base_dir_processed', type=str, help='Base directory path for processed books')
    parser.add_argument('base_dir_transcription', type=str, help='Base directory path for transcription')
    parser.add_argument('line_segments_path', type=str, help='Path to All_line_segments folder')
    parser.add_argument('line_texts_path', type=str, help='Path to All_line_texts folder')
    
    args = parser.parse_args()
    
    base_dir_processed = args.base_dir_processed
    base_dir_transcription = args.base_dir_transcription
    line_segments_path = args.line_segments_path
    line_texts_path = args.line_texts_path

    create_folders(line_segments_path, line_texts_path)
    process_folders(base_dir_processed, base_dir_transcription, line_segments_path, line_texts_path)
    print("Processing completed successfully.")

if __name__ == "__main__":
    main()
