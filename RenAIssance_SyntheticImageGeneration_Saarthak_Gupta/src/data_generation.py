import warnings
warnings.filterwarnings("ignore")
import os
from data_utils import *

def data_generation_pipeline():
    
    # 1. Split books and transcripts into individual files
    process_books_with_transcripts(
        input_books_folder="data/GAN-DATA/1_raw/books/",
        input_transcripts_folder="data/GAN-DATA/1_raw/transcripts/",
        output_books_folder="data/GAN-DATA/2_splitted/books/",
        output_transcripts_folder="data/GAN-DATA/2_splitted/transcripts/"
    )

    # 2. Preprocessing images
    copy_all_transcripts()
    copy_all_images()

    book_transformations = {
        'book1': {
            'denoise_image': {'method': 'bilateral'},
            'denoise_image': {'method': 'nlm'}
        },
        'book2': {
            'ensure_300ppi': {'target_dpi': 150},
            'remove_bleed_dual_layer': {},
            'denoise_image': {'method': 'bilateral'},
            'denoise_image': {'method': 'nlm'},
            'denoise_image': {'method': 'nlm'}
        },
        'book3': {
            'ensure_300ppi': {'target_dpi': 300},
            'remove_bleed_dual_layer': {},
            'denoise_image': {'method': 'bilateral'}
            },
        'book4': {
            'ensure_300ppi': {'target_dpi': 300},
            'remove_bleed_dual_layer': {},
            'denoise_image': {'method': 'bilateral'}
        },
        'book5': {
            'ensure_300ppi': {'target_dpi': 300}
        },
        'book6': {
            'ensure_300ppi': {'target_dpi': 300},
            'remove_bleed_dual_layer': {},
            'denoise_image': {'method': 'bilateral'}
        },
        'book7': {
            'remove_bleed_dual_layer': {},
            'denoise_image': {'method': 'nlm'},
            'denoise_image': {'method': 'bilateral'}
        },
        'book8': {
            'remove_bleed_dual_layer': {},
            'denoise_image': {'method': 'nlm'},
            'denoise_image': {'method': 'bilateral'}
        },
        'book9': {
            'ensure_300ppi': {'target_dpi': 300},
            'remove_bleed_dual_layer': {},
            'denoise_image': {'method': 'wiener'},
            'sharpen_image': {'method': 'laplacian'}
        },
        'book10': {
            'ensure_300ppi': {'target_dpi': 300},
            'remove_bleed_dual_layer': {},
            'denoise_image': {'method': 'wiener'},
            'sharpen_image': {'method': 'laplacian'}
        },
        'book11': {
            'ensure_300ppi': {'target_dpi': 300},
            'remove_bleed_dual_layer': {},
            'denoise_image': {'method': 'bilateral'},
        },
        'book12': {
            'ensure_300ppi': {'target_dpi': 300},
            'remove_bleed_dual_layer': {},
            'denoise_image': {'method': 'bilateral'},
            'sharpen_image': {'method': 'laplacian'},
        },
    }

    stats = process_multiple_books(
        books_base_dir="data/GAN-DATA/2_splitted/books", 
        output_dir="data/GAN-DATA/3_processed/books",
        book_transformations=book_transformations
    )
    
    # 3. Generate bounding boxes for each book page
    input_root = "data/GAN-DATA/3_processed/books"
    output_root = "data/GAN-DATA/4_bounding_boxes//"
    model_path = "CRAFT-pytorch/weights/craft_mlt_25k.pth"

    text_detection(
        input_root=input_root,
        output_root=output_root,
        model_path=model_path
    )

    plot_random_pages(
        images_dir="data/GAN-DATA/3_processed/books",
        bbox_dir="data/GAN-DATA/4_bounding_boxes",
        save_dir="assets/plots/"
    )
    # 4. Mapping bounding boxes to transcripts
    # This function will map the bounding boxes generated in the previous step to the corresponding words in the transcripts.
    similarity_threshold = 0.8
    
    total_words, total_bboxes, total_mapped = mapping_bounding_boxes(
        image_dir="data/GAN-DATA/3_processed/books",
        bbox_dir="data/GAN-DATA/4_bounding_boxes", 
        transcript_dir="data/GAN-DATA/3_processed/transcripts",
        output_dir="data/GAN-DATA/5_mapped",
        tesseract_model=1,
        similarity_threshold=0.8
    )

    # 5. Extract and process all regions to create word data
    # This will extract words from the mapped bounding boxes and save them in the specified output directory.
    df = extract_and_process_all_regions(
        image_root = "data/GAN-DATA/3_processed/books",
        aligned_root = "data/GAN-DATA/5_mapped",
        output_root = "data/GAN-DATA/6_word_data/images",
        csv_output_path = "data/GAN-DATA/6_word_data/words.csv"
    )
        
    # 6. Resize and pad images to a fixed size

    target_height, target_width = 64 , 128

    resize_and_pad(
        input_dir="data/GAN-DATA/6_word_data/images",
        output_dir="data/GAN-DATA/final_dataset/target",
        target_height=target_height,
        target_width=target_width
    )

    # 7. render source text images using matplotlib
    config = {
                'text.usetex': False,
                'mathtext.fontset': 'cm',
                'font.family': 'serif',
                'font.serif': ['Times New Roman'],
                'font.weight': 'bold',
                'mathtext.bf': 'bold',
            }

    
    df = generate_text_image_dataset(
    csv_path='data/GAN-DATA/6_word_data/words.csv',
    output_dir='data/GAN-DATA/final_dataset',
    target_dir_prefix='data/GAN-DATA/final_dataset/target/',
    image_width=target_width,
    image_height=target_height,
    base_fontsize=70,
    matplotlib_config=config,
    custom_font_path='fonts/RomanAntique.ttf', 
    progress_bar=True
    )
    
    
    df = pd.read_csv("data/GAN-DATA/final_dataset/data.csv")
    grid_df = create_image_grids(
        df=df[df['book'].isin(["book_1", "book_5"])],
        output_directory="data/GAN-DATA/grid_dataset",
        num_grids=3000,
        grid_size=(4,2),
        target_image_size=(128,64),  # (width, height)
        random_seed=42
    )
    
    
if __name__ == "__main__":        
        data_generation_pipeline()
        