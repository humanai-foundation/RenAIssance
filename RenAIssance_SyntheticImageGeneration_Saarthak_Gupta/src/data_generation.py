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
        'book1': [
            ('denoise_image', {'method': 'bilateral'}),
            ('denoise_image', {'method': 'nlm'}),
        ],
        'book2': [
            ('ensure_300ppi', {'target_dpi': 150}),
            ('remove_bleed_dual_layer', {}),
            ('denoise_image', {'method': 'bilateral'}),
            ('denoise_image', {'method': 'nlm'}),
        ],
        'book3': [
            ('ensure_300ppi', {'target_dpi': 300}),
            ('remove_bleed_dual_layer', {}),
            ('denoise_image', {'method': 'bilateral'}),
        ],
        'book4': [
            ('ensure_300ppi', {'target_dpi': 300}),
            ('remove_bleed_dual_layer', {}),
            ('denoise_image', {'method': 'bilateral'}),
        ],
        'book5': [
            ('ensure_300ppi', {'target_dpi': 300}),
        ],
        'book6': [
            ('ensure_300ppi', {'target_dpi': 300}),
            ('remove_bleed_dual_layer', {}),
            ('denoise_image', {'method': 'bilateral'}),
        ],
        'book7': [
            ('remove_bleed_dual_layer', {}),
            ('denoise_image', {'method': 'nlm'}),
            ('denoise_image', {'method': 'bilateral'}),
        ],
        'book8': [
            ('remove_bleed_dual_layer', {}),
            ('denoise_image', {'method': 'nlm'}),
            ('denoise_image', {'method': 'bilateral'}),
        ],
        'book9': [
            ('ensure_300ppi', {'target_dpi': 300}),
            ('remove_bleed_dual_layer', {}),
            ('denoise_image', {'method': 'wiener'}),
            ('sharpen_image', {'method': 'laplacian'}),
        ],
        'book10': [
            ('ensure_300ppi', {'target_dpi': 300}),
            ('remove_bleed_dual_layer', {}),
            ('denoise_image', {'method': 'wiener'}),
            ('sharpen_image', {'method': 'laplacian'}),
        ],
        'book11': [
            ('ensure_300ppi', {'target_dpi': 300}),
            ('remove_bleed_dual_layer', {}),
            ('denoise_image', {'method': 'bilateral'}),
        ],
        'book12': [
            ('ensure_300ppi', {'target_dpi': 300}),
            ('remove_bleed_dual_layer', {}),
            ('denoise_image', {'method': 'bilateral'}),
            ('sharpen_image', {'method': 'laplacian'}),
        ],
    }