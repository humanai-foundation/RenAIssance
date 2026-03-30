import cv2
import numpy as np
from pathlib import Path

class DocumentPreprocessor:
    """Centralized preprocessing pipeline for historical OCR documents."""
    
    def __init__(self, target_height=32):
        self.target_height = target_height

    def read_image(self, image_path):
        """Reads an image in grayscale, safely handling Path objects."""
        path_str = str(image_path)
        # This MUST stay as cv2.imread!
        img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE) 
        if img is None:
            raise FileNotFoundError(f"Image not found at: {path_str}")
        return img

    def binarize(self, image):
        """Applies adaptive Otsu's thresholding."""
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def process(self, image_path):
        """Executes the standard historical preprocessing pipeline."""
        img = self.read_image(image_path)
        img = self.binarize(img)
        return img