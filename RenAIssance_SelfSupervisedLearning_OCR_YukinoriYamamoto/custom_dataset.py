import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import (
    RandomApply, GaussianBlur, Resize, Compose, ToTensor, Lambda,
    RandomPerspective, RandomAffine, RandomRotation, Grayscale
)
from PIL import Image
import os
import numpy as np
import random
from torch.utils.data import Dataset


class RandomVerticalCrop:
    """
    Randomly crops the top and bottom of the image based on a ratio.
    """
    def __init__(self, crop_height_ratio):
        self.crop_height_ratio = crop_height_ratio

    def __call__(self, img):
        (width, height) = img.size
        # content to remove (split between top and bottom)
        crop_amount = int(self.crop_height_ratio * height)
        
        # Ensure we don't crop the whole image if ratio is too high
        if crop_amount >= height:
            crop_amount = height // 2

        # Randomize how much to take from top vs bottom
        top_cut = random.randint(0, crop_amount // 2)
        bottom_cut = random.randint(0, crop_amount // 2)
        
        start = top_cut
        end = height - bottom_cut
        
        return img.crop((0, start, width, end))


class ResizeAndPad:
    """
    Resizes image to target size while maintaining aspect ratio, 
    then pads the remainder with white (255) or specified color.
    """
    def __init__(self, target_size, fill=255):
        # target_size should be (height, width) to match torchvision convention
        self.target_height, self.target_width = target_size
        self.fill = fill

    def __call__(self, img):
        img_width, img_height = img.size
        
        original_aspect = img_width / img_height
        target_aspect = self.target_width / self.target_height

        if original_aspect > target_aspect:
            # Width is the limiter
            new_width = self.target_width
            new_height = round(new_width / original_aspect)
        else:
            # Height is the limiter
            new_height = self.target_height
            new_width = round(new_height * original_aspect)

        # Resize
        resized_img = F.resize(img, (new_height, new_width))

        # Calculate padding
        pad_vert = self.target_height - new_height
        pad_top = pad_vert // 2
        pad_bot = pad_vert - pad_top
        
        pad_horz = self.target_width - new_width
        pad_left = pad_horz // 2
        pad_right = pad_horz - pad_left

        # Apply padding (left, top, right, bottom)
        return F.pad(resized_img, (pad_left, pad_top, pad_right, pad_bot), fill=self.fill)


class ContrastiveLearningDataset(Dataset):
    def __init__(self, img_dir, crop_height_ratio=0.2, img_size=(64, 384)):
        super().__init__()
        self.img_size = img_size
        assert os.path.isdir(img_dir), f"{img_dir} is not a valid directory"
        
        self.filepaths = [
            os.path.join(img_dir, filename) for filename in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, filename))
        ]
        
        if not self.filepaths:
            raise ValueError(f"No image files found in {img_dir}")

        # Transforms
        # Note: Using ResizeAndPad prevents text distortion
        self.original_transform = Compose([
            Lambda(lambda img: img.convert("RGB")),
            ResizeAndPad(img_size, fill=255), 
            Grayscale(num_output_channels=3),
            ToTensor(),
        ])
        
        self.augmented_transform = Compose([
            Lambda(lambda img: img.convert("RGB")),
            RandomApply([RandomVerticalCrop(crop_height_ratio=crop_height_ratio)], p=0.5),
            RandomRotation(degrees=3.5, fill=255),
            RandomApply([GaussianBlur(kernel_size=3)], p=0.3),
            ResizeAndPad(img_size, fill=255),
            Grayscale(num_output_channels=3),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        try:
            img = Image.open(path)
            # Force loading data to catch corrupt files immediately
            img.load() 
        except Exception as e:
            print(f"Warning: Could not load image {path}. Error: {e}")
            # Recursively pick a random valid image to prevent batch crash
            new_idx = random.randint(0, len(self.filepaths) - 1)
            return self.__getitem__(new_idx)

        original = self.original_transform(img)
        augmented = self.augmented_transform(img)
        return {"original": original, "augmented": augmented}


class DecoderDataset(Dataset):
    def __init__(self, csv_file, img_dir, token_dict, img_size=(64, 384), max_length=20, transform=None):
        self.img_dir = img_dir
        # Assumes no header in CSV. If header exists, remove header=None
        self.annotations = pd.read_csv(csv_file, header=None)
        self.token_dict = token_dict
        self.max_length = max_length
        
        # Base transform pipeline
        self.transform = transforms.Compose([
            Lambda(lambda img: img.convert("RGB")),
            RandomRotation(degrees=3.5, fill=255),
            RandomApply([GaussianBlur(kernel_size=3)], p=0.3),
            ResizeAndPad(img_size, fill=255), # Using the aspect-ratio preserving resize
            Grayscale(num_output_channels=3),
            ToTensor(),
            *([transform] if transform else [])
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Retry logic for this dataset as well
        try:
            img_name = self.annotations.iloc[index, 1]
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path)
            image.load()
        except Exception as e:
            print(f"Warning: Error loading {img_name}. Error: {e}")
            new_index = random.randint(0, len(self.annotations) - 1)
            return self.__getitem__(new_index)

        image = self.transform(image)

        # Label processing
        label = str(self.annotations.iloc[index, 0])
        label_tokenized = [
            self.token_dict[char.lower()] if char.lower() in self.token_dict
            else self.token_dict.get("<UNK>", 0) for char in label
        ]
        
        # 1. Truncate if longer than max_length (leaving room for EOS)
        label_tokenized = label_tokenized[:self.max_length]
        
        # 2. Append EOS
        if '<EOS>' in self.token_dict:
            label_tokenized.append(self.token_dict['<EOS>'])
        
        # 3. Pad to fixed length (max_length + 1 for EOS)
        target_len = self.max_length + 1
        pad_token = self.token_dict.get('<PAD>', 0)
        
        while len(label_tokenized) < target_len:
            label_tokenized.append(pad_token)

        return image, torch.tensor(label_tokenized, dtype=torch.long)
