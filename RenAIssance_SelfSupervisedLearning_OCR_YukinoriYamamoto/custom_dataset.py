import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import RandomApply, GaussianBlur, Resize, Compose, ToTensor, Lambda, RandomPerspective, \
    RandomAffine, RandomRotation, Grayscale
from PIL import Image
import os
import numpy as np
import random
from torch.utils.data import Dataset


class RandomVerticalCrop:
    def __init__(self, crop_height_ratio):
        self.crop_height_ratio = crop_height_ratio

    def __call__(self, img):
        (width, height) = img.size
        crop_height = int(self.crop_height_ratio * height)
        start = random.randint(0, crop_height // 2)
        end = height - random.randint(0, crop_height // 2)
        return img.crop((0, start, width, end))


def resize_and_pad(img, target_size):
    target_height, target_width = target_size
    img_width, img_height = img.size

    # Calculate the original aspect ratio
    original_aspect = img_width / img_height
    target_aspect = target_width / target_height

    # Compare the original aspect ratio with the target one
    if original_aspect > target_aspect:
        # If the original aspect ratio is greater, resize the width to the target and scale the height accordingly
        new_width = target_width
        new_height = round(new_width / original_aspect)
    else:
        # If the original aspect ratio is smaller, resize the height to the target and scale the width accordingly
        new_height = target_height
        new_width = round(new_height * original_aspect)

    # Resize the image
    resized_img = F.resize(img, (new_height, new_width))

    # Calculate the padding
    pad_vert = target_height - new_height
    pad_top, pad_bot = pad_vert // 2, pad_vert - (pad_vert // 2)
    pad_horz = target_width - new_width
    pad_left, pad_right = pad_horz // 2, pad_horz - (pad_horz // 2)

    # Apply the padding
    resized_img = F.pad(resized_img, (pad_left, pad_top, pad_right, pad_bot), fill=255)

    return resized_img


class ContrastiveLearningDataset(Dataset):
    def __init__(self, img_dir, crop_height_ratio=0.2, img_size=(64, 384)):
        super().__init__()
        self.img_size = img_size
        assert os.path.isdir(img_dir)
        self.filepaths = [
            os.path.join(img_dir, filename) for filename in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, filename))
        ]
        if not self.filepaths:
            raise ValueError(f"No image files found in {img_dir}")
        self.original_transform = Compose([
            Lambda(lambda img: img.convert("RGB")),
            Grayscale(num_output_channels=3),
            Resize(img_size),
            ToTensor(),
        ])
        self.augmented_transform = Compose([
            Lambda(lambda img: img.convert("RGB")),
            RandomApply([RandomVerticalCrop(crop_height_ratio=crop_height_ratio)], p=0.5),
            RandomApply([GaussianBlur(kernel_size=3)], p=0.3),
			RandomApply([RandomAffine(degrees=3.5, translate=(0.05, 0.05), shear=15)], p=0.4),
			Grayscale(num_output_channels=3),
            Resize(img_size),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.filepaths[idx])
        except IOError:
            return "cannot identify image file '%s'", self.filepaths[idx]
        original = self.original_transform(img)
        augmented = self.augmented_transform(img)
        return {"original": original, "augmented": augmented}


class DecoderDataset(Dataset):
    def __init__(self, csv_file, img_dir, token_dict, img_size=(64, 384), max_length=20, transform=None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(csv_file)
        self.token_dict = token_dict
        self.max_length = max_length
        self.transform = transforms.Compose([
            Lambda(lambda img: img.convert("RGB")),
            # Lambda(lambda img: resize_and_pad(img=img, target_size=img_size)),
            RandomApply([GaussianBlur(kernel_size=3)], p=0.3),
			# RandomApply([RandomAffine(degrees=3.5, translate=(0.05, 0.05), shear=15)], p=0.4),
			RandomRotation(degrees=3.5, fill=255),
			Grayscale(num_output_channels=3),
            Resize(img_size),
            ToTensor(),  # Convert image to PyTorch Tensor in CHW format
            *([transform] if transform else [])
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 1]
        image = Image.open(os.path.join(self.img_dir, img_name))  # Use PIL to read the image
        image = self.transform(image)  # Image is in CHW format now

        label = self.annotations.iloc[index, 0]
        label_tokenized = [
            self.token_dict[char.lower()] if char.lower() in self.token_dict
            else self.token_dict["<UNK>"] for char in label]
        label_tokenized = label_tokenized[:self.max_length]
        label_tokenized.append(self.token_dict['<EOS>'])
        label_length = len(label_tokenized)
        for i in range(label_length, self.max_length+1):
            label_tokenized.append(self.token_dict['<PAD>'])
        return image, torch.tensor(label_tokenized, dtype=torch.long)
