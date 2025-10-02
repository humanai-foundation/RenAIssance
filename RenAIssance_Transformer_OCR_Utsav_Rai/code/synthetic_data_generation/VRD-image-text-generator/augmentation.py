from torchvision import transforms
import torch
import numpy as np

class Transformations:
    class RandomNoise(object):
        def __init__(self, min_noise_level=0.1, max_noise_level=0.3):
            self.min_noise_level = min_noise_level
            self.max_noise_level = max_noise_level

        def __call__(self, tensor):
            noise = torch.randn_like(tensor[0]) * np.random.uniform(self.min_noise_level, self.max_noise_level)
            return torch.clamp(tensor + noise.unsqueeze(0), 0, 1)

    class ElasticGrid(object):
        def __init__(self, sigma=5.0):
            self.sigma = sigma

        def __call__(self, tensor):
            return transforms.ElasticTransform(
                alpha=max(2, 9-(20/1000)*tensor.shape[0]), 
                sigma=self.sigma, 
                interpolation=transforms.InterpolationMode.BILINEAR, 
                fill=1)(tensor)

    class Resize(object):
        def __init__(self, horizontal_ratio=(0.3,1.5), vertical_ratio=(0.9,1.1)):
            self.horizontal_ratio = horizontal_ratio
            self.vertical_ratio = vertical_ratio

        def __call__(self, tensor):
            _, height, width = tensor.shape
            return transforms.Resize(
                (int(height * np.random.uniform(*self.vertical_ratio)), int(width * np.random.uniform(*self.horizontal_ratio))), 
                interpolation=transforms.InterpolationMode.BILINEAR
            )(tensor)
            
    
data_transformer = transforms.Compose([
    transforms.ToTensor(),   
    transforms.Grayscale(), 
    transforms.RandomApply([transforms.RandomRotation(degrees=1, fill=1)], p=0.5),
    transforms.RandomApply([Transformations.RandomNoise(min_noise_level=0.1, max_noise_level=0.2)], p=0.8),
    transforms.RandomApply([transforms.functional.invert],p=0.01),
    transforms.RandomApply([Transformations.ElasticGrid(sigma=5.0)], p=0.8),
    transforms.RandomApply([Transformations.Resize()], p=0.5),
    transforms.ToPILImage()
])