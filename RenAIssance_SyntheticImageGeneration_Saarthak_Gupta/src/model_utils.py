from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
import random
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import io
from pathlib import Path
import os
from typing import List, Optional, Tuple
import matplotlib


#########################################################################################################################
# Dataset class for loading word-image pairs
#########################################################################################################################
class WordImageDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = Image.open(self.data.iloc[idx]['source_path']).convert('RGB')
        target = Image.open(self.data.iloc[idx]['target_path']).convert('RGB')
        return self.transform(source), self.transform(target)
#########################################################################################################################
# Model
#########################################################################################################################
# Generator
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super().__init__()

        def down(in_f, out_f, norm=True):
            layers = [nn.Conv2d(in_f, out_f, 4, 2, 1, bias=False)]
            if norm:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        def up(in_f, out_f, dropout=False):
            layers = [nn.ConvTranspose2d(in_f, out_f, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(out_f), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.down1 = down(in_channels, features, norm=False)
        self.down2 = down(features, features*2)
        self.down3 = down(features*2, features*4)
        self.down4 = down(features*4, features*8)
        self.down5 = down(features*8, features*8)
        self.down6 = down(features*8, features*8)
        self.down7 = down(features*8, features*8)
        self.down8 = down(features*8, features*8, norm=False)

        self.up1 = up(features*8, features*8, dropout=True)
        self.up2 = up(features*16, features*8, dropout=True)
        self.up3 = up(features*16, features*8, dropout=True)
        self.up4 = up(features*16, features*8)
        self.up5 = up(features*16, features*4)
        self.up6 = up(features*8, features*2)
        self.up7 = up(features*4, features)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(d1)
        d3 = self.down3(d2); d4 = self.down4(d3)
        d5 = self.down5(d4); d6 = self.down6(d5)
        d7 = self.down7(d6); d8 = self.down8(d7)

        u1 = self.up1(d8); u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        return self.final(torch.cat([u7, d1], 1))

# Discriminator
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=2, features=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features*2, 4, 2, 1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features*2, features*4, 4, 2, 1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features*4, features*8, 4, 1, 1),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features*8, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))
#########################################################################################################################
# Training function for Pix2Pix model
#########################################################################################################################

def train_pix2pix(csv_file, epochs=100, batch_size=8, lr=2e-4, save_dir="models", val_interval=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ---- Dataset Splitting ----
    full_dataset = WordImageDataset(csv_file)
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = int(0.05 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # ---- Models ----
    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)

    # ---- Optimizers and Loss Functions ----
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()

    # ---- History ----
    history = {
        "epoch": [],
        "loss_G": [],
        "loss_D": [],
        "val_loss_L1": [],
        "val_loss_GAN": []
    }

    for epoch in range(epochs):
        G.train()
        D.train()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        running_loss_G = 0.0
        running_loss_D = 0.0

        for x, y in loop:
            x, y = x.to(device), y.to(device)
            valid = torch.ones((x.size(0), 1, 30, 30), device=device)
            fake = torch.zeros((x.size(0), 1, 30, 30), device=device)

            # ---- Train Generator ----
            opt_G.zero_grad()
            y_fake = G(x)
            pred_fake = D(y_fake, x)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_L1 = criterion_L1(y_fake, y) * 100
            loss_G = loss_GAN + loss_L1
            loss_G.backward()
            opt_G.step()

            # ---- Train Discriminator ----
            opt_D.zero_grad()
            pred_real = D(y, x)
            loss_real = criterion_GAN(pred_real, valid)
            pred_fake = D(y_fake.detach(), x)
            loss_fake = criterion_GAN(pred_fake, fake)
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()

            loop.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())

        avg_loss_G = running_loss_G / len(train_loader)
        avg_loss_D = running_loss_D / len(train_loader)

        history["epoch"].append(epoch + 1)
        history["loss_G"].append(avg_loss_G)
        history["loss_D"].append(avg_loss_D)

        # ---- Validation ----
        if (epoch + 1) % val_interval == 0 or epoch == 0:
            G.eval()
            total_L1 = 0.0
            total_GAN = 0.0

            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    y_pred = G(x_val)
                    pred = D(y_pred, x_val)

                    loss_l1 = criterion_L1(y_pred, y_val)
                    loss_gan = criterion_GAN(pred, torch.ones_like(pred))

                    total_L1 += loss_l1.item()
                    total_GAN += loss_gan.item()

            val_L1_avg = total_L1 / len(val_loader)
            val_GAN_avg = total_GAN / len(val_loader)

            history["val_loss_L1"].append(val_L1_avg)
            history["val_loss_GAN"].append(val_GAN_avg)

            print(f"\n[Validation @ Epoch {epoch+1}] L1 Loss: {val_L1_avg:.4f}, GAN Loss: {val_GAN_avg:.4f}\n")
        else:
            history["val_loss_L1"].append(np.nan)
            history["val_loss_GAN"].append(np.nan)

    # ---- Save Final Models ----
    os.makedirs(save_dir, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(save_dir, "generator_final.pth"))
    torch.save(D.state_dict(), os.path.join(save_dir, "discriminator_final.pth"))

    # ---- Save Training History ----
    pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_history.csv"), index=False)
    print("Training complete. Models and history saved.")

    return G, D, train_loader, val_loader, test_loader


#########################################################################################################################
def plot_gan_history(csv_path):
    # Load the history CSV
    df = pd.read_csv(csv_path)

    # Plot generator and discriminator losses
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['loss_G'], label='Generator Loss', linewidth=2)
    plt.plot(df['epoch'], df['loss_D'], label='Discriminator Loss', linewidth=2)

    # Plot validation losses if present
    if 'val_loss_L1' in df.columns:
        plt.plot(df['epoch'], df['val_loss_L1'], label='Val L1 Loss', linestyle='--', color='green')
    if 'val_loss_GAN' in df.columns:
        plt.plot(df['epoch'], df['val_loss_GAN'], label='Val GAN Loss', linestyle='--', color='orange')

    # Labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    save_path = csv_path.replace('.csv', '_history.png')
    plt.savefig(save_path)


def visualize_pix2pix_results(generator_path, dataloader, num_samples=5, figsize=(12, 4)):
    """
    Visualize Pix2Pix outputs given a test dataloader.

    Args:
        generator_path (str): Path to the trained generator model.
        dataloader (DataLoader): PyTorch DataLoader for test data.
        num_samples (int): Number of samples to visualize.
        figsize (tuple): Width x Height of each row in the plot.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load trained generator
    generator = UNetGenerator(in_channels=1, out_channels=1)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval().to(device)

    # Image un-normalization for display
    to_pil = T.Compose([
        T.Normalize((0.0,), (1 / 0.5,)),  # std
        T.Normalize((-0.5,), (1.0,)),     # mean
        T.ToPILImage()
    ])

    # Gather samples from dataloader
    samples = []
    for batch in dataloader:
        inputs, targets = batch
        for i in range(inputs.size(0)):
            samples.append((inputs[i], targets[i]))
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break

    # Plotting
    fig, axs = plt.subplots(len(samples), 3, figsize=(figsize[0], figsize[1] * len(samples)))
    if len(samples) == 1:
        axs = [axs]  # For consistent 2D indexing

    fig.suptitle('Pix2Pix GAN Results: Font → Handwriting Translation', fontsize=16, fontweight='bold', y=0.98)

    for row_idx, (input_img, target_img) in enumerate(samples):
        input_tensor = input_img.unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = generator(input_tensor)

        input_pil = to_pil(input_img.cpu())
        output_pil = to_pil(output_tensor.squeeze(0).cpu())
        target_pil = to_pil(target_img.cpu())

        images = [input_pil, output_pil, target_pil]
        for col_idx, image in enumerate(images):
            axs[row_idx][col_idx].imshow(image, cmap='gray')
            axs[row_idx][col_idx].axis('off')

            # Column titles only on the first row
            if row_idx == 0:
                if col_idx == 0:
                    axs[row_idx][col_idx].set_title("Input Font", fontsize=12, fontweight='bold')
                elif col_idx == 1:
                    axs[row_idx][col_idx].set_title("Generated Handwriting", fontsize=12, fontweight='bold', color='blue')
                elif col_idx == 2:
                    axs[row_idx][col_idx].set_title("Target Handwriting", fontsize=12, fontweight='bold', color='green')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.2)
    save_path = generator_path.replace('generator_final.pth', 'test_visualizations.png')
    plt.savefig(save_path)

#############################################################################################################################
# Inference Class
#############################################################################################################################
class GANInferencePipeline:
    def __init__(self, 
                 generator_path: str,
                 custom_font_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 single_image_size: Tuple[int, int] = (128, 64),
                 grid_size: Tuple[int, int] = (4, 2),
                 base_fontsize: int = 45):
        """
        Initialize the GAN inference pipeline.
        
        Args:
            generator_path: Path to the trained generator model
            custom_font_path: Path to custom font file (optional)
            device: Device to run inference on
            single_image_size: Size of individual word images (width, height)
            grid_size: Grid arrangement (rows, cols)
            base_fontsize: Base font size for rendering
        """
        self.device = device
        self.single_image_size = single_image_size
        self.grid_size = grid_size
        self.base_fontsize = base_fontsize
        self.custom_font_path = custom_font_path
        
        # Load the generator model
        self.generator = self.load_generator(generator_path)
        
        # Setup matplotlib configuration
        self.setup_matplotlib_config()
        
        # Load custom font if provided
        self.custom_font_prop = self.load_custom_font()
        
        print(f"GAN Inference Pipeline initialized on {device}")
        print(f"Single image size: {single_image_size}")
        print(f"Grid size: {grid_size[0]}x{grid_size[1]} = {grid_size[0]*grid_size[1]} images")
        
    def load_generator(self, generator_path: str):
        """Load the trained generator model."""
        # You'll need to replace this with your actual UNetGenerator class
        # For now, I'll create a placeholder
        generator = UNetGenerator()  # Replace with your actual generator class
        generator.load_state_dict(torch.load(generator_path, map_location=self.device))
        generator.to(self.device)
        generator.eval()
        return generator
    
    def setup_matplotlib_config(self):
        """Setup matplotlib configuration for text rendering."""
        matplotlib_config = {
            'text.usetex': False,
            'mathtext.fontset': 'cm',
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.weight': 'bold',
            'mathtext.bf': 'bold',
        }
        matplotlib.rcParams.update(matplotlib_config)
    
    def load_custom_font(self):
        """Load custom font if provided."""
        if self.custom_font_path and os.path.exists(self.custom_font_path):
            try:
                custom_font_prop = fm.FontProperties(fname=self.custom_font_path)
                print(f"Successfully loaded custom font: {self.custom_font_path}")
                return custom_font_prop
            except Exception as e:
                print(f"Warning: Could not load custom font {self.custom_font_path}: {str(e)}")
                print("Falling back to default font configuration")
        return None
    
    def render_single_word(self, word: str) -> Image.Image:
        """
        Render a single word as an image.
        
        Args:
            word: The word to render
            
        Returns:
            PIL Image of the rendered word (grayscale)
        """
        width, height = self.single_image_size
        
        # Create figure with exact pixel dimensions
        dpi = 150
        figsize = (width/dpi, height/dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white')
        
        # Create axis that fills the figure completely
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Adjust fontsize based on word length
        length_adjustment = max(1, min(15, len(word))) / 7.0
        fontsize = int(self.base_fontsize / length_adjustment)
        
        # Prepare text formatting arguments
        text_kwargs = {
            'fontsize': fontsize,
            'ha': 'center',
            'va': 'center',
            'transform': ax.transAxes,
            'color': 'black'  # Ensure black text for good contrast
        }
        
        # Use custom font if available
        if self.custom_font_prop:
            text_kwargs['fontproperties'] = self.custom_font_prop
            display_text = word
        else:
            display_text = f"$\\mathbf{{{word}}}$"
            text_kwargs['fontweight'] = 'bold'
        
        # Render text at exact center
        ax.text(0.5, 0.5, display_text, **text_kwargs)
        
        # Remove all axes and ensure clean background
        ax.axis('off')
        ax.set_facecolor('white')
        
        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', 
                    pad_inches=0, facecolor='white', edgecolor='none',
                    dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        
        # Load and process the image
        rendered_img = Image.open(buf)
        rendered_img = rendered_img.convert('L')  # Convert to grayscale
        
        # Create final canvas with exact target dimensions (grayscale)
        final_img = Image.new('L', (width, height), 255)  # White background
        
        # Calculate scaling to fit the rendered text optimally
        rendered_width, rendered_height = rendered_img.size
        
        # Scale to use 90% of the canvas for better visual appearance
        scale_w = (width * 0.9) / rendered_width
        scale_h = (height * 0.9) / rendered_height
        scale = min(scale_w, scale_h)
        
        new_width = int(rendered_width * scale)
        new_height = int(rendered_height * scale)
        
        # Resize with high quality resampling
        resample_method = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS
        resized_img = rendered_img.resize((new_width, new_height), resample_method)
        
        # Calculate exact center position
        x_offset = (width - new_width) // 2
        y_offset = (height - new_height) // 2
        
        # Paste the resized image at the center
        final_img.paste(resized_img, (x_offset, y_offset))
        
        return final_img
    
    def create_grid_from_words(self, words: List[str]) -> Image.Image:
        """
        Create a grid image from a list of words.
        
        Args:
            words: List of words to render and arrange in grid
            
        Returns:
            PIL Image of the grid (grayscale)
        """
        rows, cols = self.grid_size
        expected_words = rows * cols
        
        # Pad or truncate words list to match grid size
        if len(words) < expected_words:
            # Pad with empty strings
            words = words + [''] * (expected_words - len(words))
        elif len(words) > expected_words:
            # Truncate to fit grid
            words = words[:expected_words]
            print(f"Warning: Truncated word list to {expected_words} words to fit {rows}x{cols} grid")
        
        # Render individual word images
        word_images = []
        for word in words:
            if word.strip():  # Only render non-empty words
                img = self.render_single_word(word)
            else:
                # Create empty image for empty words (grayscale)
                img = Image.new('L', self.single_image_size, 255)  # White background
            word_images.append(img)
        
        # Create grid
        single_width, single_height = self.single_image_size
        grid_width = cols * single_width
        grid_height = rows * single_height
        
        grid_image = Image.new('L', (grid_width, grid_height), 255)  # Grayscale grid
        
        # Paste images into grid
        for i, img in enumerate(word_images):
            row = i // cols
            col = i % cols
            
            x = col * single_width
            y = row * single_height
            
            grid_image.paste(img, (x, y))
        
        return grid_image
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for GAN input.
        
        Args:
            image: PIL Image to preprocess (grayscale)
            
        Returns:
            Preprocessed tensor ready for GAN (1-channel)
        """
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_array).float()
        
        # Add channel dimension (HW -> CHW) for grayscale
        image_tensor = image_tensor.unsqueeze(0)
        
        # Normalize to [-1, 1] range (assuming input images are in [0, 255])
        image_tensor = (image_tensor / 127.5) - 1.0
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def postprocess_output(self, output_tensor: torch.Tensor) -> Image.Image:
        """
        Convert GAN output tensor back to PIL Image.
        
        Args:
            output_tensor: Output tensor from GAN (1-channel or 3-channel)
            
        Returns:
            PIL Image
        """
        # Remove batch dimension and move to CPU
        output_tensor = output_tensor.squeeze(0).cpu()
        
        # Denormalize from [-1, 1] to [0, 255]
        output_tensor = (output_tensor + 1.0) * 127.5
        output_tensor = torch.clamp(output_tensor, 0, 255)
        
        # Convert to numpy
        output_array = output_tensor.numpy().astype(np.uint8)
        
        # Handle different channel configurations
        if output_tensor.shape[0] == 1:  # Single channel (grayscale)
            output_array = output_array.squeeze(0)  # Remove channel dimension
            return Image.fromarray(output_array, mode='L')
        elif output_tensor.shape[0] == 3:  # Three channels (RGB)
            output_array = output_array.transpose(1, 2, 0)  # CHW -> HWC
            return Image.fromarray(output_array, mode='RGB')
        else:
            raise ValueError(f"Unsupported number of channels: {output_tensor.shape[0]}")
    
    def generate_handwriting(self, words: List[str]) -> Tuple[Image.Image, Image.Image]:
        """
        Generate handwriting from a list of words.
        
        Args:
            words: List of words to convert to handwriting
            
        Returns:
            Tuple of (input_grid, output_handwriting) PIL Images
        """
        # Create grid from words (grayscale)
        input_grid = self.create_grid_from_words(words)
        
        # Preprocess for GAN
        input_tensor = self.preprocess_image(input_grid)
        
        # Generate handwriting using GAN
        with torch.no_grad():
            output_tensor = self.generator(input_tensor)
        
        # Postprocess output
        output_handwriting = self.postprocess_output(output_tensor)
        
        return input_grid, output_handwriting
    
    def save_results(self, 
                    input_grid: Image.Image, 
                    output_handwriting: Image.Image,
                    output_dir: str = "inference_results",
                    filename_prefix: str = "result"):
        """
        Save the input and output images.
        
        Args:
            input_grid: Input grid image
            output_handwriting: Generated handwriting image
            output_dir: Directory to save results
            filename_prefix: Prefix for saved filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        input_path = os.path.join(output_dir, f"{filename_prefix}_input.png")
        output_path = os.path.join(output_dir, f"{filename_prefix}_output.png")
        
        input_grid.save(input_path)
        output_handwriting.save(output_path)
        
        print(f"Results saved:")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        
        return input_path, output_path
