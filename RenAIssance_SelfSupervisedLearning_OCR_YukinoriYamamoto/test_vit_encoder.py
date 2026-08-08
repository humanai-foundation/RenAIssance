import torch
import sys
import os

# Ensure we can import from current directory
sys.path.append(os.getcwd())

from ViT_encoder import ViTEncoder

def test():
    print("Testing ViTEncoder...")
    try:
        model = ViTEncoder(input_size=(64, 384), patch_size=16, hidden_dim=516, num_heads=3, num_layers=2) # Reduced layers for speed
        x = torch.randn(2, 3, 64, 384)
        output, hidden = model(x)
        
        print("Output shape:", output.shape)
        
        # Expected shape calculation
        # H=64, W=384, p=16
        # h_patches = 4, w_patches = 24 -> N = 96
        # hidden_dim = 516
        expected_shape = (2, 96, 516)
        
        if output.shape == expected_shape:
            print("Verification successful! Output shape matches expected: (B, N, D)")
        else:
            print(f"Verification failed! Expected {expected_shape}, got {output.shape}")
            sys.exit(1)
            
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test()
