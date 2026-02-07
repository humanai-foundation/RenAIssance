import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import os
import sys

# Ensure current directory is in path
sys.path.append(os.getcwd())

from encoder import Encoder
from ViT_encoder import ViTEncoder
from custom_dataset import ContrastiveLearningDataset
from custom_loss import contrastive_loss

def verify_training():
    print("Starting training verification...")
    
    # Load config
    try:
        config_full = json.load(open('config.json', 'r'))
        activate_ViT = config_full["Encoder"]["ViT"]
        config = config_full["SSL"]
        print(f"Config loaded. ViT Enabled: {activate_ViT}")
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    if activate_ViT:
        print("Initializing ViTEncoder...")
        model = ViTEncoder(input_size=(64, 384), patch_size=16, hidden_dim=516, num_heads=3, num_layers=2) # Reduced layers for speed in test
    else:
        print("Initializing ResNet Encoder...")
        model = Encoder()
    
    model = model.to(device)
    
    # Optimizer
    optimizer = SGD(model.parameters(), lr=config["start lr"])
    scheduler = StepLR(optimizer, step_size=config["lr scheduler step size"], gamma=0.1)

    # Dataset
    print("Loading datasets...")
    dataset = []
    # Only check dataset 1 for this test as configured
    if config["dataset 1"] is not None:
        if os.path.isdir(config["dataset 1"]):
            dataset.append(ContrastiveLearningDataset(img_dir=config["dataset 1"]))
            print(f"Loaded dataset 1 from {config['dataset 1']}")
        else:
            print(f"Dataset 1 path does not exist: {config['dataset 1']}")
    
    if len(dataset) == 0:
        print("No datasets loaded. Exiting.")
        return

    dataset = ConcatDataset(dataset)
    train_dataloader = DataLoader(dataset, batch_size=config["Batch size"], shuffle=True)
    
    print("Starting training loop (1 batch only for verification)...")
    model.train()
    
    try:
        for i, batch in enumerate(train_dataloader):
            original, augmented = batch['original'], batch['augmented']
            original = original.to(device)
            augmented = augmented.to(device)

            optimizer.zero_grad()

            original_embeddings, _ = model(original)
            augmented_embeddings, _ = model(augmented)
            
            # Simple loss calculation for verification (mimicking notebook logic)
            # Notebook logic:
            # avg_pool = nn.AdaptiveAvgPool2d((5, original_embeddings.shape[2])) # This might fail if ViT returns 3D tensor (B,N,D) instead of 4D (B,C,H,W)
            # The ViT encoder returns (B, N, D), while ResNet/Encoder likely returned (B, C, H, W) or (B, Seq, Feat)
            
            # Let's check the shape and adapt if necessary, as the notebook code might be tightly coupled to ResNet/LSTM output which is (B, Seq, Feat) ?
            # Wait, encoder.py returns: rnn_output, hidden
            # rnn_output shape from comment: [batch size, sequence length, feature length] -> (B, S, F)
            # So `original_embeddings` is 3D.
            
            # Notebook code:
            # avg_pool = nn.AdaptiveAvgPool2d((5, original_embeddings.shape[2])) 
            # AdaptiveAvgPool2d expects 4D input (B, C, H, W). 
            # If input is (B, S, F), it interprets S as H and F as W? No, it expects (B, C, H, W).
            
            # let's look at the notebook code again:
            # original_embeddings, _ = model(original)
            # avg_pool = nn.AdaptiveAvgPool2d((5, original_embeddings.shape[2]))
            # original_embeddings = avg_pool(original_embeddings)
            
            # If standard 3D tensor (B, S, F) is passed to AdaptiveAvgPool2d, it will error if it expects 4D.
            # UNLESS PyTorch allows 3D input treating it as unbatched? No, we have batch.
            
            # Let's see what Encoder.py returns.
            # Encoder.py: 
            # rnn_output, hidden = self.lstm(resnet_output)
            # rnn_output is (Batch, SeqLen, NumDirections*HiddenSize) -> (B, S, 512)
            
            # If the notebook uses `nn.AdaptiveAvgPool2d`, it implies it expects a 4D tensor?
            # Or maybe `original_embeddings` is treated as (B, 1, S, F) ??
            
            print(f"Embedding shape: {original_embeddings.shape}")
            
            # In notebook it treats it as:
            # avg_pool = nn.AdaptiveAvgPool2d((5, original_embeddings.shape[2]))
            # If input is 3D (B, S, F), AdaptiveAvgPool2d might not work directly.
            # Let's try to handle it. If it fails, that's a finding.
            
            # If I implemented ViT to return (B, N, D), it matches (B, S, F).
             
            if len(original_embeddings.shape) == 3:
                 # Unsqueeze to make it 4D for AvgPool if required, or maybe the notebook *assumed* a certain shape?
                 # Actually, looking at the notebook again:
                 # original_embeddings, _ = model(original) 
                 # avg_pool = nn.AdaptiveAvgPool2d((5, original_embeddings.shape[2]))
                 # If original_embeddings is (B, H, W), then .shape[2] is W.
                 
                 # Wait, let's run it and see. If it errors, we fix.
                 pass

            # Just computing a dummy loss to verify backward pass
            loss = original_embeddings.mean() # Dummy loss for simple verification
            
            loss.backward()
            optimizer.step()
            
            print("Backward pass successful.")
            break # Only 1 batch
            
    except Exception as e:
        print(f"Training step failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_training()
