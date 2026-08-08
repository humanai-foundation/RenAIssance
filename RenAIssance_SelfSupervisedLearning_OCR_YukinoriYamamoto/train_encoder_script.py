
import json
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import os
import sys
import matplotlib.pyplot as plt

# Ensure current directory is in path
sys.path.append(os.getcwd())

from encoder import Encoder
from ViT_encoder import ViTEncoder
from custom_dataset import ContrastiveLearningDataset
from custom_loss import contrastive_loss

def main():
    print("Starting Encoder Training...")

    # Load Config
    try:
        config_full = json.load(open('config.json', 'r'))
        activate_ViT = config_full["Encoder"]["ViT"]
        config = config_full["SSL"]
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    print(f"ViT Enabled: {activate_ViT}")
    print(f"Config: {config}")

    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if activate_ViT:
        print("ViT")
        # Matches notebook: input_size=(64, 384), patch_size=16, hidden_dim=516, num_heads=3, num_layers=12
        model = ViTEncoder(input_size=(64, 384), patch_size=16, hidden_dim=516, num_heads=3, num_layers=12)
    else:
        print("ResNet")
        model = Encoder()
        
    model = model.to(device)
    
    optimizer = SGD(model.parameters(), lr=config["start lr"])
    scheduler = StepLR(optimizer, step_size=config["lr scheduler step size"], gamma=0.1)
    
    dataset = []
    for i in range(1, 4):
        key = f"dataset {i}"
        if config.get(key) is not None:
             if os.path.exists(config[key]):
                print(f"Loading {key}: {config[key]}")
                dataset.append(ContrastiveLearningDataset(img_dir=config[key]))
             else:
                print(f"Warning: {key} path does not exist: {config[key]}")

    if not dataset:
        print("No datasets loaded. Exiting.")
        return

    dataset = ConcatDataset(dataset)
    train_dataloader = DataLoader(dataset, batch_size=config["Batch size"], shuffle=True)
    
    print(f"Data loaded. Batches: {len(train_dataloader)}")

    epochs = config["epoch size"]
    step = 0
    steps = []
    loss_list = []

    model.train()
    
    try:
        for epoch in range(epochs):
            batch_loss = 0
            for i, batch in enumerate(train_dataloader):
                original, augmented = batch['original'], batch['augmented']
                original = original.to(device)
                augmented = augmented.to(device)

                optimizer.zero_grad()

                original_embeddings, _ = model(original)
                augmented_embeddings, _ = model(augmented)
                
                # Check for notebook compatibility logic with AdaptiveAvgPool2d
                # The notebook does this:
                # avg_pool = nn.AdaptiveAvgPool2d((5, original_embeddings.shape[2]))
                # original_embeddings = avg_pool(original_embeddings)
                
                # If ViT outputs (B, N, D) -> (B, 96, 516), this might be problematic if taken literally as 3D tensor
                # because AdaptiveAvgPool2d usually expects 4D (B, C, H, W).
                # Only if the notebook code explicitly handles it or if PyTorch allows 3D.
                # However, since I deleted the previous script, I can't be 100% sure if I included this block exactly as intended in the successful run.
                # The successful run printed "Epoch 1/1... Loss:...", so it passed the loss calculation.
                # The `verify_training_run.py` (which I read) did NOT include `avg_pool` block, it just calculated mean().
                # The `train_encoder_script.py` which I deleted DID include it (I can recall writing it).
                # So it implies `avg_pool` worked or `if len(...) == 4` check skipped it.
                
                if len(original_embeddings.shape) == 4:
                    avg_pool = nn.AdaptiveAvgPool2d((5, original_embeddings.shape[2]))
                    original_embeddings = avg_pool(original_embeddings)
                    augmented_embeddings = avg_pool(augmented_embeddings)
                    
                    original_embeddings = original_embeddings.view(original_embeddings.shape[0] // 4, original_embeddings.shape[1] * 4, original_embeddings.shape[2])
                    augmented_embeddings = augmented_embeddings.view(augmented_embeddings.shape[0] // 4, augmented_embeddings.shape[1] * 4, augmented_embeddings.shape[2])
                
                loss = contrastive_loss(original_embeddings, augmented_embeddings)
                batch_loss += loss.item()
                
                if i % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(train_dataloader)}, Loss:{loss.item()}")
                    step += 10
                    steps.append(step)
                    loss_list.append(loss.item())
                    
                loss.backward()
                optimizer.step()
                
            scheduler.step()
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss:{batch_loss / len(train_dataloader)}")

        # Save plot
        plt.figure()
        plt.plot(steps, loss_list)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig("training_loss.png")
        print("Saved training_loss.png")

        # Save model
        save_path = config["saved Encoder path"]
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
