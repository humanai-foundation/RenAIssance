import torch
from torch import nn
try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:
    # define a dummy mixin if helpful, or just omit if not installed, 
    # but since encoder.py has it, we assume it's there.
    class PyTorchModelHubMixin:
        pass

class ViTEncoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, input_size=(64, 384), patch_size=16, hidden_dim=516, num_heads=3, num_layers=12, dropout=0.1):
        super(ViTEncoder, self).__init__()
        self.H, self.W = input_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        self.num_patches_h = self.H // patch_size
        self.num_patches_w = self.W // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Patch embedding: (B, 3, H, W) -> (B, hidden_dim, H/p, W/p)
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, 
                                                   dim_feedforward=hidden_dim*4, 
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.patch_embed(x) # (B, dim, h, w)
        x = x.flatten(2)        # (B, dim, N)
        x = x.transpose(1, 2)   # (B, N, dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Return tuple to match Encoder signature (output, hidden)
        # ViT doesn't have an LSTM hidden state, so we return None
        return x, None
