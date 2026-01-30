"""
Vision Transformer (ViT) Encoder for OCR Tasks
===============================================
This module implements a Vision Transformer encoder specifically designed
for Optical Character Recognition (OCR) tasks on historical Renaissance texts.

The encoder processes images by splitting them into patches and using
transformer architecture to extract visual features.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class VITConfig:
    """Configuration class for Vision Transformer Encoder"""
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-12
    qkv_bias: bool = True
    use_faster_attention: bool = True


class PatchEmbeddings(nn.Module):
    """
    Convert images into patches and embed them.
    
    Args:
        config: VITConfig object containing model configuration
    """
    
    def __init__(self, config: VITConfig):
        super().__init__()
        
        image_size = config.image_size
        patch_size = config.patch_size
        num_channels = config.num_channels
        hidden_size = config.hidden_size
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Convolutional layer to create patch embeddings
        self.projection = nn.Conv2d(
            num_channels, 
            hidden_size, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: Input images of shape (batch_size, num_channels, height, width)
            
        Returns:
            Patch embeddings of shape (batch_size, num_patches, hidden_size)
        """
        batch_size, num_channels, height, width = pixel_values.shape
        
        # Ensure image dimensions are correct
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Input image size ({height}x{width}) doesn't match "
                f"model image size ({self.image_size}x{self.image_size})"
            )
        
        # Project patches: (batch, hidden_size, num_patches_h, num_patches_w)
        embeddings = self.projection(pixel_values)
        
        # Flatten patches: (batch, hidden_size, num_patches)
        embeddings = embeddings.flatten(2)
        
        # Transpose: (batch, num_patches, hidden_size)
        embeddings = embeddings.transpose(1, 2)
        
        return embeddings


class VITEmbeddings(nn.Module):
    """
    Construct position and patch embeddings for Vision Transformer.
    Optionally includes a [CLS] token.
    """
    
    def __init__(self, config: VITConfig, use_cls_token: bool = True):
        super().__init__()
        
        self.use_cls_token = use_cls_token
        self.patch_embeddings = PatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        
        # CLS token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            num_positions = self.num_patches + 1
        else:
            num_positions = self.num_patches
            
        # Position embeddings
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_positions, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: Input images
            
        Returns:
            Embeddings with position information
        """
        batch_size = pixel_values.shape[0]
        
        # Get patch embeddings
        embeddings = self.patch_embeddings(pixel_values)
        
        # Add CLS token if needed
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        
        # Add position embeddings
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        return embeddings


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, config: VITConfig):
        super().__init__()
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} is not divisible by "
                f"number of attention heads {config.num_attention_heads}"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-head attention"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor
            attention_mask: Optional mask for attention
            
        Returns:
            Output of multi-head attention
        """
        # Project to Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize attention scores to probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # Reshape back
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)
        
        return context_layer


class VITSelfOutput(nn.Module):
    """Output projection and residual connection for attention"""
    
    def __init__(self, config: VITConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class VITAttention(nn.Module):
    """Complete attention module with output projection"""
    
    def __init__(self, config: VITConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.output = VITSelfOutput(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.output(attention_output, hidden_states)
        return attention_output


class VITMLP(nn.Module):
    """Feed-forward network (MLP) used in transformer blocks"""
    
    def __init__(self, config: VITConfig):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class VITLayer(nn.Module):
    """Single transformer encoder layer"""
    
    def __init__(self, config: VITConfig):
        super().__init__()
        self.attention = VITAttention(config)
        self.mlp = VITMLP(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LayerNorm architecture
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class VITEncoder(nn.Module):
    """
    Vision Transformer Encoder
    
    This is the main encoder class that stacks multiple transformer layers.
    It's designed for OCR tasks on historical Renaissance texts.
    """
    
    def __init__(self, config: VITConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            VITLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """
        Args:
            hidden_states: Input embeddings
            attention_mask: Optional attention mask
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            Tuple of (last_hidden_state, all_hidden_states)
        """
        all_hidden_states = () if output_hidden_states else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            hidden_states = layer(hidden_states, attention_mask)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        return hidden_states, all_hidden_states


class VITModel(nn.Module):
    """
    Complete Vision Transformer Model
    
    This combines embeddings and encoder for a full ViT model
    suitable for OCR applications.
    """
    
    def __init__(self, config: VITConfig, use_cls_token: bool = True):
        super().__init__()
        self.config = config
        self.embeddings = VITEmbeddings(config, use_cls_token=use_cls_token)
        self.encoder = VITEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights following standard ViT initialization"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, VITEmbeddings):
            nn.init.trunc_normal_(module.position_embeddings, std=0.02)
            if hasattr(module, 'cls_token'):
                nn.init.trunc_normal_(module.cls_token, std=0.02)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """
        Args:
            pixel_values: Input images of shape (batch_size, channels, height, width)
            attention_mask: Optional attention mask
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            Tuple of (last_hidden_state, all_hidden_states)
        """
        # Get embeddings
        embedding_output = self.embeddings(pixel_values)
        
        # Pass through encoder
        encoder_output, all_hidden_states = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states
        )
        
        # Apply final layer normalization
        sequence_output = self.layernorm(encoder_output)
        
        return sequence_output, all_hidden_states
    
    def get_config(self):
        """Return the model configuration"""
        return self.config


# Utility function to create a ViT encoder with default config
def create_vit_encoder(
    image_size: int = 224,
    patch_size: int = 16,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    use_cls_token: bool = True
) -> VITModel:
    """
    Factory function to create a Vision Transformer encoder
    
    Args:
        image_size: Input image size (default: 224)
        patch_size: Size of each patch (default: 16)
        hidden_size: Hidden dimension size (default: 768)
        num_layers: Number of transformer layers (default: 12)
        num_heads: Number of attention heads (default: 12)
        use_cls_token: Whether to use CLS token (default: True)
        
    Returns:
        VITModel instance
    """
    config = VITConfig(
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4
    )
    
    return VITModel(config, use_cls_token=use_cls_token)


if __name__ == "__main__":
    # Example usage and testing
    print("Vision Transformer Encoder for OCR")
    print("=" * 50)
    
    # Create a sample configuration
    config = VITConfig(
        image_size=224,
        patch_size=16,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12
    )
    
    # Create the model
    model = VITModel(config)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Configuration:")
    print(f"  Image Size: {config.image_size}x{config.image_size}")
    print(f"  Patch Size: {config.patch_size}x{config.patch_size}")
    print(f"  Number of Patches: {(config.image_size // config.patch_size) ** 2}")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Number of Layers: {config.num_hidden_layers}")
    print(f"  Number of Attention Heads: {config.num_attention_heads}")
    print(f"\nModel Parameters:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    
    # Test with dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\nTesting with dummy input:")
    print(f"  Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output, _ = model(dummy_input)
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: ({batch_size}, {(224//16)**2 + 1}, {768})")
    
    print("\n✓ VIT Encoder module is working correctly!")
