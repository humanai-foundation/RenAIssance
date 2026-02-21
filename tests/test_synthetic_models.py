"""
Unit tests for RenAIssance_SyntheticImageGeneration_Saarthak_Gupta/src/model_utils.py

Instantiates UNetGenerator and PatchDiscriminator on CPU with small synthetic
tensors.  No GPU, no CRAFT model, no dataset required.
"""
import os
import sys

import pytest
import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "RenAIssance_SyntheticImageGeneration_Saarthak_Gupta",
    ),
)

try:
    from src.model_utils import PatchDiscriminator, UNetGenerator
except ImportError as exc:
    pytest.skip(f"Synthetic model_utils import failed: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# UNetGenerator
# ---------------------------------------------------------------------------

class TestUNetGenerator:
    def test_instantiation_default(self):
        model = UNetGenerator()
        assert model is not None

    def test_instantiation_custom_channels(self):
        model = UNetGenerator(in_channels=1, out_channels=1, features=32)
        assert model is not None

    def test_forward_output_shape(self):
        model = UNetGenerator(in_channels=1, out_channels=1)
        model.eval()
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 256, 256), f"Unexpected shape: {out.shape}"

    def test_output_in_tanh_range(self):
        """Generator's final activation is Tanh — output must be in [-1, 1]."""
        model = UNetGenerator(in_channels=1, out_channels=1)
        model.eval()
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.min().item() >= -1.0 - 1e-5
        assert out.max().item() <= 1.0 + 1e-5

    def test_output_is_finite(self):
        model = UNetGenerator(in_channels=1, out_channels=1)
        model.eval()
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert torch.all(torch.isfinite(out))

    def test_different_batch_sizes(self):
        model = UNetGenerator(in_channels=1, out_channels=1)
        model.eval()
        for b in [1, 2]:
            x = torch.randn(b, 1, 256, 256)
            with torch.no_grad():
                out = model(x)
            assert out.shape[0] == b


# ---------------------------------------------------------------------------
# PatchDiscriminator
# ---------------------------------------------------------------------------

class TestPatchDiscriminator:
    def test_instantiation_default(self):
        disc = PatchDiscriminator()
        assert disc is not None

    def test_instantiation_custom_channels(self):
        disc = PatchDiscriminator(in_channels=2, features=32)
        assert disc is not None

    def test_forward_returns_tensor(self):
        disc = PatchDiscriminator(in_channels=2)
        disc.eval()
        # Input = concatenated source + target (2 channels)
        x = torch.randn(1, 2, 256, 256)
        with torch.no_grad():
            out = disc(x)
        assert isinstance(out, torch.Tensor)

    def test_forward_output_is_patch_map(self):
        """PatchGAN spatial output must be smaller than input 256×256."""
        disc = PatchDiscriminator(in_channels=2)
        disc.eval()
        x = torch.randn(1, 2, 256, 256)
        with torch.no_grad():
            out = disc(x)
        assert out.shape[-1] < 256
        assert out.shape[-2] < 256

    def test_output_is_finite(self):
        disc = PatchDiscriminator(in_channels=2)
        disc.eval()
        x = torch.randn(1, 2, 256, 256)
        with torch.no_grad():
            out = disc(x)
        assert torch.all(torch.isfinite(out))
