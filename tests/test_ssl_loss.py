"""
Unit tests for:
  - RenAIssance_SelfSupervisedLearning_OCR_YukinoriYamamoto/custom_loss.py
    cosine_similarity, noise_contrastive_estimation, contrastive_loss
  - RenAIssance_SelfSupervisedLearning_OCR_YukinoriYamamoto/custom_dataset.py
    resize_and_pad, RandomVerticalCrop

All tests run on CPU tensors / PIL images — no GPU, no dataset required.
"""
import os
import sys

import pytest
import torch
from PIL import Image

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "RenAIssance_SelfSupervisedLearning_OCR_YukinoriYamamoto",
    ),
)

try:
    from custom_loss import (
        contrastive_loss,
        cosine_similarity,
        noise_contrastive_estimation,
    )
except ImportError as exc:
    pytest.skip(f"SSL custom_loss import failed: {exc}", allow_module_level=True)

try:
    from custom_dataset import RandomVerticalCrop, resize_and_pad
except ImportError as exc:
    pytest.skip(f"SSL custom_dataset import failed: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_returns_tensor(self):
        x = torch.randn(2, 4, 8)
        y = torch.randn(2, 4, 8)
        result = cosine_similarity(x, y)
        assert isinstance(result, torch.Tensor)

    def test_output_is_finite(self):
        x = torch.randn(2, 4, 8)
        y = torch.randn(2, 4, 8)
        result = cosine_similarity(x, y)
        assert torch.all(torch.isfinite(result))

    def test_identical_inputs_produce_finite_result(self):
        x = torch.randn(1, 3, 8)
        result = cosine_similarity(x, x)
        assert torch.all(torch.isfinite(result))

    def test_batch_size_preserved(self):
        x = torch.randn(3, 4, 8)
        y = torch.randn(3, 4, 8)
        result = cosine_similarity(x, y)
        # batch dimension must match
        assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# noise_contrastive_estimation
# ---------------------------------------------------------------------------

class TestNoiseContrastiveEstimation:
    def test_returns_scalar(self):
        x = torch.randn(2, 4, 8)
        y = torch.randn(2, 4, 8)
        loss = noise_contrastive_estimation(x, y)
        assert loss.ndim == 0

    def test_is_finite(self):
        x = torch.randn(2, 4, 8)
        y = torch.randn(2, 4, 8)
        loss = noise_contrastive_estimation(x, y)
        assert torch.isfinite(loss)

    def test_is_non_negative(self):
        x = torch.randn(2, 4, 8)
        y = torch.randn(2, 4, 8)
        loss = noise_contrastive_estimation(x, y)
        assert loss.item() >= 0.0

    def test_different_batch_sizes(self):
        for b in [1, 4, 8]:
            x = torch.randn(b, 4, 8)
            y = torch.randn(b, 4, 8)
            loss = noise_contrastive_estimation(x, y)
            assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# contrastive_loss
# ---------------------------------------------------------------------------

class TestContrastiveLoss:
    def test_returns_scalar(self):
        x = torch.randn(3, 4, 8)
        y = torch.randn(3, 4, 8)
        loss = contrastive_loss(x, y)
        assert loss.ndim == 0

    def test_is_finite(self):
        x = torch.randn(2, 4, 8)
        y = torch.randn(2, 4, 8)
        loss = contrastive_loss(x, y)
        assert torch.isfinite(loss)

    def test_symmetric_inputs_give_finite_loss(self):
        x = torch.randn(2, 4, 8)
        loss = contrastive_loss(x, x)
        assert torch.isfinite(loss)

    def test_is_non_negative(self):
        x = torch.randn(2, 4, 8)
        y = torch.randn(2, 4, 8)
        loss = contrastive_loss(x, y)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_is_pil_image(self):
        img = Image.new("RGB", (100, 50))
        result = resize_and_pad(img, (64, 384))
        assert isinstance(result, Image.Image)

    def test_output_matches_target_size(self):
        img = Image.new("L", (200, 100))
        result = resize_and_pad(img, (64, 384))
        # PIL .size = (width, height)
        assert result.size == (384, 64)

    def test_handles_tall_image(self):
        img = Image.new("RGB", (30, 200))
        result = resize_and_pad(img, (64, 384))
        assert isinstance(result, Image.Image)

    def test_handles_square_image(self):
        img = Image.new("RGB", (100, 100))
        result = resize_and_pad(img, (64, 384))
        assert result.size == (384, 64)


# ---------------------------------------------------------------------------
# RandomVerticalCrop
# ---------------------------------------------------------------------------

class TestRandomVerticalCrop:
    def test_output_is_pil_image(self):
        crop = RandomVerticalCrop(crop_height_ratio=0.2)
        img = Image.new("RGB", (100, 100))
        result = crop(img)
        assert isinstance(result, Image.Image)

    def test_output_width_unchanged(self):
        crop = RandomVerticalCrop(crop_height_ratio=0.1)
        img = Image.new("RGB", (200, 100))
        result = crop(img)
        assert result.size[0] == 200

    def test_zero_ratio_returns_full_height(self):
        crop = RandomVerticalCrop(crop_height_ratio=0.0)
        img = Image.new("RGB", (150, 80))
        result = crop(img)
        assert isinstance(result, Image.Image)
        assert result.size[1] == 80

    def test_repeated_calls_yield_pil_image(self):
        crop = RandomVerticalCrop(crop_height_ratio=0.15)
        img = Image.new("L", (128, 64))
        for _ in range(5):
            assert isinstance(crop(img), Image.Image)
