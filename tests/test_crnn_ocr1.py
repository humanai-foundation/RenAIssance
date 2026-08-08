"""
RenAIssance OCR-1 — Unit Tests for CRNN Module
Issue #57 : Add automated tests for CRNN/OCR-1
Author   : Abhiram G (abhiram123467)
GSoC 2026 | HumanAI Foundation
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 
                'RenAIssance_SelfSupervisedLearning_OCR_YukinoriYamamoto'))

from ResNet import ResNet18, ResNet34, ResNet50
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from llm_postprocess import clean_raw_ocr, compute_cer


class TestResNet18:

    def test_output_shape(self):
        model = ResNet18(num_classes=3)
        x = torch.randn(2, 1, 32, 128)
        out = model(x)
        assert out.shape == (2, 3), f"Expected (2,3), got {out.shape}"

    def test_num_classes_respected(self):
        for n in [3, 5, 10]:
            model = ResNet18(num_classes=n)
            x = torch.randn(1, 1, 32, 128)
            out = model(x)
            assert out.shape[1] == n

    def test_fc_layer_exists(self):
        model = ResNet18(num_classes=3)
        assert hasattr(model, 'fc'), "ResNet18 missing fc layer"

    def test_no_nan_output(self):
        model = ResNet18(num_classes=3)
        x = torch.randn(2, 1, 32, 128)
        out = model(x)
        assert not torch.isnan(out).any(), "Output contains NaN"


class TestResNet34:

    def test_output_shape(self):
        model = ResNet34(num_classes=3)
        x = torch.randn(2, 1, 32, 128)
        out = model(x)
        assert out.shape == (2, 3), f"Expected (2,3), got {out.shape}"

    def test_fc_layer_exists(self):
        model = ResNet34(num_classes=3)
        assert hasattr(model, 'fc'), "ResNet34 missing fc layer"

    def test_num_classes_respected(self):
        for n in [3, 5, 10]:
            model = ResNet34(num_classes=n)
            x = torch.randn(1, 1, 32, 128)
            out = model(x)
            assert out.shape[1] == n


class TestCleanRawOCR:

    def test_removes_extra_spaces(self):
        result = clean_raw_ocr("hello   world")
        assert "  " not in result

    def test_strips_whitespace(self):
        result = clean_raw_ocr("  hello world  ")
        assert result == "hello world"

    def test_empty_string(self):
        result = clean_raw_ocr("")
        assert result == ""

    def test_normal_text_unchanged(self):
        result = clean_raw_ocr("Esta es la instruccion")
        assert result == "Esta es la instruccion"


class TestComputeCER:

    def test_identical_strings(self):
        assert compute_cer("hello", "hello") == 0.0

    def test_completely_different(self):
        cer = compute_cer("abc", "xyz")
        assert cer == 1.0

    def test_one_substitution(self):
        cer = compute_cer("hello", "hella")
        assert cer == pytest.approx(0.2, 0.01)

    def test_empty_reference(self):
        cer = compute_cer("", "hello")
        assert cer == 0.0

    def test_partial_match(self):
        cer = compute_cer("abcde", "abcxy")
        assert 0.0 < cer < 1.0
```

---

Once pasted, scroll down and write this commit message:
```
ci: Add unit tests for CRNN OCR-1 module (refs #57)
