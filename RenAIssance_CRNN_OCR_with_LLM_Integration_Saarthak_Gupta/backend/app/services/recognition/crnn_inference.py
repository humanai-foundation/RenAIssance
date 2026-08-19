"""CRNN line-level OCR.

The architecture and helpers are copy-pasted from RenAIssanceExperimental on
purpose, so the backend doesn't depend on that repo at runtime.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from app.utils.torch_device import select_torch_device

logger = logging.getLogger(__name__)


# ── Model architecture ────────────────────────────────────────────────


class ResBlock(nn.Module):
    """Basic residual block with optional projection shortcut."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNetCNN(nn.Module):
    """Lightweight ResNet-style CNN backbone.  (B,1,H,W) → (B,256,1,W')"""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64), nn.MaxPool2d(2, 2))
        self.stage2 = nn.Sequential(ResBlock(64, 128), ResBlock(128, 128), nn.MaxPool2d(2, 2))
        self.stage3 = nn.Sequential(ResBlock(128, 256), ResBlock(256, 256), nn.MaxPool2d((2, 1)))
        self.stage4 = nn.Sequential(ResBlock(256, 256), nn.MaxPool2d((2, 1)))
        self.height_collapse = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.height_collapse(x)
        return x


class CRNN(nn.Module):
    """ResNet CNN → BiLSTM → Linear → CTC log-softmax."""

    def __init__(
        self,
        vocab_size: int,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cnn = ResNetCNN()
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)          # (B, 256, 1, W')
        x = x.squeeze(2)         # (B, 256, W')
        x = x.permute(0, 2, 1)   # (B, W', 256)
        x, _ = self.lstm(x)      # (B, W', H*2)
        x = self.dropout(x)
        x = self.classifier(x)   # (B, W', V)
        return x.log_softmax(-1)


# ── Helpers ───────────────────────────────────────────────────────────


def resize_keep_ratio(img: Image.Image, height: int = 64) -> Image.Image:
    """Resize a PIL image to a fixed height, preserving the aspect ratio."""
    w, h = img.size
    new_w = max(1, int(w * (height / h)))
    return img.resize((new_w, height), Image.BILINEAR)


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    idx2char: dict,
    blank: int = 0,
) -> list[str]:
    """Greedy best-path CTC decode.  (B, T, V) → list[str]"""
    preds = log_probs.argmax(-1).cpu().numpy()
    results = []
    for seq in preds:
        chars, prev = [], None
        for idx in seq:
            if idx != blank and idx != prev:
                chars.append(idx2char.get(int(idx), ""))
            prev = idx
        results.append("".join(chars))
    return results


def crop_polygon_gray(image_bgr: np.ndarray, poly_pts) -> Image.Image:
    """Crop a polygon region from a BGR image and return as grayscale PIL."""
    pts = np.array(poly_pts, dtype=np.float32)
    x1, y1 = pts.min(axis=0).astype(int)
    x2, y2 = pts.max(axis=0).astype(int)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_bgr.shape[1], x2)
    y2 = min(image_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return Image.new("L", (8, 8), 255)  # degenerate box
    crop = image_bgr[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))


# ── Model discovery ───────────────────────────────────────────────────


def discover_models(search_dir: str) -> list[dict]:
    """Find CRNN .pth/.pt checkpoints under search_dir.

    Returns [{"id": "crnn:<stem>", "name", "model_type", "path"}, ...].
    """
    models = []
    if not os.path.isdir(search_dir):
        logger.warning("Model search directory does not exist: %s", search_dir)
        return models

    for root, _dirs, files in os.walk(search_dir):
        for fname in files:
            if fname.endswith((".pth", ".pt")):
                if "crnn" not in fname.lower():
                    continue
                abs_path = os.path.abspath(os.path.join(root, fname))
                stem = os.path.splitext(fname)[0]
                models.append(
                    {
                        "id": f"crnn:{stem}",
                        "name": f"CRNN - {stem}",
                        "model_type": "crnn",
                        "path": abs_path,
                    }
                )
    return sorted(models, key=lambda item: item["name"].lower())


# ── Recognizer ────────────────────────────────────────────────────────


class CRNNRecognizer:
    """Loads the checkpoint on first predict, then keeps it in memory."""

    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model_path = model_path
        self._model: Optional[CRNN] = None
        self._idx2char: Optional[dict] = None
        self._img_height: int = 64

        if device is None:
            self.device = select_torch_device()
        else:
            self.device = device

        logger.info(
            "CRNNRecognizer created (model=%s, device=%s, lazy-load)",
            os.path.basename(model_path),
            self.device,
        )

    def _ensure_loaded(self):
        if self._model is not None:
            return

        logger.info("Loading CRNN checkpoint: %s", self.model_path)
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)

        self._idx2char = ckpt["idx2char"]
        self._img_height = ckpt.get("img_height", 64)

        self._model = CRNN(
            vocab_size=len(ckpt["vocab"]),
            lstm_hidden=ckpt.get("lstm_hidden", 256),
            lstm_layers=ckpt.get("lstm_layers", 2),
        ).to(self.device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()

        logger.info(
            "CRNN loaded — vocab=%d, img_height=%d, device=%s",
            len(ckpt["vocab"]),
            self._img_height,
            self.device,
        )

    def predict(self, image: Image.Image) -> str:
        """Recognise text from a single grayscale line image."""
        self._ensure_loaded()
        img = resize_keep_ratio(image.convert("L"), self._img_height)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        inp = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_probs = self._model(inp)
        return ctc_greedy_decode(log_probs, self._idx2char)[0]

    def predict_batch(self, images: list[Image.Image]) -> list[str]:
        """Recognise text from a batch of grayscale line images."""
        if not images:
            return []

        self._ensure_loaded()

        resized = [resize_keep_ratio(img.convert("L"), self._img_height) for img in images]

        tensors = []
        for img in resized:
            arr = np.asarray(img, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).unsqueeze(0))

        # Lines vary in width after the fixed-height resize, so pad to the widest.
        max_w = max(t.shape[-1] for t in tensors)
        padded = [torch.nn.functional.pad(t, (0, max_w - t.shape[-1])) for t in tensors]
        batch = torch.stack(padded).to(self.device)

        with torch.no_grad():
            log_probs = self._model(batch)

        return ctc_greedy_decode(log_probs, self._idx2char)


# ── Global model cache ────────────────────────────────────────────────

_recognizer_cache: dict[str, CRNNRecognizer] = {}


def get_recognizer(model_path: str) -> CRNNRecognizer:
    """Get (or create and cache) a CRNNRecognizer for the given path."""
    if model_path not in _recognizer_cache:
        _recognizer_cache[model_path] = CRNNRecognizer(model_path)
    return _recognizer_cache[model_path]
