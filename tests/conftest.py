"""
Pytest configuration — mock optional/heavy imports before any test module is
loaded so that test_crnn_utils.py can import utility/utils.py without requiring
PyMuPDF, python-docx, or other binary-dependent packages to be installed.
"""
import sys
from unittest.mock import MagicMock

_MOCK_MODULES = [
    "fitz",             # PyMuPDF  — CRNN utils.py pdf_to_images
    "docx",             # python-docx — CRNN utils.py save_pages_to_text
    "docx.shared",
    "docx.enum",
    "docx.enum.text",
]

for _mod in _MOCK_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
