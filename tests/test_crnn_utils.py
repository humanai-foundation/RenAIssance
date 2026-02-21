"""
Unit tests for RenAIssance_CRNN_OCR_Shashank_Shekhar_Singh/utility/utils.py

Covers pure-logic and lightweight file I/O functions that require no GPU,
no real dataset, and no external binaries (fitz/docx are mocked in conftest.py).
"""
import os
import sys

import numpy as np
import pytest

# Make the CRNN sub-project importable
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "RenAIssance_CRNN_OCR_Shashank_Shekhar_Singh",
    ),
)

try:
    from utility.utils import (
        add_black_gaussian_noise,
        add_gaussian_noise,
        count_files_in_folder,
        count_lines_in_file,
        count_occurrences_of_semicolon,
        read_nth_line,
        remove_punctuation,
    )
except ImportError as exc:
    pytest.skip(f"CRNN utils import failed: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# remove_punctuation
# ---------------------------------------------------------------------------

class TestRemovePunctuation:
    def test_removes_comma_and_exclamation(self):
        assert remove_punctuation("Hello, world!") == "Hello world"

    def test_empty_string_returns_empty(self):
        assert remove_punctuation("") == ""

    def test_no_punctuation_unchanged(self):
        assert remove_punctuation("hello world") == "hello world"

    def test_digits_preserved(self):
        result = remove_punctuation("abc 123 def")
        assert "123" in result

    def test_all_punctuation_returns_whitespace_or_empty(self):
        result = remove_punctuation(".,!?;:")
        assert result.strip() == ""


# ---------------------------------------------------------------------------
# add_gaussian_noise / add_black_gaussian_noise
# ---------------------------------------------------------------------------

class TestGaussianNoise:
    def _make_image(self, h=64, w=64, c=3):
        return np.zeros((h, w, c), dtype=np.uint8)

    def test_output_shape_matches_input(self):
        img = self._make_image()
        noisy = add_gaussian_noise(img)
        assert noisy.shape == img.shape

    def test_output_is_ndarray(self):
        img = self._make_image()
        result = add_gaussian_noise(img, mean=0, std=5)
        assert isinstance(result, np.ndarray)

    def test_non_zero_std_produces_noise(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        noisy = add_gaussian_noise(img, mean=0, std=20)
        # With std=20 it is extremely unlikely all pixels are still 0
        assert not np.all(noisy == 0)

    def test_black_gaussian_noise_shape(self):
        img = self._make_image()
        result = add_black_gaussian_noise(img)
        assert result.shape == img.shape

    def test_black_gaussian_noise_is_ndarray(self):
        img = self._make_image()
        result = add_black_gaussian_noise(img, mean=0, std=25)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# count_files_in_folder
# ---------------------------------------------------------------------------

class TestCountFilesInFolder:
    def test_counts_matching_extension(self, tmp_path):
        (tmp_path / "a.png").touch()
        (tmp_path / "b.png").touch()
        (tmp_path / "c.txt").touch()
        assert count_files_in_folder(str(tmp_path), [".png"]) == 2

    def test_empty_folder_returns_zero(self, tmp_path):
        assert count_files_in_folder(str(tmp_path), [".png"]) == 0

    def test_multiple_extensions(self, tmp_path):
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.png").touch()
        (tmp_path / "c.txt").touch()
        assert count_files_in_folder(str(tmp_path), [".jpg", ".png"]) == 2


# ---------------------------------------------------------------------------
# count_occurrences_of_semicolon
# ---------------------------------------------------------------------------

class TestCountOccurrencesOfSemicolon:
    def test_counts_semicolons(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello;world;foo\nbar;\n")
        assert count_occurrences_of_semicolon(str(f)) == 3

    def test_no_semicolons_returns_zero(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("no semicolons here\n")
        assert count_occurrences_of_semicolon(str(f)) == 0


# ---------------------------------------------------------------------------
# read_nth_line
# ---------------------------------------------------------------------------

class TestReadNthLine:
    def test_returns_string_for_valid_index(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("line1\nline2\nline3\n")
        result = read_nth_line(str(f), 1)
        assert result is not None
        assert isinstance(result, str)

    def test_returns_none_for_out_of_range(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("only one line\n")
        assert read_nth_line(str(f), 9999) is None


# ---------------------------------------------------------------------------
# count_lines_in_file
# ---------------------------------------------------------------------------

class TestCountLinesInFile:
    def test_counts_lines(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("a\nb\nc\n")
        assert count_lines_in_file(str(f)) == 3

    def test_empty_file_returns_zero(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("")
        assert count_lines_in_file(str(f)) == 0
