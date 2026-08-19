"""Export builders must emit non-empty, well-formed TXT/DOCX/PDF buffers."""

from app.services.export import (
    build_combined_transcript,
    build_docx_export,
    build_pdf_export,
    build_txt_export,
)


def test_combined_transcript_includes_all_pages(sample_transcripts):
    combined = build_combined_transcript(sample_transcripts)
    assert "Hello world" in combined
    assert "Second page of text" in combined


def test_txt_export_has_bom_and_content(sample_transcripts):
    data = build_txt_export(sample_transcripts).getvalue()
    assert data.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM
    assert "Hello world".encode("utf-8") in data


def test_docx_export_is_nonempty_zip(sample_transcripts):
    data = build_docx_export(sample_transcripts).getvalue()
    assert len(data) > 0
    # .docx is a zip container — starts with the PK signature.
    assert data[:2] == b"PK"


def test_pdf_export_has_pdf_magic(sample_transcripts):
    data = build_pdf_export(sample_transcripts).getvalue()
    assert data[:4] == b"%PDF"
