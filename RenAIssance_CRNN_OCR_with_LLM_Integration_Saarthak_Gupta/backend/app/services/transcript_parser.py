"""Split a transcript file (TXT/DOCX/PDF/MD) into {page_key: [lines]}.

Page markers are matched loosely — "PDF p1", "--- Page 4 - left ---",
"[Page 5]", "Page 6", "p1-right" all work, case-insensitively.
"""

import io
import re
from typing import Dict, List, Optional


# Optional " left"/" right" suffix, with any of space/-/–/— as the separator.
_SIDE_SUFFIX = r"(?:[\s\-–—]+(?P<side>left|right))?"

_PAGE_PATTERNS = [
    # "PDF p1", "PDF p2 - left", "PDF p2 left"
    re.compile(
        r"^\s*PDF\s+p\s*(?P<num>\d+)" + _SIDE_SUFFIX + r"\s*$",
        re.IGNORECASE,
    ),
    # "--- Page 4 ---" or "--- page 4 left ---" or "--- page 4 - right ---"
    re.compile(
        r"^\s*-{2,}\s*[Pp]age\s+(?P<num>\d+)" + _SIDE_SUFFIX + r"\s*-*\s*$",
        re.IGNORECASE,
    ),
    # "[Page 5]" or "[Page 5 - left]" or "[Page 5 right]"
    re.compile(
        r"^\s*\[\s*[Pp]age\s+(?P<num>\d+)" + _SIDE_SUFFIX + r"\s*\]\s*$",
        re.IGNORECASE,
    ),
    # Bare "Page 6" / "Page 6 left" / "page 6 - right"
    re.compile(
        r"^\s*[Pp]age\s+(?P<num>\d+)" + _SIDE_SUFFIX + r"\s*$",
        re.IGNORECASE,
    ),
    # Shorthand "p1" / "p1 left" / "p1-right"
    re.compile(
        r"^\s*[Pp](?P<num>\d+)" + _SIDE_SUFFIX + r"\s*$",
        re.IGNORECASE,
    ),
]


def _match_page_marker(line: str) -> Optional[str]:
    """-> '3' / '3_left' if the line is a page marker, else None."""
    for pat in _PAGE_PATTERNS:
        m = pat.match(line)
        if m:
            page_num = m.group("num")
            try:
                side = m.group("side")
            except IndexError:
                side = None
            if side:
                return f"{page_num}_{side.lower()}"
            return page_num
    return None


# ── Text extraction per format ──────────────────────────────────────

def _extract_text_from_txt(data: bytes) -> str:
    text = data.decode("utf-8-sig", errors="replace")
    return text


def _extract_text_from_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_text_from_pdf(data: bytes) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError(
            "PyMuPDF (fitz) is required to parse PDF transcripts. "
            "Install it with: pip install PyMuPDF"
        )
    doc = fitz.open(stream=data, filetype="pdf")
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()
    return "\n".join(pages_text)


def _extract_text_from_markdown(data: bytes) -> str:
    return data.decode("utf-8-sig", errors="replace")


_EXTRACTORS = {
    "text/plain": _extract_text_from_txt,
    "text/markdown": _extract_text_from_markdown,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": _extract_text_from_docx,
    "application/pdf": _extract_text_from_pdf,
}

# Used when the browser sends a useless content-type.
_EXT_MAP = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
}


def extract_text(data: bytes, filename: str = "", content_type: str = "") -> str:
    """Get raw text out of the uploaded bytes, by content-type or extension."""
    mime = content_type.lower().split(";")[0].strip() if content_type else ""

    if mime not in _EXTRACTORS:
        import os
        ext = os.path.splitext(filename)[1].lower()
        mime = _EXT_MAP.get(ext, "text/plain")

    extractor = _EXTRACTORS.get(mime, _extract_text_from_txt)
    return extractor(data)


# ── Split into pages ────────────────────────────────────────────────

def parse_transcript(
    raw_text: str,
    default_page: str = "1",
) -> Dict[str, List[str]]:
    """Split raw text on page markers into {page_key: [lines]}.

    No markers at all -> everything lands under default_page. Blank lines go.
    """
    pages: Dict[str, List[str]] = {}
    lines = raw_text.splitlines()
    has_any_marker = any(_match_page_marker(line) is not None for line in lines)

    # With markers present, anything before the first one is preface — drop it.
    current_key: Optional[str] = None if has_any_marker else default_page

    for raw_line in lines:
        marker = _match_page_marker(raw_line)
        if marker is not None:
            current_key = marker
            continue

        if current_key is None:  # still in the preface
            continue

        cleaned = raw_line.strip()
        if not cleaned:
            continue
        if re.match(r"^end\s+of\s+extract\s*$", cleaned, re.IGNORECASE):
            continue
        pages.setdefault(current_key, []).append(cleaned)

    return pages


def parse_transcript_bytes(
    data: bytes,
    filename: str = "",
    content_type: str = "",
) -> Dict[str, List[str]]:
    """extract_text + parse_transcript in one call."""
    raw = extract_text(data, filename, content_type)
    return parse_transcript(raw)
