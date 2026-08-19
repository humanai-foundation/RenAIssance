"""Register DejaVu Sans so PDF export can render non-ASCII characters."""

import os
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

try:
    font_paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux (Debian/Ubuntu)
        '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',  # Linux (Fedora/RHEL)
        '/usr/share/fonts/TTF/DejaVuSans.ttf',  # Linux (Arch)
        'C:/Windows/Fonts/DejaVuSans.ttf',  # Windows
        '/System/Library/Fonts/Supplemental/DejaVuSans.ttf',  # macOS
        '/Library/Fonts/DejaVuSans.ttf',  # macOS alternative
    ]

    UNICODE_FONT_REGISTERED = False
    for font_path in font_paths:
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
            UNICODE_FONT_REGISTERED = True
            break

    if not UNICODE_FONT_REGISTERED:
        print("Warning: DejaVu Sans font not found. PDF export may have limited Unicode support.")
except Exception as e:
    UNICODE_FONT_REGISTERED = False
    print(f"Warning: Could not register Unicode font for PDF: {e}")
