"""
RenAIssance OCR Backend — launcher script.

Delegates to the modular FastAPI application in app.main.
All business logic lives under the ``app`` package.
"""

import os
import sys

# Ensure the backend directory is on sys.path so that
# the ``preprocessing`` package (a sibling of ``app``) is importable.
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
