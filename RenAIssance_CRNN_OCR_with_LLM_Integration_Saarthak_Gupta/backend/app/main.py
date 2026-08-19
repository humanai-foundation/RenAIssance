"""FastAPI app wiring: mount every router, start the signup-tracking loop.

Logic lives under app.api / app.services / app.core / app.schemas / app.utils.
"""

import os
import sys

# `preprocessing` is a sibling of `app`, not a submodule of it.
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import APP_TITLE, APP_VERSION, CORS_ORIGINS
from .api.health import router as health_router
from .api.ocr import router as ocr_router
from .api.preprocess import router as preprocess_router
from .api.export import router as export_router
from .api.layout_detection import router as layout_detection_router
from .api.dataset import router as dataset_router
from .api.recognition import router as recognition_router
from .api.llm_postprocess import router as llm_router
from .api.storage import router as storage_router
from .api.auth import router as auth_router
from .auth.db import init_db
from .auth.tracking import retry_loop, tracking_enabled

app = FastAPI(title=APP_TITLE, version=APP_VERSION)


@app.on_event("startup")
async def _startup() -> None:
    init_db()  # no-op once the tables exist
    # Pushes signups that never reached Supabase, including ones from while it
    # was down or unconfigured.
    if tracking_enabled():
        import asyncio

        app.state.tracking_task = asyncio.create_task(retry_loop())


@app.on_event("shutdown")
async def _shutdown() -> None:
    task = getattr(app.state, "tracking_task", None)
    if task is not None:
        task.cancel()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(ocr_router)
app.include_router(preprocess_router)
app.include_router(export_router)
app.include_router(layout_detection_router)
app.include_router(dataset_router)
app.include_router(recognition_router)
app.include_router(llm_router)
app.include_router(storage_router)
app.include_router(auth_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
