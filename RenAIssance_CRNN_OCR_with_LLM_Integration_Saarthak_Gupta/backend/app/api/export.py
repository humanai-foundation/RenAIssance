"""Download the combined transcript as txt / docx / pdf."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..schemas.ocr import ExportRequest
from ..services.export import build_txt_export, build_docx_export, build_pdf_export


router = APIRouter()


@router.post("/api/export/txt")
async def export_txt(request: ExportRequest):
    buffer = build_txt_export(request.transcripts)

    return StreamingResponse(
        buffer,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=transcript_full.txt"}
    )


@router.post("/api/export/docx")
async def export_docx(request: ExportRequest):
    buffer = build_docx_export(request.transcripts)

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=transcript_full.docx"}
    )


@router.post("/api/export/pdf")
async def export_pdf(request: ExportRequest):
    buffer = build_pdf_export(request.transcripts)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=transcript_full.pdf"}
    )
