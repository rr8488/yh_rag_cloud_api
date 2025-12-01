# ai_doc_nav_api/routers/parse_pdf.py
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Query, Depends
import tempfile, os

from ..pdf_parser.parse_pdf_auto import parse_pdf_auto
from ..deps import get_ocr_params, OcrParams
from ..settings import settings

router = APIRouter(prefix="/parse", tags=["parse"])

@router.post("/pdf")
async def parse_pdf(
    file: UploadFile = File(...),
    prefer_dynamic: bool = Query(False),
    ocr: OcrParams = Depends(get_ocr_params),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        pdf_path = tmp.name
    try:
        cfg, chunks = parse_pdf_auto(
            pdf_path,
            prefer_dynamic=prefer_dynamic,
            min_builtin_toc_items=settings.MIN_BUILTIN_TOC_ITEMS,
            ocr_fallback=ocr.fallback,
            ocr_dpi=ocr.dpi,
            ocr_lang=ocr.lang,
        )
        return {
            "meta": {
                "pages": (max((c["end_page"] for c in chunks), default=None)),
                "sections": len(cfg.get("sections", [])),
                "chunks": sum(1 + len(c.get("children", [])) for c in chunks),
                "env": settings.ENV,
                "strategy": "dynamic" if prefer_dynamic else "auto",
            },
            "config": cfg,
            "chunks": chunks,
        }
    finally:
        try: os.remove(pdf_path)
        except Exception: pass