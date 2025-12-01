# yh_rag_cloud_api/deps.py
from __future__ import annotations
from dataclasses import dataclass
from fastapi import Query
from .settings import settings

@dataclass
class OcrParams:
    fallback: bool
    dpi: int
    lang: str

# Define the dependency *once*, but branch based on ENV at import time.
if settings.env == "dev":
    # Dev: show as query params in Swagger; callers can tweak.
    def get_ocr_params(
        ocr_fallback: bool = Query(settings.ocr_fallback),
        ocr_dpi: int = Query(settings.ocr_dpi, ge=72, le=600),
        ocr_lang: str = Query(",".join(settings.ocr_langs)),  # ← Convert list to comma-separated string
    ) -> OcrParams:
        return OcrParams(
            fallback=ocr_fallback,
            dpi=max(72, min(600, int(ocr_dpi))),
            lang=ocr_lang,
        )
else:
    # Prod: no query params; values fixed from settings.
    def get_ocr_params() -> OcrParams:
        return OcrParams(
            fallback=settings.ocr_fallback,
            dpi=max(72, min(600, int(settings.ocr_dpi))),
            lang=",".join(settings.ocr_langs),  # ← Convert list to comma-separated string
        )