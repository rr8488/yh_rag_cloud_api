from __future__ import annotations

import re
from typing import List, Optional, Dict
import io

# External deps
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("⚠️  pdfplumber not installed - table extraction disabled")


def _extract_page_with_tables(pdf_path: str, page_num: int) -> str:
    """
    Extract text and tables from a page using pdfplumber.
    """
    if not HAS_PDFPLUMBER:
        return ""

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num > len(pdf.pages):
                return ""

            page = pdf.pages[page_num - 1]

            # Get regular text
            text = page.extract_text() or ""

            # Extract tables
            tables = page.extract_tables()

            if tables:
                table_text = "\n\n=== TABLES ON THIS PAGE ===\n"

                for tidx, table in enumerate(tables):
                    if not table or len(table) == 0:
                        continue

                    table_text += f"\n[Table {tidx + 1}]\n"

                    # Process each row
                    for ridx, row in enumerate(table):
                        if not row:
                            continue

                        # Clean cells
                        cells = [str(cell or "").strip() for cell in row]

                        # Format as pipe-separated
                        table_text += " | ".join(cells) + "\n"

                        # Add separator after header row
                        if ridx == 0:
                            table_text += "-" * 100 + "\n"

                    table_text += "\n"

                # Append to text
                return text + "\n" + table_text

            return text

    except Exception as e:
        print(f"⚠️  pdfplumber failed for page {page_num}: {e}")
        return ""


def extract_text_per_page(
        pdf_path: str,
        *,
        ocr_fallback: bool = True,
        ocr_dpi: int = 300,
        ocr_lang: str = "eng",
        strip: bool = True,
        use_pdfplumber: bool = True,  # NEW
) -> List[str]:
    """
    Extract text from PDF pages with optional table extraction.
    """
    texts: List[str] = []

    with fitz.open(pdf_path) as doc:
        for page_num in range(1, doc.page_count + 1):
            txt = ""

            # Try pdfplumber first (better table handling)
            if use_pdfplumber and HAS_PDFPLUMBER:
                txt = _extract_page_with_tables(pdf_path, page_num)

            # Fallback to PyMuPDF
            if not txt:
                page = doc[page_num - 1]
                txt = page.get_text("text") or ""

            if strip:
                txt = txt.strip()

            # OCR fallback
            if ocr_fallback and len(txt) < 40:
                page = doc[page_num - 1]
                ocr_txt = _page_to_ocr_text(page, dpi=ocr_dpi, lang=ocr_lang).strip()
                if len(ocr_txt) > len(txt):
                    txt = ocr_txt

            texts.append(txt)

    return texts


def _page_to_ocr_text(page: "fitz.Page", dpi: int = 300, lang: str = "eng") -> str:
    """OCR a page."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    return pytesseract.image_to_string(img, lang=lang) or ""


def sniff_and_extract(
    pdf_path: str,
    *,
    prefer_ocr: bool = False,
    ocr_dpi: int = 300,
    ocr_lang: str = "eng",
) -> List[str]:
    """
    Convenience wrapper: if prefer_ocr=True, always OCR pages; else try native then fallback.
    """
    if prefer_ocr:
        # Force OCR on all pages
        pages: List[str] = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pages.append(_page_to_ocr_text(page, dpi=ocr_dpi, lang=ocr_lang).strip())
        return pages

    return extract_text_per_page(
        pdf_path,
        ocr_fallback=True,
        ocr_dpi=ocr_dpi,
        ocr_lang=ocr_lang,
        strip=True,
    )

PAGE_OF_RE = re.compile(r'\bPage\s+(\d+)\s+of\s+\d+\b', re.I)

def build_printed_to_pdf_map_no_interp(pages_text: List[str],
                                       footer_map: Optional[Dict[int,int]] = None
                                       ) -> Dict[int,int]:
    """
    Returns {printed_page: pdf_index} using only direct detections.
    No interpolation – gaps remain missing by design.
    """
    pp_to_pdf: Dict[int,int] = {}

    # (1) From inline text "Page X of Y"
    for pdf_idx, txt in enumerate(pages_text or []):
        if not txt:
            continue
        m = PAGE_OF_RE.search(txt)
        if m:
            pp = int(m.group(1))
            # keep first occurrence for a given printed page
            if pp not in pp_to_pdf:
                pp_to_pdf[pp] = pdf_idx

    # (2) Merge optional footer_map {pdf_idx -> printed_page}
    if footer_map:
        for pdf_idx, printed in footer_map.items():
            if printed not in pp_to_pdf:
                pp_to_pdf[printed] = pdf_idx

    return pp_to_pdf

def printed_to_pdf_index(pp_to_pdf: Dict[int,int], printed_page: int) -> Optional[int]:
    return pp_to_pdf.get(printed_page, None)