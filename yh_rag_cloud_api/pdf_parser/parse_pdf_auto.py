# ai_doc_nav_api/pdf_parser/parse_pdf_auto.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple, Optional

import fitz  # PyMuPDF
from pathlib import Path

# OCR imports (optional)
try:
    import pytesseract
    from PIL import Image
    import io

    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("⚠️  pytesseract not available - OCR fallback disabled")
    print("   Install with: pip install pytesseract pillow")

# Built-in (bookmark/outline) pipeline
try:
    from ..pdf_parse import parse_pdf_to_toc_and_chunks as builtin_parse
except Exception:
    builtin_parse = None

# Dynamic pipeline
from .text_extract import extract_text_per_page
from .dynamic_toc import parse_pdf_to_chunks, split_by_config, split_by_llm_sections
from .regex_toc_extractor import extract_toc_with_regex
from .llm_toc_extractor import extract_toc_with_llm


def extract_text_with_ocr_fallback(page: "fitz.Page", ocr_enabled: bool = True, dpi: int = 300) -> str:
    """
    Extract text from a PDF page with OCR fallback for scanned pages.

    Args:
        page: PyMuPDF page object
        ocr_enabled: Whether to try OCR if standard extraction fails
        dpi: DPI for OCR rendering

    Returns:
        Extracted text
    """
    # Try standard extraction first
    try:
        text = page.get_text()
        if text and text.strip() and len(text.strip()) > 50:
            return text
    except Exception as e:
        print(f"   Warning: Standard text extraction failed: {e}")

    # If OCR is not available or disabled, return what we have
    if not ocr_enabled or not HAS_OCR:
        return text if text else ""

    # Try OCR fallback
    try:
        print(f"   📷 Using OCR for page {page.number + 1}...")

        # Render page as image
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        # Run OCR
        ocr_text = pytesseract.image_to_string(img)

        if ocr_text and ocr_text.strip():
            print(f"   ✓ OCR extracted {len(ocr_text)} characters")
            return ocr_text
        else:
            print(f"   ⚠️  OCR found no text")
            return text if text else ""

    except Exception as e:
        print(f"   ⚠️  OCR failed: {e}")
        return text if text else ""


def _has_usable_builtin_toc(doc: "fitz.Document", min_items: int = 3) -> bool:
    """Decide if the PDF's internal outline is usable."""
    try:
        toc = doc.get_toc()
    except Exception:
        return False
    if not toc or len(toc) < min_items:
        return False

    levels = {row[0] for row in toc if isinstance(row, (list, tuple)) and len(row) >= 3}
    if 1 not in levels:
        return False

    page_nums = [row[2] for row in toc if isinstance(row, (list, tuple)) and len(row) >= 3 and isinstance(row[2], int)]
    if not page_nums:
        return False

    if min(page_nums) < 1 or max(page_nums) > max(1, doc.page_count):
        return False

    return True


def _normalize_result(res: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Normalize various return shapes into (config, chunks)."""
    if isinstance(res, tuple) and len(res) == 2:
        return res
    if isinstance(res, dict):
        if "config" in res and "chunks" in res:
            return res["config"], res["chunks"]
        if "toc" in res and "sections" in res:
            return res.get("toc", {}), res["sections"]
    return {"version": 1, "sections": []}, []


def parse_pdf_auto(
        pdf_path: str,
        *,
        prefer_dynamic: bool = False,
        use_llm_toc: bool = True,
        min_builtin_toc_items: int = 3,
        ocr_fallback: bool = True,
        ocr_dpi: int = 300,
        ocr_lang: str = "eng",
        ollama_url: str = "http://localhost:11434",
        llm_model: str = "llama3.1:8b"
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Auto-select the best parsing strategy for a PDF.
    Priority: Built-in bookmarks > Regex TOC > LLM TOC > Regex fallback
    """
    print(f"\n{'=' * 60}")
    print(f"Reading PDF: {pdf_path}")
    print(f"{'=' * 60}\n")

    # Fast path: built-in bookmarks
    if not prefer_dynamic and builtin_parse is not None:
        try:
            with fitz.open(pdf_path) as doc:
                if _has_usable_builtin_toc(doc, min_items=min_builtin_toc_items):
                    print("✅ Using built-in PDF bookmarks")
                    res = builtin_parse(pdf_path)
                    return _normalize_result(res)
        except Exception as e:
            print(f"⚠️  Built-in bookmark extraction failed: {e}")

    # Extract text from all pages
    print("Extracting text from PDF pages...")
    pages: List[str] = extract_text_per_page(
        pdf_path,
        ocr_fallback=ocr_fallback,
        ocr_dpi=ocr_dpi,
        ocr_lang=ocr_lang,
        strip=True,
        use_pdfplumber=True,
    )
    print(f"Extracted {len(pages)} pages\n")

    # Try regex-based TOC extraction (fast and accurate)
    if use_llm_toc:
        try:
            print("Extracting TOC using regex patterns...")
            cfg = extract_toc_with_regex(pages)

            if cfg.get("sections") and len(cfg["sections"]) >= 3:
                print(f"Regex extracted {len(cfg['sections'])} top-level sections\n")
                print("Splitting document by detected sections...")

                chunks = split_by_llm_sections(pages, cfg)

                print(f"Created {len(chunks)} content sections\n")
                return cfg, chunks
            else:
                print("Regex found insufficient sections, trying LLM fallback...\n")

        except Exception as e:
            print(f"⚠️  Regex extraction failed: {e}")
            print("   Trying LLM fallback...\n")

    # Fallback to LLM-based extraction
    if use_llm_toc:
        try:
            print("Using LLM to extract Table of Contents...")
            cfg = extract_toc_with_llm(pages, api_url=ollama_url, model=llm_model)

            if cfg.get("sections"):
                print(f"LLM extracted {len(cfg['sections'])} top-level sections\n")
                print("Splitting document by detected sections...")

                chunks = split_by_llm_sections(pages, cfg)

                print(f"Created {len(chunks)} content sections\n")
                return cfg, chunks
            else:
                print("LLM found no TOC, falling back to regex detection\n")
        except Exception as e:
            print(f"⚠️  LLM extraction failed: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to regex detection\n")

    # Final fallback: regex-based body detection
    print("Using regex-based body detection...")
    cfg, chunks = parse_pdf_to_chunks(pages)
    print(f"Created {len(chunks)} content chunks\n")
    return cfg, chunks


def extract_page_number_from_text_content(text: str) -> Optional[int]:
    """
    Extract page number from already-extracted text content.
    Handles "Page X of Y" format correctly.
    """
    if not text or len(text) < 5:
        return None

    # Priority patterns - extract the CURRENT page number (not total)
    patterns = [
        # BEST: "Page X of Y" - capture X (group 1)
        (r'Page\s+(\d+)\s+of\s+\d+', 1),
        # GOOD: "X of Y" - capture X (group 1)
        (r'(\d+)\s+of\s+\d+', 1),
        # FALLBACK: "Page X" at start/end
        (r'Page\s+(\d+)(?:\s|$)', 1),
    ]

    # Search in the LAST 500 chars (footers are usually at end)
    search_text = text[-500:] if len(text) > 500 else text

    for pattern, group_idx in patterns:
        match = re.search(pattern, search_text, re.IGNORECASE)
        if match:
            try:
                num = int(match.group(group_idx))
                # Sanity check: page number should be reasonable
                if 1 <= num <= 500:
                    return num
            except (ValueError, IndexError):
                pass

    return None


def interpolate_page_map(page_map: Dict[int, int], total_pdf_pages: int) -> Dict[int, int]:
    """
    Fill gaps in page map by interpolating between known pages.

    Example: If we have page 26→20 and page 33→25,
    we can estimate that page 29→22 or 23
    """
    if len(page_map) < 2:
        return page_map

    sorted_pages = sorted(page_map.items())  # [(printed_page, pdf_idx), ...]
    complete_map = dict(page_map)  # Start with known pages

    print("\n🔧 Interpolating missing pages...")
    interpolated_count = 0

    for i in range(len(sorted_pages) - 1):
        curr_printed, curr_pdf_idx = sorted_pages[i]
        next_printed, next_pdf_idx = sorted_pages[i + 1]

        gap = next_printed - curr_printed

        if gap > 1:  # There are missing pages between these two
            # Linear interpolation
            for j in range(1, gap):
                missing_printed = curr_printed + j

                if missing_printed not in complete_map:  # Don't overwrite known pages
                    # Estimate PDF index
                    ratio = j / gap
                    estimated_pdf_idx = int(curr_pdf_idx + ratio * (next_pdf_idx - curr_pdf_idx))
                    estimated_pdf_idx = max(0, min(estimated_pdf_idx, total_pdf_pages - 1))

                    complete_map[missing_printed] = estimated_pdf_idx
                    interpolated_count += 1

                    if interpolated_count <= 10:  # Show first 10
                        print(f"   📍 Page {missing_printed} estimated → PDF index {estimated_pdf_idx}")

    if interpolated_count > 10:
        print(f"   ... and {interpolated_count - 10} more")

    print(f"✅ Interpolated {interpolated_count} missing pages\n")

    return complete_map


def build_page_number_map_from_extracted_text(pages: List[str]) -> Dict[int, int]:
    """
    Build mapping of printed page numbers to physical indices with interpolation.
    """
    page_map = {}
    page_to_indices = {}  # printed_page -> [list of physical indices]

    print("\nBuilding page number map from extracted text content...")
    print(f"Processing {len(pages)} pages...\n")

    # Step 1: Extract page numbers from all pages
    readable_pages = []

    for physical_idx, page_text in enumerate(pages):
        printed_num = extract_page_number_from_text_content(page_text)

        if printed_num:
            readable_pages.append((physical_idx, printed_num))

            # Track duplicates
            if printed_num not in page_to_indices:
                page_to_indices[printed_num] = []
            page_to_indices[printed_num].append(physical_idx)

            # Print progress
            if len(readable_pages) <= 20 or physical_idx % 20 == 0:
                print(f"   ✓ PDF index {physical_idx:3d} → Printed page {printed_num:3d}")

    if not readable_pages:
        print("   ⚠️  ERROR: No page numbers extracted")
        print("   Using fallback: 1:1 mapping\n")
        return {i + 1: i for i in range(len(pages))}

    print(f"\n   Found {len(readable_pages)} pages with readable numbers")

    # Step 2: Detect problems (duplicate page numbers)
    duplicates = {num: indices for num, indices in page_to_indices.items() if len(indices) > 1}
    if duplicates:
        print(f"\n   ⚠️  WARNING: Duplicate page numbers:")
        for num, indices in sorted(duplicates.items())[:5]:
            print(f"      Page {num}: at PDF indices {indices[:5]}")

    # Step 3: Build direct map from found pages
    readable_pages.sort(key=lambda x: x[1])

    for phys_idx, print_num in readable_pages:
        # Only use first occurrence if duplicates
        if print_num not in page_map:
            page_map[print_num] = phys_idx

    print(f"\n   Sequence: {readable_pages[0][1]} → {readable_pages[-1][1]}")
    print(f"\nDirect page map: {len(page_map)} pages")

    # Step 4: Interpolate missing pages
    complete_map = interpolate_page_map(page_map, len(pages))

    print(f"Final page map: {len(complete_map)} distinct pages\n")

    # Show mapping
    sorted_map = sorted(complete_map.items())
    for print_page, pdf_idx in sorted_map[:10]:
        print(f"   Page {print_page:3d} → PDF index {pdf_idx:3d}")
    if len(sorted_map) > 20:
        print(f"   ... ({len(sorted_map) - 20} more) ...")
        for print_page, pdf_idx in sorted_map[-10:]:
            print(f"   Page {print_page:3d} → PDF index {pdf_idx:3d}")

    return complete_map


# ai_doc_nav_api/pdf_parser/parse_pdf_auto.py

def extract_page_number_from_page_with_rotation(page: "fitz.Page") -> Optional[int]:
    """
    Extract printed page number from footer, handling rotated pages.

    Args:
        page: PyMuPDF page object

    Returns:
        Page number or None
    """
    # Try all 4 rotations (0°, 90°, 180°, 270°)
    rotations = [0, 90, 180, 270]

    for rotation in rotations:
        try:
            # Create a temporary rotated page if needed
            if rotation != 0:
                # Get page as pixmap, rotate it, then extract text
                mat = fitz.Matrix(1, 1).prerotate(rotation)
                pix = page.get_pixmap(matrix=mat)
                # Convert back to temp PDF to extract text
                temp_pdf = fitz.open()
                temp_page = temp_pdf.new_page(width=pix.width, height=pix.height)
                temp_page.insert_image(temp_page.rect, pixmap=pix)
                text = temp_page.get_text()
                temp_pdf.close()
            else:
                text = page.get_text()

            # Look for page number pattern in footer area
            # Pattern: "Page X of Y" where X is what we want
            patterns = [
                r'Page\s+(\d+)\s+of\s+\d+',  # "Page 36 of 157"
                r'(\d+)\s+of\s+\d+',  # "36 of 157"
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    page_num = int(match.group(1))
                    # Sanity check
                    if 1 <= page_num <= 500:
                        if rotation != 0:
                            print(f"      📐 Page rotated {rotation}° - extracted page {page_num}")
                        return page_num
        except Exception as e:
            continue

    return None


def build_physical_page_to_printed_map(pdf_path: str) -> Dict[int, int]:
    """
    Build mapping: {physical_pdf_index: printed_page_number}
    FALLBACK: If no numbers found, return a 1:1 mapping {0: 1, 1: 2, ...}
    """
    # ... (keep the initial print statements) ...
    print("\n" + "=" * 80)
    print("Building Physical-to-Printed Page Map")
    print("=" * 80)

    doc = None # Initialize doc to None
    physical_to_printed = {}
    total_pages = 0

    try: # Add try block for file opening
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"\nScanning {total_pages} pages for footer page numbers...\n")

        for physical_idx in range(total_pages):
            page = doc[physical_idx]
            printed_num = extract_page_number_from_page_with_rotation(page) # Reuse existing helper

            if printed_num:
                physical_to_printed[physical_idx] = printed_num
                if physical_idx < 20 or physical_idx % 10 == 0:
                    print(f"   ✓ PDF index {physical_idx:3d} → Printed page {printed_num:3d}")
            # --- REMOVED the else print statement for missing numbers ---
            # else:
            #     print(f"   ⚠️  PDF index {physical_idx:3d} → No page number found")

    except Exception as e:
         print(f"  ERROR opening or scanning PDF for page numbers: {e}")
         # If we can't even open the doc, return an empty map
         return {}
    finally:
        if doc:
            doc.close() # Ensure doc is closed even if errors occur

    # --- MODIFICATION START: Add Fallback Logic ---
    if not physical_to_printed and total_pages > 0:
        print("\n   ⚠️  WARNING: No footer page numbers found in the entire document.")
        print("   Using fallback: 1-to-1 mapping (Physical Index + 1 = Printed Page)\n")
        # Create a simple mapping: {0: 1, 1: 2, 2: 3, ...}
        physical_to_printed = {i: i + 1 for i in range(total_pages)}
        # Print a sample of the fallback map
        for i in range(min(total_pages, 5)):
             print(f"   Fallback: PDF index {i:3d} → Assumed page {i+1:3d}")
        if total_pages > 5: print("   ...")

    elif physical_to_printed:
         print(f"\n✅ Found {len(physical_to_printed)} pages with printed numbers")
         print(f"   (Missing/unreadable numbers for {total_pages - len(physical_to_printed)} pages)\n")
    # --- MODIFICATION END ---
    else: # Case where total_pages was 0 (empty doc?)
        print("\n   INFO: PDF appears to have 0 pages or could not be read.\n")


    return physical_to_printed


def find_printed_page_range(
        toc_start_page: int,
        toc_end_page: int,
        physical_to_printed: Dict[int, int]
) -> Optional[Tuple[int, int]]:
    """
    Find physical PDF indices for a range of printed page numbers.

    Args:
        toc_start_page: Starting printed page (e.g., 36)
        toc_end_page: Ending printed page (e.g., 40)
        physical_to_printed: Mapping of {physical_idx: printed_page}

    Returns:
        (start_physical_idx, end_physical_idx) or None if not found
    """
    # Build reverse map: {printed_page: physical_idx}
    printed_to_physical = {v: k for k, v in physical_to_printed.items()}

    # Find physical indices for the page range
    physical_indices = []

    for printed_page in range(toc_start_page, toc_end_page + 1):
        if printed_page in printed_to_physical:
            physical_indices.append(printed_to_physical[printed_page])

    if not physical_indices:
        return None

    # Return the range
    return (min(physical_indices), max(physical_indices))


def parse_pdf_auto_with_split(
        pdf_path: str,
        output_dir: str,
        prefer_dynamic: bool = False,
        use_llm_toc: bool = True,
        ocr_fallback: bool = True,
        ocr_dpi: int = 300,
        ocr_lang: str = "eng",
):
    """
    Parse PDF and extract sections based on EXACT printed page numbers.

    Simple algorithm:
    1. Read TOC to get section page ranges (e.g., "36-40")
    2. Scan every PDF page to find printed page numbers
    3. Extract pages that match the range
    4. Ignore sections where pages don't exist
    """

    print("\n" + "=" * 80)
    print(f"Processing PDF: {pdf_path}")
    print("=" * 80 + "\n")

    # Step 1: Build the physical-to-printed page map (SOURCE OF TRUTH)
    physical_to_printed = build_physical_page_to_printed_map(pdf_path)

    # Step 2: Extract text for TOC parsing
    print("Extracting text from PDF for TOC detection...")

    if ocr_fallback and HAS_OCR:
        print("   OCR fallback: ENABLED")
        doc = fitz.open(pdf_path)
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = extract_text_with_ocr_fallback(page, ocr_enabled=True, dpi=ocr_dpi)
            pages.append(text)
        doc.close()
    else:
        if ocr_fallback and not HAS_OCR:
            print("   ⚠️  OCR requested but pytesseract not available")
        pages = extract_text_per_page(
            pdf_path,
            ocr_fallback=False,
            strip=True,
            use_pdfplumber=True,
        )

    print(f"Extracted {len(pages)} pages\n")

    # Step 3: Parse TOC
    print("Parsing Table of Contents...")
    parse_result = parse_pdf_auto(
        pdf_path,
        prefer_dynamic=prefer_dynamic,
        use_llm_toc=use_llm_toc,
        ocr_fallback=ocr_fallback,
        ocr_dpi=ocr_dpi,
        ocr_lang=ocr_lang,
    )

    if isinstance(parse_result, tuple):
        config, chunks = parse_result
        result = {"config": config, "chunks": chunks, "meta": {}}
    else:
        result = parse_result

    # Step 4: Flatten all sections
    def flatten_sections(sections_list):
        flat = []
        for section in sections_list:
            flat.append(section)
            if section.get("children"):
                flat.extend(flatten_sections(section["children"]))
        return flat

    all_sections = flatten_sections(result["config"].get("sections", []))

    print(f"Found {len(all_sections)} sections in TOC\n")

    # Step 5: Map each section to its page range
    print("=" * 80)
    print("Mapping TOC Sections to Printed Pages")
    print("=" * 80 + "\n")

    section_mappings = []

    for idx, section in enumerate(all_sections):
        section_id = section.get("id")
        section_num = section.get("number", "?")
        section_title = section.get("title", "")[:60]
        toc_start_page = section.get("page_hint")  # This is a PRINTED page number

        if not toc_start_page:
            # Section has no page number in TOC
            section_mappings.append({
                "section_id": section_id,
                "section_number": section_num,
                "section_title": section_title,
                "toc_page": None,
                "start_printed_page": None,
                "end_printed_page": None,
                "start_pdf_idx": None,
                "end_pdf_idx": None,
                "exists": False,
                "page_count": 0
            })
            print(f"   ⊘ [{section_num}] {section_title}")
            print(f"      No page number in TOC\n")
            continue

        # Find where this section ends (page before next section)
        toc_end_page = None

        for next_idx in range(idx + 1, len(all_sections)):
            next_section = all_sections[next_idx]
            next_page = next_section.get("page_hint")

            if next_page:
                # The end page of the current section is the page *before* the next section starts.
                toc_end_page = next_page - 1
                break

        # If no next section, use last available printed page (no change here)
        if toc_end_page is None:
            toc_end_page = max(physical_to_printed.values())

        # --- START FIX FOR END PAGE LOGIC ---
        # If the calculated end page is *before* the start page, it means:
        # 1. The next section starts on the same page or earlier (e.g., a sub-section is on the same page as a parent section's title)
        # 2. The start page is the end page (i.e., the section is just a title on a page)
        if toc_end_page is not None and toc_end_page < toc_start_page:
            toc_end_page = toc_start_page
        # --- END FIX FOR END PAGE LOGIC ---

        # Find physical pages for this range
        page_range = find_printed_page_range(
            toc_start_page,
            toc_end_page,
            physical_to_printed
        )

        if page_range:
            start_pdf_idx, end_pdf_idx = page_range

            # Get actual printed page numbers for this range
            actual_printed_pages = [
                physical_to_printed[i]
                for i in range(start_pdf_idx, end_pdf_idx + 1)
                if i in physical_to_printed
            ]

            actual_start = min(actual_printed_pages) if actual_printed_pages else toc_start_page
            actual_end = max(actual_printed_pages) if actual_printed_pages else toc_end_page

            section_mappings.append({
                "section_id": section_id,
                "section_number": section_num,
                "section_title": section_title,
                "toc_page": toc_start_page,
                "start_printed_page": actual_start,
                "end_printed_page": actual_end,
                "start_pdf_idx": start_pdf_idx,
                "end_pdf_idx": end_pdf_idx,
                "exists": True,
                "page_count": end_pdf_idx - start_pdf_idx + 1
            })

            print(f"   ✓ [{section_num}] {section_title}")
            print(f"      TOC: page {toc_start_page}")
            print(f"      Printed pages: {actual_start}-{actual_end}")
            print(f"      PDF indices: {start_pdf_idx}-{end_pdf_idx} ({end_pdf_idx - start_pdf_idx + 1} pages)\n")
        else:
            # Pages don't exist in PDF
            section_mappings.append({
                "section_id": section_id,
                "section_number": section_num,
                "section_title": section_title,
                "toc_page": toc_start_page,
                "start_printed_page": None,
                "end_printed_page": None,
                "start_pdf_idx": None,
                "end_pdf_idx": None,
                "exists": False,
                "page_count": 0
            })

            print(f"   ✗ [{section_num}] {section_title}")
            print(f"      TOC: page {toc_start_page}")
            print(f"      Pages {toc_start_page}-{toc_end_page} NOT FOUND in PDF\n")

    # Step 6: Extract PDF sections
    print("=" * 80)
    print("Extracting Section PDFs")
    print("=" * 80 + "\n")

    doc = fitz.open(pdf_path)
    section_files = {}

    for mapping in section_mappings:
        if not mapping["exists"]:
            section_files[mapping["section_id"]] = None
            continue

        section_id = mapping["section_id"]
        section_num = mapping["section_number"]
        section_title = mapping["section_title"]
        start_idx = mapping["start_pdf_idx"]
        end_idx = mapping["end_pdf_idx"]

        print(f"   [{section_num}] {section_title}")
        print(f"      Pages: {mapping['start_printed_page']}-{mapping['end_printed_page']}")
        print(f"      PDF indices: {start_idx}-{end_idx}")

        try:
            # Create new PDF with these pages
            output_pdf = fitz.open()

            # Insert each page, un-rotating if needed
            for page_idx in range(start_idx, end_idx + 1):
                source_page = doc[page_idx]

                # Check if page is rotated
                rotation = source_page.rotation

                if rotation != 0:
                    print(f"         📐 Page {page_idx} rotated {rotation}° - correcting")
                    # Insert and rotate back to 0°
                    output_pdf.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
                    new_page = output_pdf[-1]
                    new_page.set_rotation(0)
                else:
                    output_pdf.insert_pdf(doc, from_page=page_idx, to_page=page_idx)

            # Save PDF
            safe_num = str(section_num).replace(".", "_")
            safe_title = "".join(c for c in section_title if c.isalnum() or c in (' ', '_', '-'))[:50]
            safe_title = safe_title.replace(' ', '_').upper()

            filename = f"{safe_num}_{safe_title}.pdf"
            filepath = os.path.join(output_dir, filename)

            output_pdf.save(filepath)
            output_pdf.close()

            section_files[section_id] = filepath
            print(f"      ✓ Saved: {filename}\n")

        except Exception as e:
            print(f"      ✗ ERROR: {e}\n")
            section_files[section_id] = None
            import traceback
            traceback.print_exc()

    doc.close()

    # Step 7: Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✅ Extracted: {sum(1 for v in section_files.values() if v)} sections")
    print(f"❌ Missing: {sum(1 for v in section_files.values() if not v)} sections")
    print("=" * 80 + "\n")

    # Store results
    result["section_files"] = section_files
    result["section_mappings"] = section_mappings

    return result


# Keep OCR helper function
def extract_text_with_ocr_fallback(page: "fitz.Page", ocr_enabled: bool = True, dpi: int = 300) -> str:
    """Extract text with OCR fallback for scanned pages."""
    # Try standard extraction first
    try:
        text = page.get_text()
        if text and text.strip() and len(text.strip()) > 50:
            return text
    except Exception as e:
        print(f"   Warning: Standard text extraction failed: {e}")

    # OCR fallback
    if not ocr_enabled or not HAS_OCR:
        return text if text else ""

    try:
        print(f"   📷 Using OCR for page {page.number + 1}...")

        # Render page as image
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        # Run OCR
        ocr_text = pytesseract.image_to_string(img)

        if ocr_text and ocr_text.strip():
            print(f"   ✓ OCR extracted {len(ocr_text)} characters")
            return ocr_text
        else:
            print(f"   ⚠️  OCR found no text")
            return text if text else ""

    except Exception as e:
        print(f"   ⚠️  OCR failed: {e}")
        return text if text else ""


# Keep other helper functions for compatibility
def parse_pdf_auto_with_meta(*args, **kwargs):
    """Convenience wrapper."""
    cfg, chunks = parse_pdf_auto(*args, **kwargs)

    try:
        with fitz.open(args[0]) as doc:
            pages = doc.page_count
    except Exception:
        pages = None

    return {
        "meta": {
            "pages": pages,
            "sections": len(cfg.get("sections", [])),
            "chunks": len(chunks),
            "strategy": "auto",
        },
        "config": cfg,
        "chunks": chunks,
    }