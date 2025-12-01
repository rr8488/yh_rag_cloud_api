from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
import re
import requests


# ai_doc_nav_api/pdf_parser/llm_toc_extractor.py

# llm_toc_extractor.py
from dataclasses import dataclass
from typing import List
import re
import json

@dataclass
class TocEntry:
    section_id: str
    title: str
    level: int
    printed_page: int

# --- BEGIN REPLACEMENT: normalize LLM output to TocEntry list ---
SECTION_LINE_RE = re.compile(
    r'^\s*(?P<section>\d+(?:\.\d+)*)\s+(?P<title>.*?)\s+(?:\.{2,}\s*)?(?P<page>\d{1,4})\s*$'
)

def extract_toc_entries_from_llm(raw_llm_text: str) -> List[TocEntry]:
    """
    Accepts raw LLM text (or JSON) and emits a normalized list of TocEntry.
    Handles either a JSON array of objects or plain lines.
    """
    out: List[TocEntry] = []

    # 1) Try JSON first
    raw_llm_text = raw_llm_text.strip()
    if raw_llm_text.startswith("["):
        try:
            arr = json.loads(raw_llm_text)
            for item in arr:
                sec = str(item.get("section_id") or item.get("section") or "").strip()
                title = str(item.get("title") or "").strip()
                page = int(str(item.get("page") or item.get("printed_page") or "0").strip())
                if not sec or not title or page <= 0:
                    continue
                level = sec.count(".") + 1
                out.append(TocEntry(section_id=sec, title=title, level=level, printed_page=page))
            if out:
                return out
        except Exception:
            pass  # fall through to line parsing

    # 2) Parse free-form lines
    for raw_line in raw_llm_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = SECTION_LINE_RE.match(line)
        if not m:
            continue
        sec = m.group("section").strip()
        title = re.sub(r'\s+', ' ', m.group("title").strip())
        page = int(m.group("page"))
        level = sec.count(".") + 1
        out.append(TocEntry(section_id=sec, title=title, level=level, printed_page=page))

    # 3) Deduplicate
    dedup = []
    seen = set()
    for e in out:
        key = (e.section_id, e.printed_page)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(e)
    return dedup
# --- END REPLACEMENT ---

def extract_toc_with_llm(
        pages: List[str],
        api_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b"
) -> Dict[str, Any]:
    """
    Use local Ollama LLM to extract table of contents from PDF pages.
    """
    # Search for TOC in first 20 pages
    toc_pages = []
    toc_start_idx = None

    for i, page in enumerate(pages[:20]):
        page_upper = page.upper()

        # Look for TOC markers
        if toc_start_idx is None and ("TABLE OF CONTENTS" in page_upper or "CONTENTS" in page_upper[:500]):
            toc_start_idx = i
            toc_pages.append(page)
            print(f"📑 Found TOC marker on page {i + 1}")
            continue

        # If we've found TOC start, keep collecting pages that look like TOC
        if toc_start_idx is not None:
            # Check if this page has TOC-like content (numbered entries with page numbers)
            has_toc_entries = bool(re.search(r'^\s*\d+\.?\d*\s+[A-Z][A-Za-z\s]+.*\d{1,4}\s*$', page, re.MULTILINE))

            # Stop if we see actual content starting (like "1 PROJECT LAYOUT" followed by actual content)
            # But don't stop if we just see TOC entries
            is_content_page = bool(
                re.search(r'^\s*1\s+PROJECT\s+LAYOUT\s*$.*?\n\n[A-Z]', page, re.DOTALL | re.MULTILINE))

            if has_toc_entries and not is_content_page:
                toc_pages.append(page)
                if len(toc_pages) >= 4:  # Max 4 pages
                    break
            else:
                # Stop collecting TOC pages
                break

    # Fallback: use first 10 pages
    if not toc_pages:
        print("⚠️  No explicit TOC marker found, searching first 10 pages")
        toc_text = "\n\n=== PAGE BREAK ===\n\n".join(pages[:10])
    else:
        toc_text = "\n\n=== PAGE BREAK ===\n\n".join(toc_pages)
        print(f"📄 Using {len(toc_pages)} pages for TOC extraction")

    # Limit size
    if len(toc_text) > 35000:
        toc_text = toc_text[:35000] + "\n\n[... truncated ...]"

    prompt = f"""You are extracting a Table of Contents from a document that spans multiple pages.

    CRITICAL: Extract ALL numbered entries with their page numbers. Look for this pattern:
    - A number or code (examples: "1", "2.0", "3.1", "4.3", "7.5", "10", "15", "18")
    - A title in CAPITAL or Mixed case
    - Dots, spaces, or nothing
    - A page number (1-4 digits)

    Example entries:
    1 PROJECT LAYOUT ........................................................................................................................ 7
    4.3 LIST OF SUB-CONTRACTORS ...................................................................................................... 29
    7.5 SECURITY SYSTEM ELEMENT ..................................................................................................... 84
    10 QUALITY ASSURANCE & QUALITY CONTROL ........................................................................ 91
    15 WEATHER CHART ..................................................................................................................... 125
    18 PERFORMANCE S-CURVES ..................................................................................................... 131

    Return ONLY valid JSON (no explanations):
    {{
      "sections": [
        {{"number": "1", "title": "1 PROJECT LAYOUT", "page": 7, "level": 1}},
        {{"number": "4.3", "title": "4.3 LIST OF SUB-CONTRACTORS", "page": 29, "level": 2}},
        {{"number": "7.5", "title": "7.5 SECURITY SYSTEM ELEMENT", "page": 84, "level": 2}},
        {{"number": "10", "title": "10 QUALITY ASSURANCE & QUALITY CONTROL", "page": 91, "level": 1}},
        {{"number": "15", "title": "15 WEATHER CHART", "page": 125, "level": 1}},
        {{"number": "18", "title": "18 PERFORMANCE S-CURVES", "page": 131, "level": 1}}
      ]
    }}

    Rules:
    - **IMPORTANT**: Include the section number in the title (e.g., "1 PROJECT LAYOUT", "4.3 LIST OF SUB-CONTRACTORS")
    - level = 1 for main sections (single number: 1, 2, 3, 4, 10, 15, 18)
    - level = 2 for subsections (has dot: 1.0, 1.1, 4.3, 7.5)
    - level = 3 for sub-subsections (two dots: 1.1.1, 2.2.1)
    - Extract EVERY entry across ALL pages provided
    - IGNORE: headers, footers, "APPENDIX", "LIST OF APPENDICES"
    - Include entries numbered 1 through 18 or higher
    - If no entries found, return {{"sections": []}}

    Document (may span multiple pages):
    {toc_text}

    JSON:"""

    try:
        print(f"🤖 Calling Ollama at {api_url} with model {model}...")

        response = requests.post(
            f"{api_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.0,
                "options": {
                    "num_predict": 8000,  # Increased for more entries
                }
            },
            timeout=300
        )
        response.raise_for_status()

        result = response.json()
        llm_response = result.get("response", "")

        print(f"✅ LLM response received ({len(llm_response)} chars)")

        # Extract JSON
        json_match = re.search(r'\{\s*"sections"\s*:\s*\[[\s\S]*?\]\s*\}', llm_response, re.DOTALL)

        if json_match:
            toc_json = json_match.group(0)
        else:
            print("⚠️  No JSON structure found, attempting to salvage")
            toc_json = llm_response.strip()
            for marker in ["```json", "```"]:
                if toc_json.startswith(marker):
                    toc_json = toc_json[len(marker):]
                if toc_json.endswith(marker):
                    toc_json = toc_json[:-len(marker)]
            toc_json = toc_json.strip()

        # Parse JSON
        toc_data = json.loads(toc_json)
        sections = toc_data.get('sections', [])

        print(f"📋 Extracted {len(sections)} TOC entries")

        if sections:
            print(f"\n📑 Sample entries:")
            print(
                f"   First: [{sections[0].get('number', '')}] {sections[0].get('title', '')} (page {sections[0].get('page', '?')})")
            if len(sections) > 1:
                print(
                    f"   Last:  [{sections[-1].get('number', '')}] {sections[-1].get('title', '')} (page {sections[-1].get('page', '?')})")

            # Show section 4.3 if it exists
            section_43 = next((s for s in sections if s.get('number') == '4.3'), None)
            if section_43:
                print(f"   ✅ Found: [4.3] {section_43.get('title')} (page {section_43.get('page')})")
            else:
                print(f"   ⚠️  Section 4.3 (LIST OF SUB-CONTRACTORS) not found")

        return build_config_from_llm_toc(toc_data)

    except Exception as e:
        print(f"❌ LLM TOC extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return {"version": 1, "sections": []}


def build_config_from_llm_toc(toc_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert LLM-extracted TOC into config format with nested children.
    Maintains correct ordering of sections.
    """
    sections = []
    level_stack = [None, None, None, None]  # Stack to track parents at each level

    for entry in toc_data.get("sections", []):
        # Get the actual level from the number
        number = entry.get("number", "")
        if number:
            # Count dots to determine level
            # "1" = level 1, "1.0" or "1.1" = level 2, "1.1.1" = level 3
            dot_count = number.count(".")
            level = dot_count + 1
        else:
            level = entry.get("level", 1)

        # Generate section ID
        section_id = _generate_section_id(entry["title"], number)

        section_config = {
            "id": section_id,
            "title": entry["title"],
            "number": number,  # Keep number for debugging
            "pattern": _create_pattern_from_title(entry["title"], number),
            "page_hint": entry["page"],
            "children": []
        }

        # Add to appropriate parent based on ACTUAL level
        if level == 1:
            # Top-level section
            sections.append(section_config)
            level_stack[0] = section_config
            level_stack[1] = None
            level_stack[2] = None
        elif level == 2 and level_stack[0]:
            # Level 2: add to level 1 parent
            level_stack[0]["children"].append(section_config)
            level_stack[1] = section_config
            level_stack[2] = None
        elif level == 3 and level_stack[1]:
            # Level 3: add to level 2 parent
            level_stack[1]["children"].append(section_config)
            level_stack[2] = section_config
        else:
            # Fallback: if no valid parent, treat as top-level
            sections.append(section_config)
            level_stack[0] = section_config

    return {"version": 1, "sections": sections}


def _generate_section_id(title: str, number: Optional[str] = None) -> str:
    """Generate a clean section ID."""
    if number:
        prefix = number.replace(".", "_")
    else:
        prefix = ""

    # Clean title
    clean_title = re.sub(r'[^A-Z0-9]+', '_', title.upper())
    clean_title = clean_title.strip('_')[:40]

    if prefix:
        return f"{prefix}_{clean_title}"
    return clean_title


def _create_pattern_from_title(title: str, number: Optional[str] = None) -> str:
    """Create a regex pattern to find this section."""
    title_escaped = re.escape(title.upper())
    title_pattern = title_escaped.replace(r"\ ", r"\s+")

    if number:
        num_escaped = re.escape(number)
        return rf"(?mi)^\s*{num_escaped}\s+{title_pattern}\b"
    else:
        return rf"(?mi)^\s*{title_pattern}\b"