# ai_doc_nav_api/pdf_parser/regex_toc_extractor.py

import re
from typing import List, Dict, Any, Optional, Tuple


def extract_toc_with_regex(pages: List[str]) -> Dict[str, Any]:
    """
    Extract TOC using regex patterns - more reliable than LLM.
    """
    toc_pages = []
    toc_start_idx = None

    # Phase 1: Find the TOC start page
    for i, page in enumerate(pages[:20]):
        if "TABLE OF CONTENTS" in page.upper():
            toc_start_idx = i
            toc_pages.append((i, page))
            print(f"📑 Found 'TABLE OF CONTENTS' on page {i + 1}")
            break

    if toc_start_idx is None:
        print("⚠️  No 'TABLE OF CONTENTS' marker found")
        return {"version": 1, "sections": []}

    # Phase 2: Continue reading subsequent pages with TOC patterns
    toc_pattern = re.compile(
        r'^\s*\d+(?:\.\d+)*\s+[A-Z][A-Z\s&\(\)\/\-,]+.*?\d{1,4}\s*$',
        re.MULTILINE
    )

    for i in range(toc_start_idx + 1, min(toc_start_idx + 5, len(pages))):
        page = pages[i]
        first_lines = '\n'.join(page.split('\n')[:3])

        if "LIST OF APPENDICES" in first_lines.upper():
            print(f"📑 Found 'LIST OF APPENDICES' on page {i + 1} - stopping")
            break

        toc_matches = toc_pattern.findall(page)

        if len(toc_matches) >= 3:
            toc_pages.append((i, page))
            print(f"📑 Found TOC continuation on page {i + 1} ({len(toc_matches)} entries)")
        else:
            print(f"📑 No more TOC entries after page {i}")
            break

    print(f"📄 Processing {len(toc_pages)} TOC pages")

    # Combine all TOC pages
    combined_toc = "\n".join(page for _, page in toc_pages)

    # DEBUG: Print the last 1000 chars
    print(f"\n🔍 DEBUG: Last 1000 chars of TOC:")
    print(combined_toc[-1000:])
    print("=" * 60 + "\n")

    # IMPROVED: More flexible pattern that handles digits, parentheses, etc.
    pattern = re.compile(
        r'^\s*'
        r'(\d+(?:\.\d+)*)'  # Number
        r'\s+'
        r'([A-Z0-9][A-Z0-9\s&\(\)\/\-,\.]+?)'  # Title (added \. for periods)
        r'\s*'
        r'(?:\.{2,}|-{2,}|…+|\s{3,})'  # Dot leaders
        r'\s*'
        r'(\d{1,4})'  # Page number
        r'\s*$',
        re.MULTILINE
    )

    entries = []
    seen = set()

    # DEBUG: Show all matches
    all_matches = list(pattern.finditer(combined_toc))
    print(f"🔍 DEBUG: Found {len(all_matches)} total regex matches")
    if len(all_matches) > 0:
        print(f"   Last 5 matches:")
        for match in all_matches[-5:]:
            print(f"      {match.group(1)} {match.group(2).strip()[:40]} → page {match.group(3)}")
    print()

    for match in all_matches:
        number = match.group(1)
        title = match.group(2).strip()
        page_str = match.group(3)

        try:
            page_num = int(page_str)
        except ValueError:
            print(f"⚠️  Skipping: invalid page number '{page_str}' for section {number}")
            continue

        level = number.count('.') + 1

        title_upper = title.upper()
        if "APPENDIX" in title_upper or "APPENDICES" in title_upper:
            print(f"⚠️  Skipping: appendix section {number} {title[:30]}")
            continue

        key = (number, title.upper())
        if key in seen:
            print(f"⚠️  Skipping: duplicate section {number} {title[:30]}")
            continue
        seen.add(key)

        entries.append({
            "number": number,
            "title": f"{number} {title}",
            "page": page_num,
            "level": level
        })

    print(f"📋 Extracted {len(entries)} TOC entries after filtering")

    if entries:
        print(f"\n📑 Final entries:")
        print(f"   First: [{entries[0]['number']}] {entries[0]['title']} (page {entries[0]['page']})")
        print(f"   Last:  [{entries[-1]['number']}] {entries[-1]['title']} (page {entries[-1]['page']})")

        # Show sections 16 and 18
        section_16_entries = [e for e in entries if e['number'].startswith('16')]
        if section_16_entries:
            print(f"   📊 Chapter 16: {len(section_16_entries)} entries")
            for e in section_16_entries:
                print(f"      - {e['number']} {e['title'][:60]}")
        else:
            print(f"   ⚠️  Chapter 16: NO ENTRIES FOUND")

        section_18_entries = [e for e in entries if e['number'].startswith('18')]
        if section_18_entries:
            print(f"   📊 Chapter 18: {len(section_18_entries)} entries")
            for e in section_18_entries:
                print(f"      - {e['number']} {e['title'][:60]}")
        else:
            print(f"   ⚠️  Chapter 18: NO ENTRIES FOUND")

    return build_config_from_toc_entries(entries)


def _natural_sort_key(number: str) -> tuple:
    """
    Convert section number to tuple for natural sorting.
    Examples: "1" -> (1,), "4.3" -> (4, 3), "5.11" -> (5, 11)
    """
    try:
        return tuple(int(x) for x in number.split('.'))
    except:
        return (999,)  # Put invalid numbers at end


def build_config_from_toc_entries(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build hierarchical structure from flat entries.
    CRITICAL: Build hierarchy first, then sort.
    """
    # Step 1: Build hierarchy WITHOUT sorting
    sections = []
    level_stack = [None, None, None, None]

    for entry in entries:
        number = entry["number"]
        level = entry["level"]

        section_id = _generate_section_id(entry["title"], number)

        section_config = {
            "id": section_id,
            "title": entry["title"],
            "number": number,
            "pattern": _create_pattern_from_title(entry["title"], number),
            "page_hint": entry["page"],
            "children": []
        }

        # Add to parent based on level
        if level == 1:
            sections.append(section_config)
            level_stack[0] = section_config
            level_stack[1] = None
            level_stack[2] = None
        elif level == 2 and level_stack[0]:
            level_stack[0]["children"].append(section_config)
            level_stack[1] = section_config
            level_stack[2] = None
        elif level == 3 and level_stack[1]:
            level_stack[1]["children"].append(section_config)
            level_stack[2] = section_config
        else:
            # Orphaned section - treat as top-level
            sections.append(section_config)
            level_stack[0] = section_config

    # Recursive sorting function
    def sort_all_levels(section_list):
        """Sort sections at current level AND all children."""
        # Sort current level by section number
        section_list.sort(key=lambda s: _natural_sort_key(s.get("number", "999")))

        # Recursively sort children
        for section in section_list:
            if section.get("children"):
                sort_all_levels(section["children"])

    # Sort everything
    sort_all_levels(sections)

    # Debug output
    print("\n🔍 DEBUG: Final TOC structure:")
    for section in sections[-5:]:  # Show last 5 top-level
        print(f"   [{section['number']}] {section['title'][:50]}")
        for child in section.get("children", [])[:3]:  # Show first 3 children
            print(f"      └─ [{child['number']}] {child['title'][:50]}")

    return {"version": 1, "sections": sections}


def _generate_section_id(title: str, number: Optional[str] = None) -> str:
    """Generate a clean section ID."""
    if number:
        prefix = number.replace(".", "_")
    else:
        prefix = ""

    # Remove number prefix from title if present
    clean_title = re.sub(r'^\d+(?:\.\d+)*\s+', '', title)
    clean_title = re.sub(r'[^A-Z0-9]+', '_', clean_title.upper())
    clean_title = clean_title.strip('_')[:40]

    if prefix:
        return f"{prefix}_{clean_title}"
    return clean_title


def _create_pattern_from_title(title: str, number: Optional[str] = None) -> str:
    """Create a regex pattern to find this section in document body."""
    # Remove number from title if present
    title_only = re.sub(r'^\d+(?:\.\d+)*\s+', '', title)

    title_escaped = re.escape(title_only.upper())
    title_pattern = title_escaped.replace(r"\ ", r"\s+")

    if number:
        num_escaped = re.escape(number)
        return rf"(?mi)^\s*{num_escaped}\s+{title_pattern}\b"
    else:
        return rf"(?mi)^\s*{title_pattern}\b"


__all__ = ["extract_toc_with_regex", "build_config_from_toc_entries"]