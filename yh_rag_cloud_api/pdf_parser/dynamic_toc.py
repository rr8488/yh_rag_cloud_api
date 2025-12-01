# ai_doc_nav_api/pdf_parser/dynamic_toc.py
from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from yh_rag_cloud_api.pdf_parser.text_extract import printed_to_pdf_index

NUM_PAT = r"(?:\d+|[IVXLCDM]+|[A-Z])"  # 1, 2, III, A
DOT_NUM_SEQ = rf"{NUM_PAT}(?:\.{NUM_PAT})*"  # 1.2.3 or III.2
END_PAGE_NUM = r"(\d{1,4})\s*$"
DOT_LEADERS = r"\.{2,}|\·{2,}|—{2,}|-{3,}"

TOC_HDR_PAT = re.compile(r"\b(TABLE OF CONTENTS|CONTENTS)\b", re.I)

@dataclass
class TocEntry:
    raw: str
    title: str
    page: int
    level: int  # 1 or 2

# --------------------------
# Normalization helpers
# --------------------------

def _norm(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("LIASON", "LIAISON")  # light OCR healing
    return s

def _ucase(s: str) -> str:
    return _norm(s).upper()

def _fuzzy_eq(a: str, b: str, thresh: float = 0.86) -> bool:
    return SequenceMatcher(None, _ucase(a), _ucase(b)).ratio() >= thresh

# --------------------------
# 1) Detect & parse ToC
# --------------------------

def _find_candidate_toc_pages(pages: List[str], max_scan_pages: int = 12) -> List[int]:
    scores = []
    for i, tx in enumerate(pages[:max_scan_pages]):
        if not tx:
            continue
        lines = [l for l in tx.splitlines() if l.strip()]
        tocish = sum(bool(re.search(END_PAGE_NUM, l)) for l in lines)
        leaders = sum(bool(re.search(DOT_LEADERS, l)) for l in lines)
        hdr = 5 if TOC_HDR_PAT.search(tx) else 0
        score = tocish + leaders + hdr
        if score >= 5:  # heuristic threshold
            scores.append((i, score))
    return [i for i, _ in sorted(scores, key=lambda t: -t[1])]

TOC_LINE_PAT = re.compile(
    rf"""
    ^\s*(?P<num>{DOT_NUM_SEQ})?    # optional numbering
    \s*(?P<title>[^\.].*?)        # title (lazy)
    \s*(?:{DOT_LEADERS}\s*)?      # dot leaders
    {END_PAGE_NUM}                 # trailing page number
    """,
    re.X | re.I,
)


def parse_toc_from_pages(pages: List[str]) -> List[TocEntry]:
    candidate_idxs = _find_candidate_toc_pages(pages)
    if not candidate_idxs:
        return []

    # Merge consecutive candidate pages into spans
    spans: List[Tuple[int, int]] = []
    last = None
    for idx in candidate_idxs:
        if last is None or idx != last + 1:
            spans.append((idx, idx))
        else:
            s, _ = spans[-1]
            spans[-1] = (s, idx)
        last = idx

    # Use the highest-scoring span (first by our sort)
    start, end = spans[0]

    entries: List[TocEntry] = []
    for p in range(start, end + 1):
        for line in pages[p].splitlines():
            m = TOC_LINE_PAT.search(line)
            if not m:
                continue
            title = _norm(m.group("title"))

            # Safely extract page number
            page_str = m.group(1)  # captured END_PAGE_NUM
            if not page_str:
                continue  # Skip if no page number captured

            try:
                page = int(page_str)
            except (ValueError, TypeError):
                # Skip entries where page number isn't a valid integer
                continue

            num = m.group("num") or ""
            depth = num.count(".") + 1 if num else 1
            level = 1 if depth == 1 else 2
            entries.append(TocEntry(raw=line, title=title, page=page, level=level))

    # De-dup while preserving order
    dedup: List[TocEntry] = []
    seen = set()
    for e in entries:
        key = (_ucase(e.title), e.page, e.level)
        if key not in seen:
            seen.add(key)
            dedup.append(e)

    # Clip deeper levels to 2
    for e in dedup:
        e.level = 1 if e.level == 1 else 2
    return dedup

# --------------------------
# 2) Fallback: infer headings from body
# --------------------------

H_L1 = re.compile(rf"^\s*({NUM_PAT})\s+[^\d].+$")               # "5 SUMMARY ..."
H_L2 = re.compile(rf"^\s*({NUM_PAT}\.{NUM_PAT})\s+[^\d].+$")    # "5.1 DELAY ..."
APP_PAT = re.compile(r"^\s*APPENDIX\s+\w+", re.I)

def infer_headings_from_body(pages: List[str]) -> List[TocEntry]:
    entries: List[TocEntry] = []
    for i, tx in enumerate(pages):
        if not tx:
            continue
        for ln in tx.splitlines():
            line = ln.strip()
            if not line:
                continue
            if APP_PAT.match(line):
                entries.append(TocEntry(ln, _norm(line), i + 1, 1))
                continue
            if H_L2.match(line):
                entries.append(TocEntry(ln, _norm(line), i + 1, 2))
            elif H_L1.match(line):
                entries.append(TocEntry(ln, _norm(line), i + 1, 1))

    # Prefer ALL‑CAPS or multi‑word titles
    filtered: List[TocEntry] = []
    for e in entries:
        t = e.title
        letters = [c for c in t if c.isalpha()]
        caps_ratio = (sum(1 for c in letters if c.isupper()) / max(1, len(letters))) if letters else 0
        if caps_ratio >= 0.6 or len(t.split()) >= 2:
            filtered.append(e)

    # Collapse near-duplicates within same page
    out: List[TocEntry] = []
    for e in filtered:
        if out and out[-1].page == e.page and _fuzzy_eq(out[-1].title, e.title):
            continue
        out.append(e)

    # Clip levels
    for e in out:
        if re.match(rf"^\s*{NUM_PAT}\.{NUM_PAT}\.", e.title):
            e.level = 2
        elif re.match(rf"^\s*{NUM_PAT}\.{NUM_PAT}\b", e.title):
            e.level = 2
        else:
            e.level = 1 if e.level == 1 else 2
    return out

# --------------------------
# 3) Build dynamic config (L1→L2 anchors)
# --------------------------

def _guess_id(title: str) -> str:
    t = _ucase(title)
    t = re.sub(r"[^A-Z0-9]+", "_", t).strip("_")
    return t[:48]

def _mk_pat(title: str) -> str:
    # Escape and allow whitespace/punct variations + common aliases
    t = _ucase(title)
    t = re.sub(r"\s+", r"\\s+", re.escape(t))
    t = t.replace("QUALITY\\s+ASSURANCE\\s*&\\s+QUALITY\\s+CONTROL",
                  "(?:QUALITY\\s+ASSURANCE\\s*&\\s+QUALITY\\s+CONTROL|QA\\s*&\\s*QC)")
    t = t.replace("LIAISON", "LIAI?SON")
    # Start‑of‑line with optional leading numbering (1 or 1.2)
    return rf"(?mi)^\s*(?:{DOT_NUM_SEQ}\s+)?{t}\b"

def build_config(entries: List[TocEntry]) -> Dict[str, Any]:
    groups: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for e in entries:
        if e.level == 1:
            current = {
                "id": _guess_id(e.title),
                "title": _ucase(e.title),
                "pattern": _mk_pat(e.title),
                "page_hint": e.page,
                "children": [],
            }
            groups.append(current)
        elif e.level == 2 and current:
            current["children"].append({
                "id": _guess_id(e.title),
                "title": _ucase(e.title),
                "pattern": _mk_pat(e.title),
                "page_hint": e.page,
            })

    # Move appendices group to the end if detected as a section
    app_idx = next((i for i, g in enumerate(groups)
                    if re.search(r"\bAPPENDIX|APPENDICES\b", g["title"], re.I)), None)
    if app_idx is not None and app_idx != len(groups) - 1:
        groups.append(groups.pop(app_idx))

    return {"version": 1, "sections": groups}

# --------------------------
# 4) Split per config
# --------------------------

def split_by_config(pages: List[str], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Given extracted pages and a built config (sections with regex patterns + page hints),
    find anchors (L1 + L2), then slice into page-bounded chunks.

    Returns a list of dicts:
      {
        "section_id": str,
        "section_title": str,
        "level": 1,
        "start_page": int,     # 1-based
        "end_page": int,       # 1-based inclusive
        "text": str,           # concatenated pages
        "children": [          # for L2 children within this L1 span
          {
            "section_id": str,
            "section_title": str,
            "level": 2,
            "start_page": int,
            "end_page": int,
            "text": str,
          }, ...
        ]
      }
    """
    anchors: List[Tuple[int, int, str, str, int]] = []  # (page, level, id, title, idx)
    flat: List[Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]] = []

    # Flatten L1/L2 nodes
    for sec in cfg.get("sections", []):
        flat.append((1, sec, None))
        for ch in sec.get("children", []):
            flat.append((2, ch, sec))

    # Locate each node's first occurrence by regex, biasing near its page_hint
    for idx, (lvl, node, _parent) in enumerate(flat):
        pat = re.compile(node["pattern"])  # patterns already (?mi)
        start_at = max(0, int(node.get("page_hint", 1)) - 2)
        found: Optional[int] = None

        # Fast pass near hint
        for p in range(start_at, len(pages)):
            if pat.search(pages[p] or ""):
                found = p + 1  # 1-based
                break

        # Full scan fallback
        if found is None:
            for p in range(len(pages)):
                if pat.search(pages[p] or ""):
                    found = p + 1
                    break

        if found:
            anchors.append((found, lvl, node["id"], node["title"], idx))

    # Order: page asc, then level (L1 before L2 on same page)
    anchors.sort(key=lambda x: (x[0], x[1]))

    chunks: List[Dict[str, Any]] = []
    i = 0
    while i < len(anchors):
        page, lvl, sid, title, _ = anchors[i]

        if lvl == 1:
            # Find where this L1 ends: next L1 page - 1, or end of doc
            j = i + 1
            end_page = len(pages)
            while j < len(anchors):
                if anchors[j][1] == 1:  # next L1
                    end_page = anchors[j][0] - 1
                    break
                j += 1

            # Collect L2s within [page, end_page]
            children: List[Dict[str, Any]] = []
            k = i + 1
            while k < len(anchors) and anchors[k][0] <= end_page:
                if anchors[k][1] == 2:
                    c_start = anchors[k][0]
                    # L2 ends at next L2 (<= end_page) - 1, else end_page
                    m = k + 1
                    c_end = end_page
                    while m < len(anchors) and anchors[m][0] <= end_page:
                        if anchors[m][1] == 2:
                            c_end = anchors[m][0] - 1
                            break
                        m += 1

                    safe_start = max(1, c_start)
                    safe_end = min(len(pages), c_end)
                    children.append({
                        "section_id": anchors[k][2],
                        "section_title": anchors[k][3],
                        "level": 2,
                        "start_page": c_start,
                        "end_page": c_end,
                        "text": "\n".join(pages[safe_start - 1:safe_end]) if safe_start <= safe_end else "",
                    })
                k += 1

            # L1 chunk text
            safe_l1_start = max(1, page)
            safe_l1_end = min(len(pages), end_page)
            l1_text = "\n".join(pages[safe_l1_start - 1:safe_l1_end]) if safe_l1_start <= safe_l1_end else ""

            chunks.append({
                "section_id": sid,
                "section_title": title,
                "level": 1,
                "start_page": page,
                "end_page": end_page,
                "text": l1_text,
                "children": children,
            })

            # Jump to next L1 (j already positioned)
            i = j
        else:
            # Skip stray L2 that isn't inside a detected L1 span
            i += 1

    return chunks

## --- public API --------------------------------------------------------------

def build_dynamic_config(pages: List[str]) -> Dict[str, Any]:
    """
    Try ToC-based config; if none is found, fall back to headings inferred
    from the body (H1/H2). Returns a config with sections->children and
    useful regex patterns + page hints.
    """
    entries = parse_toc_from_pages(pages)
    if not entries:
        entries = infer_headings_from_body(pages)
    if not entries:
        # Last resort: single bucket of the whole doc
        return {
            "version": 1,
            "sections": [{
                "id": "FULL_DOCUMENT",
                "title": "FULL DOCUMENT",
                "pattern": r"(?s)^",   # match from the start of the doc
                "page_hint": 1,
                "children": []
            }]
        }
    return build_config(entries)


def parse_pdf_to_chunks(pages: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    High-level helper: (1) build dynamic config, (2) split by that config.
    Returns (config, chunks).
    """
    cfg = build_dynamic_config(pages)
    chunks = split_by_config(pages, cfg)
    return cfg, chunks


# ai_doc_nav_api/pdf_parser/dynamic_toc.py

def split_by_llm_sections(pages: List[str], toc_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split document pages into sections based on TOC page hints.
    TRUST THE TOC PAGE NUMBERS - they are authoritative.
    """
    sections_config = toc_config.get("sections", [])
    if not sections_config:
        return []

    # Flatten sections while preserving hierarchy
    flat_sections = []

    def flatten(section_list, parent_idx=None, current_level=1):
        for section in section_list:
            idx = len(flat_sections)
            flat_sections.append({
                "config": section,
                "idx": idx,
                "parent_idx": parent_idx,
                "level": current_level,
                "children_indices": []
            })
            if section.get("children"):
                child_start = len(flat_sections)
                flatten(section["children"], parent_idx=idx, current_level=current_level + 1)
                child_end = len(flat_sections)
                flat_sections[idx]["children_indices"] = list(range(child_start, child_end))

    flatten(sections_config)

    print(f"📄 Processing {len(flat_sections)} sections across {len(pages)} pages")

    result = []

    for i, section_info in enumerate(flat_sections):
        config = section_info["config"]

        page_hint = config.get("page_hint")
        if page_hint is None:
            print(f"⚠️  Section '{config.get('title')}' has no page_hint, skipping")
            continue

        # TRUST THE TOC: Start at the exact page listed
        # PDF pages are 1-indexed, but our array is 0-indexed
        start_page = page_hint - 1  # TOC says page 29 → array index 28

        # Determine end page: look for next sibling or parent's next sibling
        end_page = len(pages) - 1  # Default: end of document

        # Find the next section at the same or higher level
        for j in range(i + 1, len(flat_sections)):
            next_section = flat_sections[j]
            next_page_hint = next_section["config"].get("page_hint")

            if next_page_hint is None:
                continue

            # If next section is sibling or uncle (same/higher level), stop here
            if (next_section["parent_idx"] == section_info["parent_idx"] or
                    next_section["level"] <= section_info["level"]):
                # End at the page BEFORE the next section starts
                end_page = next_page_hint - 2  # TOC says next is page 34 → end at 33 (index 32)
                break

        # Ensure valid range
        start_page = max(0, min(start_page, len(pages) - 1))
        end_page = max(start_page, min(end_page, len(pages) - 1))

        # Extract text from the page range
        section_text = "\n\n".join(pages[start_page:end_page + 1])

        # Debug for specific sections
        section_number = config.get("number", "")
        if section_number in ["4.1", "4.3", "5.6", "5.7", "16.0", "18.0"]:
            print(f"\n🔍 DEBUG [{section_number}] {config.get('title')[:50]}")
            print(f"   TOC says page: {page_hint}")
            print(f"   Using pages: {start_page + 1} to {end_page + 1} (array: {start_page}-{end_page})")
            print(f"   Text preview: {section_text[:200]}")

        section_data = {
            "section_id": config["id"],
            "title": config["title"],
            "level": section_info["level"],
            "start_page": start_page + 1,  # Convert back to 1-indexed for display
            "end_page": end_page + 1,
            "text": section_text.strip(),
            "children": []
        }

        result.append(section_data)

    # Rebuild hierarchy
    hierarchical_result = []

    for i, section_info in enumerate(flat_sections):
        if i >= len(result):
            continue

        section = result[i]
        if section_info["parent_idx"] is None:
            hierarchical_result.append(section)
        else:
            parent_idx = section_info["parent_idx"]
            if parent_idx < len(result):
                parent = result[parent_idx]
                parent["children"].append(section)

    print(f"✅ Created {len(hierarchical_result)} top-level sections with hierarchy")
    return hierarchical_result

def compute_printed_spans(toc: List["TocEntry"], last_printed_page: int) -> List[dict]:
    """
    For each TOC entry, compute [start_pp, end_pp] in printed pages (inclusive).
    End = next.start_pp - 1, clamped at >= start_pp. No interpolation.
    """
    spans = []
    for i, e in enumerate(toc):
        start_pp = int(e.printed_page)
        if i + 1 < len(toc):
            next_pp = int(toc[i+1].printed_page)
            end_pp = max(start_pp, next_pp - 1)
        else:
            end_pp = max(start_pp, int(last_printed_page))
        spans.append({
            "section_id": e.section_id,
            "title": e.title,
            "level": e.level,
            "start_pp": start_pp,
            "end_pp": end_pp,
        })
    return spans

def map_spans_to_pdf(spans: List[dict], pp_to_pdf: Dict[int,int]) -> List[dict]:
    """
    Map printed spans to pdf indices. Missing mappings remain None (no interpolation).
    Clamp reversed pairs by making end_pdf = start_pdf if both exist and end < start.
    """
    mapped = []
    for s in spans:
        start_pdf = printed_to_pdf_index(pp_to_pdf, s["start_pp"])
        end_pdf   = printed_to_pdf_index(pp_to_pdf, s["end_pp"])

        # If both exist but backwards, clamp to single page
        if start_pdf is not None and end_pdf is not None and end_pdf < start_pdf:
            end_pdf = start_pdf

        node = {**s, "start_pdf": start_pdf, "end_pdf": end_pdf}
        # Add a convenience flag your Flutter UI / logger can read
        node["resolvable"] = (start_pdf is not None and end_pdf is not None)
        mapped.append(node)
    return mapped


__all__ = ["build_dynamic_config", "split_by_config", "parse_pdf_to_chunks"]