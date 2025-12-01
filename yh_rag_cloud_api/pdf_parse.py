# app/pdf_parse.py
import re, uuid
from typing import Dict, List, Tuple
import fitz  # PyMuPDF

HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+(.+)$")  # e.g., "1.0 Title", "2.3.1 Subtitle"

def _level_from_code(code: str) -> int:
    return len(code.split("."))

def parse_pdf_to_toc_and_chunks(path: str) -> Tuple[Dict, List[Dict]]:
    """
    Returns (toc_root_dict, chunks_list)
      toc_root_dict: {id, toc_code, title, level, children:[...]}
      chunks_list:   [{node_id, page_from, page_to, content}, ...]
    """
    doc = fitz.open(path)

    # Try built-in PDF TOC first (if present)
    toc = doc.get_toc(simple=True)  # list of [level, title, page]
    if toc:
        # Build nodes from toc; synthesize toc_code per sibling order
        root = {"id": str(uuid.uuid4()), "toc_code": "ROOT", "title": "ROOT", "level": 1, "children": []}
        stacks = {1: root}
        counters = {}  # level -> counter
        flat_nodes = []

        for lvl, title, page in toc:
            counters[lvl] = counters.get(lvl, 0) + 1
            # build "1.0", "1.1" style codes
            code_parts = [str(counters[i]) for i in range(2, lvl + 1)]
            code = ".".join(code_parts) if code_parts else "1.0"
            node = {
                "id": str(uuid.uuid4()),
                "toc_code": code,
                "title": f"{code} {title}",
                "level": lvl,
                "page": page,
                "children": [],
            }
            parent = stacks.get(lvl - 1, root)
            parent["children"].append(node)
            stacks[lvl] = node
            # reset deeper counters
            keys_to_del = [k for k in counters if k > lvl]
            for k in keys_to_del:
                del counters[k]
            flat_nodes.append(node)

        # page ranges
        flat_nodes.sort(key=lambda n: n["page"])
        page_ranges = {}
        for i, n in enumerate(flat_nodes):
            start = n["page"]
            end = flat_nodes[i + 1]["page"] - 1 if i + 1 < len(flat_nodes) else doc.page_count
            page_ranges[n["id"]] = (start, end)

        # chunks
        chunks = []
        for n in flat_nodes:
            p_from, p_to = page_ranges[n["id"]]
            texts = [doc[p - 1].get_text("text") for p in range(p_from, p_to + 1)]
            chunks.append({
                "node_id": n["id"],
                "page_from": p_from,
                "page_to": p_to,
                "content": "\n".join(texts).strip(),
            })
        return root, chunks

    # Fallback: detect headings via regex
    lines_by_page = []
    for p in doc:
        txt = p.get_text("text")
        lines_by_page.append([ln.rstrip() for ln in txt.splitlines()])

    heads = []
    for pi, lines in enumerate(lines_by_page):
        for li, ln in enumerate(lines):
            m = HEADING_RE.match(ln)
            if m:
                code, title = m.group(1), m.group(2).strip()
                heads.append((pi, li, code, title))

    if not heads:
        root_id = str(uuid.uuid4())
        whole = "\n".join("\n".join(l) for l in lines_by_page)
        return (
            {"id": root_id, "toc_code": "ROOT", "title": "Document", "level": 1, "children": []},
            [{"node_id": root_id, "page_from": 1, "page_to": doc.page_count, "content": whole}],
        )

    heads.sort(key=lambda x: (x[0], x[1]))
    nodes = []
    for pi, li, code, title in heads:
        nid = str(uuid.uuid4())
        nodes.append({"id": nid, "toc_code": code, "title": f"{code} {title}", "level": _level_from_code(code), "page": pi + 1})

    root = {"id": str(uuid.uuid4()), "toc_code": "ROOT", "title": "ROOT", "level": 1, "children": []}
    stack = [root]
    for n in nodes:
        while stack and stack[-1]["level"] >= n["level"]:
            stack.pop()
        parent = stack[-1] if stack else root
        parent.setdefault("children", []).append({**n, "children": []})
        stack.append(parent["children"][-1])

    flat = []
    def walk(n):
        flat.append(n)
        for c in n.get("children", []):
            walk(c)
    walk(root)

    ordered = [n for n in flat if n["toc_code"] != "ROOT"]
    ordered.sort(key=lambda n: n["page"])
    page_ranges = {}
    for i, n in enumerate(ordered):
        start = n["page"]
        end = ordered[i + 1]["page"] - 1 if i + 1 < len(ordered) else doc.page_count
        page_ranges[n["id"]] = (start, end)

    chunks = []
    for n in ordered:
        p_from, p_to = page_ranges[n["id"]]
        texts = [doc[p - 1].get_text("text") for p in range(p_from, p_to + 1)]
        chunks.append({
            "node_id": n["id"],
            "page_from": p_from, "page_to": p_to,
            "content": "\n".join(texts).strip()
        })
    return root, chunks