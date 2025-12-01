from __future__ import annotations
import uuid
from typing import Dict, Any, Optional
from .db import db_cursor

def upsert_document(doc_id: str, title: str, month_label: str):
    with db_cursor() as cur:
        cur.execute("""
            INSERT INTO documents (id, title, month_label)
            VALUES (%s,%s,%s)
            ON CONFLICT (id) DO UPDATE SET
              title = EXCLUDED.title,
              month_label = EXCLUDED.month_label;
        """, (doc_id, title, month_label))

def delete_toc_for_document(doc_id: str):
    with db_cursor() as cur:
        # deleting root children cascades via FK; easiest is delete all by doc_id
        cur.execute("DELETE FROM toc_nodes WHERE doc_id=%s;", (doc_id,))

def insert_toc_node(doc_id: str, node: Dict[str, Any], parent_id: Optional[str]) -> str:
    nid = node.get("id") or str(uuid.uuid4())
    with db_cursor() as cur:
        cur.execute("""
            INSERT INTO toc_nodes (id, doc_id, parent_id, toc_code, title, level, sort_key)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (id) DO UPDATE SET
              parent_id = EXCLUDED.parent_id,
              toc_code  = EXCLUDED.toc_code,
              title     = EXCLUDED.title,
              level     = EXCLUDED.level,
              sort_key  = EXCLUDED.sort_key;
        """, (
            nid, doc_id, parent_id,
            node["toc_code"], node["title"], int(node["level"]),
            node.get("sort_key", "9999"),
        ))
    return nid

def insert_toc_tree(doc_id: str, root: Dict[str, Any]) -> str:
    """Recursively insert tree; returns root id."""
    def _recurse(n: Dict[str, Any], parent: Optional[str]) -> str:
        this_id = insert_toc_node(doc_id, n, parent)
        for idx, child in enumerate(n.get("children", []), start=1):
            # enforce a stable sort_key like '0001'
            child.setdefault("sort_key", f"{idx:04d}")
            _recurse(child, this_id)
        return this_id
    return _recurse(root, None)

def fetch_toc_tree(doc_id: str):
    """Return tree rows; we’ll re-assemble in memory."""
    with db_cursor() as cur:
        cur.execute("""
            SELECT id, parent_id, toc_code, title, level
            FROM toc_nodes
            WHERE doc_id=%s
            ORDER BY sort_key, toc_code;
        """, (doc_id,))
        rows = cur.fetchall()
    # Build index
    by_id = {r["id"]: r for r in rows}
    children = {r["id"]: [] for r in rows}
    root_id = None
    for r in rows:
        if r["parent_id"] is None:
            root_id = r["id"]
        else:
            children[r["parent_id"]].append(r)
    # recursive build
    def build(node_row):
        return {
            "id": node_row["id"],
            "toc_code": node_row["toc_code"],
            "title": node_row["title"],
            "level": node_row["level"],
            "children": [build(c) for c in children.get(node_row["id"], [])],
        }
    return build(by_id[root_id]) if root_id else None

def fetch_document(doc_id: str):
    with db_cursor() as cur:
        cur.execute("SELECT id, title, month_label FROM documents WHERE id=%s;", (doc_id,))
        row = cur.fetchone()
    return row