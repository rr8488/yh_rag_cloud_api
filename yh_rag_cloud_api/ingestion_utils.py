# yh_rag_cloud_api/ingestion_utils.py

import os
import shutil
import json
import fitz  # PyMuPDF
from pathlib import Path
from psycopg2.extensions import cursor

# Import utilities from other modules in the package
try:
    from .rag_utils import embed_text
    from .pdf_parser.parse_pdf_auto import parse_pdf_auto_with_split
    from .parsing_utils import (
        extract_text_with_tables,
        extract_text_from_docx,
        _create_text_chunks
    )
except ImportError:
    print("ERROR: Could not perform relative imports in ingestion_utils.py")
    # Define simple fallbacks for standalone analysis
    def embed_text(text: str): return [0.1] * 384
    def parse_pdf_auto_with_split(pdf_path, out_dir, **kwargs):
        print(f"[MOCK] Splitting PDF {pdf_path}")
        return {"section_mappings": [], "section_files": {}}
    def extract_text_with_tables(pdf_path): return f"[MOCK OCR TEXT FOR {pdf_path}]"
    def extract_text_from_docx(docx_path): return f"[MOCK DOCX TEXT FOR {docx_path}]"
    def _create_text_chunks(text, **kwargs): return text.split("\n\n")


def ingest_pdf_for_rag(
        cur: cursor,
        document_id: str,
        project_id: str,
        pdf_path: str
):
    """
    Orchestrates the full RAG ingestion for a single PDF.
    (MOVED from parsing_utils.py)
    """
    print(f"  Starting RAG ingestion for doc {document_id}...")

    # --- 1. Create RAG Document Entry ---
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = doc.page_count
    except Exception:
        total_pages = 0

    cur.execute("""
        INSERT INTO rag_documents (id, project_id, total_pages)
        VALUES (%s, %s, %s)
    """, (document_id, project_id, total_pages))

    # --- 2. Run your Advanced PDF Splitting ---
    doc_sections_dir = Path("sections") / document_id
    if doc_sections_dir.exists(): shutil.rmtree(doc_sections_dir)
    doc_sections_dir.mkdir(exist_ok=True, parents=True)

    print(f"  Calling parse_pdf_auto_with_split...")
    parse_result = parse_pdf_auto_with_split(
        pdf_path,
        str(doc_sections_dir),
        use_llm_toc=True,
        ocr_fallback=False  # OCR is handled by our extractor
    )

    section_mappings = parse_result.get("section_mappings", [])
    section_files = parse_result.get("section_files", {})

    # Update rag_documents with the TOC config
    cur.execute(
        "UPDATE rag_documents SET toc_config = %s WHERE id = %s",
        (json.dumps({
            "config": parse_result.get("config", {}),
            "section_mappings": section_mappings,
            "section_files": {k: str(v) if v else None for k, v in section_files.items()},
        }), document_id)
    )
    print(f"  Split PDF into {len(section_mappings)} sections.")

    if not section_mappings:
        print("  WARNING: No sections found by parser. Ingesting entire PDF as one chunk source.")
        full_content = extract_text_with_tables(pdf_path)
        if full_content and not full_content.startswith("Error"):
            node_id = f"{document_id}::FULL_PDF"
            try:
                cur.execute(
                    "INSERT INTO toc_nodes (id, document_id, toc_code, title, level, sort_key, section_id) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (node_id, document_id, "PDF", Path(pdf_path).name, 0, 0, "FULL_PDF"))
                cur.execute("INSERT INTO section_content (node_id, content) VALUES (%s, %s)", (node_id, full_content))
                small_chunks = _create_text_chunks(full_content, chunk_size=512, chunk_overlap=100)
                total_small_chunks = 0
                for chunk_text in small_chunks:
                    if not chunk_text or not chunk_text.strip(): continue
                    try:
                        embedding = embed_text(chunk_text)
                        cur.execute("INSERT INTO chunks (node_id, chunk_text, embedding) VALUES (%s, %s, %s::vector)",
                                    (node_id, chunk_text, embedding))
                        total_small_chunks += 1
                    except Exception as chunk_err:
                        print(f"    ERROR inserting fallback chunk for {node_id}: {chunk_err}")
                print(f"  Fallback ingestion: Created {total_small_chunks} chunks for the entire PDF.")
            except Exception as fallback_err:
                print(f"  ERROR during fallback ingestion for {document_id}: {fallback_err}")
        else:
            print(f"  ERROR: Could not extract text for fallback ingestion of {document_id}.")
        return  # Stop here if we used fallback

    # --- 3. Insert TOC Nodes ---
    nodes_inserted_count = 0
    for sort_key, mapping in enumerate(section_mappings):
        sec_id = mapping.get('section_id')
        if not sec_id: continue

        node_id = f"{document_id}::{sec_id}"
        level = len(str(mapping.get('section_number', '1')).split('.'))
        try:
            cur.execute("""
                INSERT INTO toc_nodes (id, document_id, toc_code, title, level, sort_key, section_id, start_pp, end_pp, start_pdf, end_pdf)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                node_id, document_id, mapping.get('section_number', ''),
                mapping.get('section_title', ''), level, sort_key, sec_id,
                mapping.get('start_printed_page'), mapping.get('end_printed_page'),
                mapping.get('start_pdf_idx'), mapping.get('end_pdf_idx')
            ))
            nodes_inserted_count += 1
        except Exception as node_err:
            print(f"    Error inserting node {node_id}: {node_err}")
    print(f"  Inserted {nodes_inserted_count} toc_nodes.")

    # --- 4. Extract, Chunk, and Embed ---
    sections_processed = 0
    total_small_chunks = 0
    for mapping in section_mappings:
        sec_id = mapping.get('section_id')
        if not sec_id or not mapping.get('exists', False): continue

        node_id = f"{document_id}::{sec_id}"
        pdf_path_str = str(section_files.get(sec_id)) if section_files.get(sec_id) else None

        if pdf_path_str and os.path.exists(pdf_path_str):
            full_section_content = extract_text_with_tables(pdf_path_str)
            sec_title = mapping.get('section_title', '')
            sec_toc = mapping.get('section_number', '')
            header = f"DOCUMENT_SECTION_HEADER: TOC {sec_toc} {sec_title}\n\n"
            full_section_content = header + full_section_content
        else:
            full_section_content = "Error: Section file path invalid or missing"

        # 4a. Insert "Large" content
        try:
            cur.execute(
                "INSERT INTO section_content (node_id, content) VALUES (%s, %s)",
                (node_id, full_section_content)
            )
        except Exception as e:
            print(f"    ERROR inserting full content for {node_id}: {e}")
            continue

        # 4b. Create and Insert "Small" Chunks
        if full_section_content.startswith("Error:"): continue

        small_chunks = _create_text_chunks(full_section_content, chunk_size=512, chunk_overlap=100)

        for chunk_text in small_chunks:
            if not chunk_text or not chunk_text.strip(): continue
            try:
                embedding = embed_text(chunk_text)
                cur.execute(
                    "INSERT INTO chunks (node_id, chunk_text, embedding) VALUES (%s, %s, %s::vector)",
                    (node_id, chunk_text, embedding)
                )
                total_small_chunks += 1
            except Exception as chunk_err:
                print(f"    ERROR inserting chunk for {node_id}: {chunk_err}")

        sections_processed += 1

    print(f"  Processed {sections_processed} sections, created {total_small_chunks} chunks.")


def ingest_docx_for_rag(
    cur: cursor,
    document_id: str,
    project_id: str,
    docx_path: str
):
    """
    Simplified RAG ingestion pipeline for DOCX files.
    (MOVED from parsing_utils.py)
    """
    print(f"  Starting RAG ingestion for DOCX {document_id}...")

    # --- 1. Create RAG Document Entry (No TOC config needed) ---
    cur.execute("""
        INSERT INTO rag_documents (id, project_id, total_pages)
        VALUES (%s, %s, %s)
    """, (document_id, project_id, 0)) # No pages concept

    # --- 2. Extract Full Text ---
    full_content = extract_text_from_docx(docx_path) # Use our DOCX extractor

    if not full_content or full_content.startswith("Error"):
        print(f"  ERROR: Could not extract text from DOCX {docx_path}. Aborting RAG ingestion.")
        return

    # --- 3. Create a SINGLE 'toc_node' representing the whole document ---
    node_id = f"{document_id}::FULL_DOC"
    try:
        cur.execute("""
            INSERT INTO toc_nodes (id, document_id, toc_code, title, level, sort_key, section_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            node_id, document_id, "DOC", Path(docx_path).name, 0, 0, "FULL_DOC"
        ))
    except Exception as node_err:
        print(f"  Error inserting 'FULL_DOC' node {node_id}: {node_err}")
        return # Cannot proceed without a node

    # --- 4. Insert the "Large" content (entire document) ---
    try:
        cur.execute(
            "INSERT INTO section_content (node_id, content) VALUES (%s, %s)",
            (node_id, full_content)
        )
    except Exception as e:
        print(f"  ERROR inserting full DOCX content for {node_id}: {e}")
        # We can potentially continue to chunking even if this fails

    # --- 5. Chunk, Embed, and Insert "Small" Chunks ---
    total_small_chunks = 0
    try:
        small_chunks = _create_text_chunks(full_content, chunk_size=512, chunk_overlap=100)
        print(f"    Created {len(small_chunks)} small chunks for DOCX.")

        for chunk_text in small_chunks:
            if not chunk_text or not chunk_text.strip(): continue
            try:
                embedding = embed_text(chunk_text) # Use centralized embedder
                cur.execute(
                    "INSERT INTO chunks (node_id, chunk_text, embedding) VALUES (%s, %s, %s::vector)",
                    (node_id, chunk_text, embedding)
                )
                total_small_chunks += 1
            except Exception as chunk_err:
                print(f"    ERROR inserting DOCX chunk for {node_id}: {chunk_err}")

        print(f"  Processed DOCX, created {total_small_chunks} chunks.")

    except Exception as chunking_err:
         print(f"  ERROR during DOCX chunking/embedding for {node_id}: {chunking_err}")
