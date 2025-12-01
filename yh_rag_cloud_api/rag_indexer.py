# yh_rag_cloud_api/rag_indexer.py
import uuid
import json
from .db_schema import db_cursor
from .rag_utils import embed_text


def index_document_for_rag(project_id: str, doc_id: str, filename: str, text_content: str):
    """
    Chunks text, generates embeddings, and saves to the DB 'chunks' table.
    """
    print(f"⚡ Indexing document for RAG: {filename}")

    # 1. Basic Chunking (Split by paragraphs or character limit)
    # A simple approach: split by double newline, then aggregate to ~1000 chars
    raw_chunks = text_content.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in raw_chunks:
        if len(current_chunk) + len(para) < 1000:
            current_chunk += "\n\n" + para
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    if not chunks:
        print("⚠️ No text content to index.")
        return

    try:
        with db_cursor() as cur:
            # 2. Create Entry in rag_documents (linked to project_documents)
            # Check if exists first to avoid duplicates
            cur.execute("SELECT 1 FROM rag_documents WHERE id = %s", (doc_id,))
            if not cur.fetchone():
                cur.execute("""
                    INSERT INTO rag_documents (id, project_id, total_pages, toc_config)
                    VALUES (%s, %s, 1, '{}')
                """, (doc_id, project_id))

            # 3. Create a Root TOC Node (Simplified for flat documents)
            node_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO toc_nodes (id, document_id, title, level)
                VALUES (%s, %s, %s, 1)
            """, (node_id, doc_id, filename))

            # 4. Embed and Insert Chunks
            print(f"   - Processing {len(chunks)} chunks...")
            for chunk_text in chunks:
                # Generate Vector Embedding (using your rag_utils)
                embedding = embed_text(chunk_text)

                # Insert into DB
                cur.execute("""
                    INSERT INTO chunks (node_id, chunk_text, embedding)
                    VALUES (%s, %s, %s)
                """, (node_id, chunk_text, embedding))

        print(f"✅ Successfully indexed {len(chunks)} chunks for {filename}")

    except Exception as e:
        print(f"❌ Error indexing document: {e}")
        # Don't raise here, we don't want to fail the file upload if RAG fails