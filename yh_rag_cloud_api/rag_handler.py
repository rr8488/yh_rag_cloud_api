# yh_rag_cloud_api/rag_handler.py

import os
import re
import psycopg2
from psycopg2.extensions import cursor # For type hinting
from typing import List, Dict, Any, Tuple, Optional
import json

# Import helpers from our utils file
from .rag_utils import db_cursor, ask_gemini, query_llm


# ============================================
# AGENTIC RAG COMPONENTS
# ============================================

def _analyze_query_intent(question: str) -> Dict[str, Any]:
    """
    Uses an LLM to analyze the user's question and determine the likely
    source of information within the project structure.
    """
    print("  Analyzing query intent...")
    prompt = f"""
    Analyze the user's question about a grant project and determine the most likely document type(s) needed to answer it.

    Project Document Types:
    - GA: Grant Agreement (main terms, recipient, amount, duration)
    - S1: Schedule 1 (KPIs, milestones, disbursement details - often structured data)
    - S2: Schedule 2 (Implementation Plan - narrative/timeline)
    - S3: Schedule 3 (Budget details - often structured data)
    - S4: Schedule 4 (Reporting Requirements - narrative)
    - S5: Proposal (Project description, objectives, location, rationale - narrative)
    - P#: Progress Reports (P1, P2, etc. - narrative including successes, learnings, challenges)
    - D#: Supporting Documents for KPIs (D1, D2 for P1; D1, D2 for P2 etc. - evidence like attendance lists, photos, receipts)

    Consider the keywords and the type of information requested.

    Respond with ONLY a JSON object containing:
    - "query_type": Classify the query (e.g., "kpi_details", "budget_info", "proposal_summary", "agreement_terms", "progress_update", "kpi_evidence", "general_info").
    - "target_doc_types": A list of the most relevant doc_type prefixes (e.g., ["S1"], ["S3"], ["S5"], ["GA"], ["P"], ["D"], ["S5", "GA"]). Use "P" for any progress report, "D" for any supporting doc.
    - "structured_query_possible": boolean (true if the answer likely exists in structured S1/S3 data, false otherwise).

    Examples:
    Question: "What are the KPIs for Milestone 2?"
    {{
      "query_type": "kpi_details",
      "target_doc_types": ["S1"],
      "structured_query_possible": true
    }}

    Question: "How much is the total grant amount?"
    {{
      "query_type": "agreement_terms",
      "target_doc_types": ["GA"],
      "structured_query_possible": false
    }}

    Question: "Summarize the project's objectives."
    {{
      "query_type": "proposal_summary",
      "target_doc_types": ["S5"],
      "structured_query_possible": false
    }}

    Question: "What were the main challenges reported in P1?"
    {{
      "query_type": "progress_update",
      "target_doc_types": ["P1"],
      "structured_query_possible": false
    }}

    Question: "Show the attendance list submitted for the first progress report."
    {{
      "query_type": "kpi_evidence",
      "target_doc_types": ["P1_D"],
      "structured_query_possible": false
    }}

    Question: "What is the budget for training materials?"
     {{
      "query_type": "budget_info",
      "target_doc_types": ["S3"],
      "structured_query_possible": true
    }}

    User Question: "{question}"

    JSON Response:
    """
    try:
        response_text = ask_gemini(prompt) # Use Gemini for better analysis
        # Clean potential markdown fences
        json_str = response_text.strip().lstrip("```json").rstrip("```").strip()
        analysis = json.loads(json_str)
        print(f"  Query Analysis: Type='{analysis.get('query_type')}', Docs={analysis.get('target_doc_types')}, Structured={analysis.get('structured_query_possible')}")
        return analysis
    except Exception as e:
        print(f"  ERROR: Query analysis failed: {e}. Falling back to general RAG.")
        # Fallback to searching narrative docs if analysis fails
        return {
            "query_type": "general_info_fallback",
            "target_doc_types": ["S5", "GA", "P"], # Search Proposal, GA, and Progress Reports
            "structured_query_possible": False
        }

def _retrieve_from_sql_kpis(cur: cursor, project_id: str, question: str) -> Optional[List[Dict]]:
    """ Queries the kpis table for relevant entries. (Simple keyword matching for now) """
    print(f"  Retrieving from SQL KPIs for project {project_id}...")
    # Basic keyword extraction (can be improved)
    keywords = [word for word in re.findall(r'\w+', question.lower()) if len(word) > 3 and word not in ['what', 'kpi', 'list', 'show', 'describe']]
    if not keywords: return None

    # Use ILIKE for case-insensitive matching
    query_conditions = " OR ".join([f"kpi_description ILIKE %s" for _ in keywords])
    params = [str(project_id)] + [f"%{kw}%" for kw in keywords]

    try:
        cur.execute(f"""
            SELECT milestone_number, disbursement_month, disbursement_amount, kpi_description, status
            FROM kpis
            WHERE project_id = %s AND ({query_conditions})
            ORDER BY milestone_number
        """, params)
        results = cur.fetchall()
        print(f"  SQL KPI query found {len(results)} records.")
        return results if results else None
    except Exception as e:
        print(f"  ERROR retrieving from SQL KPIs: {e}")
        return None

def _retrieve_from_sql_budget(cur: cursor, project_id: str, question: str) -> Optional[List[Dict]]:
    """ Queries the budget_items table for relevant entries. """
    print(f"  Retrieving from SQL Budget for project {project_id}...")
    keywords = [word for word in re.findall(r'\w+', question.lower()) if len(word) > 3 and word not in ['what', 'budget', 'list', 'show', 'total', 'cost']]
    if not keywords: return None

    # Use ILIKE for case-insensitive matching
    query_conditions = " OR ".join([f"category ILIKE %s OR line_item ILIKE %s" for _ in keywords])
    params = [str(project_id)] + [f"%{kw}%" for kw in keywords for _ in (1, 2)] # Duplicate keywords for OR

    try:
        cur.execute(f"""
            SELECT category, line_item, budgeted_amount
            FROM budget_items
            WHERE project_id = %s AND ({query_conditions})
            ORDER BY category, line_item
        """, params)
        results = cur.fetchall()
        print(f"  SQL Budget query found {len(results)} records.")
        return results if results else None
    except Exception as e:
        print(f"  ERROR retrieving from SQL Budget: {e}")
        return None

def _retrieve_from_rag_targeted(
    cur: cursor,
    project_id: str,
    question: str,
    target_doc_types: List[str],
    top_k: int = 25 # Fetch more initially
) -> List[Dict]:
    """ Performs FTS search filtered by project_id and specific doc_types """
    print(f"  Retrieving from RAG (FTS) for project {project_id}, types {target_doc_types}...")

    # --- Keyword Extraction (using simple regex fallback for now) ---
    words = re.findall(r'\w+', question.lower())
    stop_words = set(['what', 'are', 'the', 'how', 'who', 'is', 'for', 'describe', 'list', 'show', 'tell', 'me', 'about'])
    keywords = [word for word in words if len(word) > 2 and word not in stop_words][:5] # Limit keywords
    if not keywords: keywords = ['project'] # Fallback keyword

    keyword_search_term = ' | '.join(keywords) # Use OR for broader search
    print(f"    FTS Keywords (OR): '{keyword_search_term}'")

    # --- Build doc_type filter SQL ---
    doc_type_conditions = []
    doc_type_params = []
    for dt in target_doc_types:
        if dt in ["P", "D"]: # Handle prefixes P (P1, P2...) or D (D1, D2...)
            doc_type_conditions.append("pd.doc_type LIKE %s")
            doc_type_params.append(f"{dt}%")
        elif dt.endswith("_D"): # Handle specific milestone supporting docs (e.g., "P1_D")
             doc_type_conditions.append("pd.doc_type LIKE %s")
             doc_type_params.append(f"{dt}%")
        else: # Handle exact matches (GA, S1, S5, P1, P2...)
            doc_type_conditions.append("pd.doc_type = %s")
            doc_type_params.append(dt)

    doc_type_filter_sql = " OR ".join(doc_type_conditions)
    if not doc_type_filter_sql:
        print("    ERROR: No valid target_doc_types provided for RAG.")
        return []

    # Combine params: FTS query, project_id, doc_type filters, limit
    params = [keyword_search_term, str(project_id)] + doc_type_params + [top_k]

    # --- FTS Query ---
    try:
        # Select necessary fields for context building and citations
        search_sql = f"""
            WITH fts AS (
                SELECT
                    n.id as node_id, n.document_id, pd.original_filename,
                    n.toc_code, n.title, n.start_pp, n.end_pp, n.sort_key,
                    s.content, -- Selecting full content here for ranking simplicity
                    pd.doc_type,
                    to_tsvector('english', n.title || ' ' || s.content) AS search_vector,
                    to_tsquery('english', %s) AS query
                FROM project_documents pd
                JOIN rag_documents rd ON pd.id = rd.id
                JOIN toc_nodes n ON rd.id = n.document_id
                JOIN section_content s ON n.id = s.node_id
                WHERE pd.project_id = %s AND ({doc_type_filter_sql}) -- Filter by project AND doc_types
            )
            SELECT
                   fts.node_id, fts.document_id, fts.original_filename, fts.doc_type,
                   fts.toc_code, fts.title, fts.start_pp, fts.end_pp, fts.sort_key,
                   ts_rank(search_vector, query) AS relevance
                   -- We don't select fts.content here to avoid large data transfer,
                   -- it will be fetched later in _build_context_and_citations_with_cursor
            FROM fts
            WHERE search_vector @@ query
            ORDER BY relevance DESC, sort_key ASC -- Sort by relevance across all docs/sections
            LIMIT %s
        """
        cur.execute(search_sql, params)
        results = cur.fetchall() # Fetches metadata + relevance score
        print(f"    FTS search found {len(results)} sections across specified doc types.")
        retrieved_info = [f"{r['original_filename']}({r['doc_type']}):{r['toc_code']}" for r in results[:5]]
        print(f"    Top 5 Retrieved Sections (Metadata): {retrieved_info}")
        return results # Returns list of dicts with node metadata and relevance
    except Exception as e:
        print(f"  ERROR retrieving from RAG: {e}")
        import traceback; traceback.print_exc() # Print full trace
        return []


# ============================================
# CONTEXT BUILDING AND FINAL PROMPT
# ============================================

def _build_context_and_citations_with_cursor(
    cur: cursor, # Added cursor parameter
    retrieved_nodes: List[Dict], # List of node metadata dicts from retrieval
    max_len: int = 80000 # Allow adjustable max length
    ) -> Tuple[str, List[Dict]]:
    """Builds the context string and citation list from retrieved nodes, fetching full content."""
    context_snippets = []
    citations = []
    total_len = 0

    try:
        # Extract node IDs from the metadata fetched during retrieval
        final_node_ids = [r['node_id'] for r in retrieved_nodes if r.get('node_id')]
        content_map = {}
        if final_node_ids:
            # Fetch full content using the passed cursor
            cur.execute("SELECT node_id, content FROM section_content WHERE node_id = ANY(%s)", (final_node_ids,))
            content_rows = cur.fetchall()
            content_map = {r['node_id']: r['content'] for r in content_rows}
            print(f"  DEBUG [Context Helper]: Fetched {len(content_map)} full section contents.")
        else:
             print("  DEBUG [Context Helper]: No node IDs to fetch content for.")
             return "", []


        print(f"\n--- Building Context from {len(retrieved_nodes)} sections (Max Len: {max_len}) ---")
        # Iterate through the original retrieved_nodes list to preserve relevance order
        for meta_row in retrieved_nodes:
            node_id = meta_row.get('node_id')
            if not node_id: continue # Skip if somehow node_id is missing

            full_content = content_map.get(node_id)
            if not full_content:
                print(f"  WARN [Context Helper]: Content not found for node_id {node_id}. Skipping.")
                continue

            # Build citation info from the metadata
            citation_info = {
                "document_id": str(meta_row.get("document_id", "")),
                "filename": meta_row.get("original_filename", "Unknown File"),
                "section_id": node_id.split("::")[-1],
                "toc_code": meta_row.get("toc_code", ""),
                "title": meta_row.get("title", "Unknown Section"),
                "pages": [meta_row.get("start_pp"), meta_row.get("end_pp")]
            }
            citations.append(citation_info)

            # Use filename and TOC code in context header for clarity
            header = f"--- Source: {citation_info['filename']} [{citation_info['toc_code']}] {citation_info['title']} ---"
            content = full_content or ''
            avail = max_len - total_len - len(header) - 4 # Account for header and newlines
            if avail <= 0 and total_len > 0:
                print("  Warn: Max context prompt length reached.");
                break # Stop adding snippets if context is full

            trunc = content[:max(0, avail)]
            snippet = f"{header}\n{trunc}\n\n"
            context_snippets.append(snippet)
            total_len += len(snippet)

        print(f"--- Context Built: {total_len} chars, {len(citations)} citations ---")
        return "".join(context_snippets), citations

    except Exception as e:
        print(f"Error [Context Helper] building context: {e}")
        import traceback; traceback.print_exc()
        return "", []


def _get_final_prompt(question: str, context: str) -> str:
    """Returns the standardized, robust, generalized prompt."""
    # This prompt comes from your original rag_handler.py
    return (
        "You are a project data analyst. Your sole purpose is to answer the user's question using ONLY the provided sources.\n\n"

        "**CRITICAL INSTRUCTION FOR READING TABLES:**\n"
        "When a user's question requires a specific quantitative value from a table (like 'days', 'percentage', 'cost', or 'count'), you MUST follow this procedure:\n"
        "1.  Identify the key data type the user is- asking for (e.g., 'days', '%', 'value', 'quantity').\n"
        "2.  Scan the table's column headers to find the header that *exactly matches* or *most closely corresponds* to that data type (e.g., 'Ahead / Delay by days' for 'days', or 'Variance %' for 'percentage').\n"
        "3.  Find the correct row for the item in question (e.g., 'Package 2A').\n"
        "4.  Report the value from that specific [Row] and [Matched Column].\n"
        "5.  To ensure accuracy, you should state which column the data was retrieved from. For example: *'The delay is -61 days (from the 'Ahead / Delay by days' column), and the variance is -6.64% (from the 'Variance %' column).'*\n"
        "6.  DO NOT report a value from one column (like 'Variance %') as the answer for a query about a different column (like 'days'). This is a critical error.\n\n"

        f"User Question: {question}\n\n" # Changed from "Question:" for clarity
        "--- START OF SOURCES ---\n"
        f"{context}\n"
        "--- END OF SOURCES ---\n\n"
        "Your task: Using only the sources provided, answer all parts of the user's question. "
        "Rigorously follow the **CRITICAL INSTRUCTION FOR READING TABLES** for all quantitative data extracted from tables within the sources."
        "If the answer cannot be found in the sources, state that clearly.\n\n"
        "Answer:"
    )

# ============================================
# MAIN AGENTIC RAG PIPELINE
# ============================================

def execute_rag_pipeline(
    project_id: str,
    question: str,
    top_k: int,      # This top_k is for the FINAL context size for LLM
    llm: str
) -> Dict[str, Any]:
    """
    Main Agentic RAG pipeline for project-scoped queries.
    1. Analyzes query intent.
    2. Routes to appropriate retriever(s) (SQL or targeted RAG across multiple docs).
    3. Builds context.
    4. Generates answer using LLM.
    """
    print(f"\n--- Executing Agentic RAG Pipeline for Project {project_id} ---")
    print(f"--- Question: {question} ---")

    # --- Step 1: Analyze Query Intent ---
    query_analysis = _analyze_query_intent(question)
    query_type = query_analysis.get("query_type", "general_info")
    target_doc_types = query_analysis.get("target_doc_types", ["S5", "GA", "P"]) # Default search scope
    structured_query_possible = query_analysis.get("structured_query_possible", False)

    retrieved_context_nodes_meta = [] # Holds metadata results from RAG
    structured_results = None         # Holds results from SQL

    try:
        with db_cursor() as cur:
            # --- Step 2: Route Retrieval ---

            # 2a. Try Structured Query if applicable
            if structured_query_possible:
                if query_type == "kpi_details":
                    structured_results = _retrieve_from_sql_kpis(cur, project_id, question)
                elif query_type == "budget_info":
                    structured_results = _retrieve_from_sql_budget(cur, project_id, question)

            # 2b. Perform Targeted RAG across relevant documents
            if not target_doc_types: # Safety fallback
                target_doc_types = ["S5", "GA", "P"]
                print("  WARNING: No target_doc_types from analysis, using default.")

            # Fetch metadata + relevance scores first
            retrieved_context_nodes_meta = _retrieve_from_rag_targeted(
                cur=cur,
                project_id=project_id,
                question=question,
                target_doc_types=target_doc_types,
                top_k=25 # Fetch more metadata initially, will be pruned later by relevance
            )

    except psycopg2.Error as db_err:
        print(f"DB error during Agentic RAG retrieval: {db_err}")
        return {"answer": f"Database error during retrieval: {db_err}", "citations": []}
    except Exception as e:
         print(f"Unexpected error during retrieval routing: {e}")
         import traceback; traceback.print_exc() # Print full trace for debugging
         return {"answer": f"Error during retrieval: {e}", "citations": []}


    # --- Step 3: Build Context ---
    context = ""
    citations = []
    max_context_len = 80000 # ~20k tokens for llama3.1 8b

    # 3a. Add Structured Results (if any)
    if structured_results:
        context += "--- Relevant Structured Data ---\n"
        if query_type == "kpi_details":
            context += "KPIs:\n"
            for kpi in structured_results[:5]: # Limit results
                context += f"- M{kpi.get('milestone_number', '?')}: {kpi.get('kpi_description', 'N/A')} (Amt: {kpi.get('disbursement_amount', 'N/A')}, Month: {kpi.get('disbursement_month', 'N/A')}, Status: {kpi.get('status', '?')})\n"
            if len(structured_results) > 5: context += "- ... (more KPIs found in database)\n"
        elif query_type == "budget_info":
            context += "Budget Items:\n"
            for item in structured_results[:10]: # Limit results
                 context += f"- {item.get('category', '')} / {item.get('line_item', 'N/A')}: RM {item.get('budgeted_amount', 'N/A')}\n"
            if len(structured_results) > 10: context += "- ... (more budget items found in database)\n"
        context += "\n"

    # 3b. Add RAG Results (fetch full content only for final top_k)
    if retrieved_context_nodes_meta:
        # Prune the metadata list *before* fetching full content
        nodes_to_fetch_content_for = retrieved_context_nodes_meta[:top_k]
        try:
             with db_cursor() as cur: # Need cursor again for content fetch
                 rag_context, rag_citations = _build_context_and_citations_with_cursor(
                     cur=cur, # Pass the cursor
                     retrieved_nodes=nodes_to_fetch_content_for, # Pass pruned list
                     max_len=max_context_len - len(context) # Adjust max length
                 )
                 if rag_context:
                     context += "--- Relevant Document Sections ---\n"
                     context += rag_context
                     citations.extend(rag_citations)
        except Exception as build_ctx_err:
             print(f"ERROR building RAG context: {build_ctx_err}")

    if not context.strip():
        print(f"  DEBUG: No context generated from SQL or RAG.")
        context = "No specific information found for this query in the project database or documents."

    # --- Step 4: Generate Answer ---
    prompt = _get_final_prompt(question, context)

    answer = "Error: LLM generation failed."
    try:
        llm_choice = (llm or "gemini").lower() # Default to gemini if not specified
        print(f"--- Sending ~{len(prompt)//4} tokens to LLM ({llm_choice}) with {len(citations)} sources) ---")

        if llm_choice == "ollama":
            answer = query_llm(prompt)
        elif llm_choice == "gemini":
            answer = ask_gemini(prompt)
        else:
            answer = f"LLM provider '{llm}' not configured. Defaulting to Gemini."
            answer = ask_gemini(prompt)
    except Exception as llm_err:
        print(f"Error during LLM generation: {llm_err}")
        answer = f"Error generating answer: {llm_err}"

    print(f"\n--- LLM Answer ---\n{answer}\n--- END LLM Answer ---")
    print(f"--- Returning answer, {len(citations)} citations ---")
    return {"answer": answer, "citations": citations}


# yh_rag_cloud_api/rag_handler.py

# Assuming the following helpers are available:
# from .app import db_cursor
# from sentence_transformers import SentenceTransformer (or import the model from app)

from .rag_utils import db_cursor  # Assuming db_cursor is properly defined in rag_utils
import traceback


def execute_global_search(
        question: str,
        top_k: int,
        # embedding_model,
        # embedding_model # (If needed for semantic search)
) -> list[dict]:
    """
    Performs a global search across titles and chunk content using ILIKE (Full-Text).
    This function is required to resolve the ImportError.
    """
    print(f"RAG Handler: Executing global search (ILIKE) for '{question}' (Top K: {top_k})")
    like_query = f'%{question}%'
    results = []

    try:
        with db_cursor() as cur:
            # SQL Query to search document/section titles and content in the database.
            # This query joins toc_nodes (for titles/metadata) and section_content (for the full text).
            sql = """
                SELECT DISTINCT ON (n.id)
                    n.id as node_id, n.title, n.doc_id, n.toc_code, n.level
                FROM toc_nodes n
                JOIN section_content s ON n.id = s.node_id
                WHERE
                    n.title ILIKE %s OR
                    s.content ILIKE %s
                LIMIT %s
            """

            cur.execute(sql, (like_query, like_query, top_k))
            raw_results = cur.fetchall()

            # Format results for the Flutter frontend
            results = [{
                "node_id": r["node_id"],
                "score": 1.0,  # Placeholder score for ILIKE search
                "title": r["title"],
                "doc_id": r["doc_id"],
                "toc_code": r["toc_code"]
            } for r in raw_results]

            # NOTE: For a complete global search, you might need a separate query
            # to search the `projects` table for project-level nodes.

        print(f"RAG Handler: Found {len(results)} global search results.")
        return results

    except Exception as e:
        print(f"!!! Error during global search DB operation: {e}")
        traceback.print_exc()
        raise e  # Re-raise the exception to be caught by the FastAPI endpoint