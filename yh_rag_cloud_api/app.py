# yh_rag_cloud_api/app.py

# --- CRITICAL: Load environment variables FIRST ---
from dotenv import load_dotenv

load_dotenv()

import os
import shutil
import json
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
import psycopg2
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# --- PROJECT IMPORTS ---
from .db_schema import initialize_database, db_cursor
from .rag_utils import (
    ask_gemini,
    initialize_cloud_clients,
    embed_text,
    classify_intent,
    identify_relevant_metadata_keys
)
from .rag_handler import execute_rag_pipeline, execute_global_search
from .models import GlobalSearchReq
from .rag_indexer import index_document_for_rag

# --- UTILS ---
from .parsing_utils import (
    extract_text_with_tables,
    extract_text_from_docx,
    extract_text_with_document_ai,
    excel_to_csv,
)
from .parsers.docx_utils import extract_text_from_docx_bytes

# --- PARSERS ---
from .parsers.grant_agreement_parser_production import extract_key_grant_fields
from .parsers.kpis_parser_production import parse_milestones, parse_currency_to_float
from .parsers.budget_parser_production import parse_budget_production
from .parsers.proposal_parser_production import parse_proposal_cloud
from .parsers.progress_report_parser import parse_progress_report

# ============================================
# CONFIGURATION
# ============================================
UPLOAD_DIR = Path("uploads")
SECTIONS_DIR = Path("sections")
TEMP_DIR = UPLOAD_DIR / "temp"

app = FastAPI(title="YH GrantNav Cloud API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# HELPER FUNCTIONS
# ============================================

def _ensure_project_exists(cur, project_id: str):
    """
    Checks if a project exists in SQL. If not, creates a skeleton record.
    """
    cur.execute("SELECT 1 FROM projects WHERE id = %s", (project_id,))
    if not cur.fetchone():
        print(f"  [DB] Project {project_id} missing in SQL. Creating skeleton record...")
        cur.execute("""
            INSERT INTO projects (id, ref_no, name, created_at) 
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        """, (project_id, project_id, f"New Project {project_id}"))


def _update_project_metadata(cur, project_id: str, new_data: Dict[str, Any]):
    """
    Merges new key-value pairs into the existing JSONB metadata column.
    It flattens the input dict slightly to ensure top-level accessibility.
    """
    if not new_data:
        return

    # Clean data: Remove None values to keep JSON clean
    clean_data = {k: v for k, v in new_data.items() if v is not None}

    if not clean_data:
        return

    # Postgres specific: metadata || %s merges the new keys into existing jsonb
    # COALESCE ensures we don't try to concat with NULL if it's the first time
    cur.execute("""
        UPDATE projects 
        SET metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb 
        WHERE id = %s
    """, (json.dumps(clean_data), project_id))
    print(f"  [DB] Updated metadata for project {project_id} with {len(clean_data)} keys.")


# ============================================
# STARTUP
# ============================================
@app.on_event("startup")
def _startup():
    print("📢 API Starting up...")
    try:
        initialize_cloud_clients()
    except Exception as e:
        print(f"⚠️ ERROR initializing cloud clients: {e}")

    try:
        print("📢 Attempting DB Initialization...")
        initialize_database()
        print("✅ Database schema initialized successfully.")
    except Exception as e:
        print(f"❌ CRITICAL DATABASE ERROR: {e}")
        print("⚠️ API will start, but DB endpoints will fail until configuration is fixed.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# PYDANTIC MODELS
# ============================================

class ProjectCreateReq(BaseModel):
    id: Optional[str] = None
    ref_no: Optional[str] = None
    name: str
    description: Optional[str] = None
    org_id: str
    grant_id: str
    impact_area_id: str
    grant_amount: Optional[float] = 0.0
    duration: Optional[str] = None
    grant_recipient_name: Optional[str] = None


class OrganizationCreateReq(BaseModel):
    id: Optional[str] = None
    ref_no: Optional[str] = None
    name: str
    description: Optional[str] = None


class GrantCreateReq(BaseModel):
    id: Optional[str] = None
    name: str
    type: str
    year: int


class ImpactAreaCreateReq(BaseModel):
    id: Optional[str] = None
    name: str


class RagReq(BaseModel):
    question: str
    top_k: int = 10
    llm: Optional[str] = "gemini"


class AdvancedRagReq(BaseModel):
    question: str
    top_k: int = 15
    filter_project_ids: List[str] = []
    filter_impact_areas: List[str] = []
    filter_doc_types: List[str] = []
    filter_years: List[int] = []


# ============================================
# CRUD ENDPOINTS (Restore these for Frontend)
# ============================================

# --- 1. ORGANIZATIONS ---

@app.get("/organizations")
def get_organizations():
    try:
        with db_cursor() as cur:
            cur.execute("""
                SELECT 
                    o.id, o.ref_no, o.name, o.description, 
                    COALESCE(array_agg(p.id) FILTER (WHERE p.id IS NOT NULL), '{}') as total_project_ids
                FROM organizations o
                LEFT JOIN projects p ON o.id = p.org_id
                GROUP BY o.id, o.ref_no, o.name, o.description
                ORDER BY o.name
            """)
            return {"organizations": [dict(row) for row in cur.fetchall()]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/organizations")
def create_organization(req: OrganizationCreateReq):
    org_id = req.id if req.id else str(uuid.uuid4())
    try:
        with db_cursor() as cur:
            cur.execute("""
                INSERT INTO organizations (id, ref_no, name, description)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    ref_no = EXCLUDED.ref_no,
                    name = EXCLUDED.name,
                    description = EXCLUDED.description
            """, (org_id, req.ref_no, req.name, req.description))
        return {"status": "success", "id": org_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 2. IMPACT AREAS & GRANTS ---

@app.get("/impact-areas")
def get_impact_areas():
    try:
        with db_cursor() as cur:
            cur.execute("SELECT id, name FROM impact_areas ORDER BY name")
            return {"impact_areas": [dict(row) for row in cur.fetchall()]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/impact-areas")
def create_impact_area(req: ImpactAreaCreateReq):
    ia_id = req.id if req.id else str(uuid.uuid4())
    try:
        with db_cursor() as cur:
            cur.execute("""
                INSERT INTO impact_areas (id, name) VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name
            """, (ia_id, req.name))
        return {"status": "success", "id": ia_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/grants")
def get_grants():
    try:
        with db_cursor() as cur:
            cur.execute("SELECT id, name, type, year FROM grants ORDER BY year DESC, name")
            return {"grants": [dict(row) for row in cur.fetchall()]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grants")
def create_grant(req: GrantCreateReq):
    grant_id = req.id if req.id else str(uuid.uuid4())
    try:
        with db_cursor() as cur:
            cur.execute("""
                INSERT INTO grants (id, name, type, year) VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, type=EXCLUDED.type, year=EXCLUDED.year
            """, (grant_id, req.name, req.type, req.year))
        return {"status": "success", "id": grant_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 3. PROJECTS ---

@app.get("/projects")
def get_projects():
    try:
        with db_cursor() as cur:
            cur.execute("""
                SELECT 
                    p.id, p.ref_no, p.name, p.description,
                    p.grant_amount, p.created_at,
                    p.org_id, o.name as grant_recipient_name,
                    p.grant_id, g.type as grant_type, g.year as grant_year,
                    p.impact_area_id, ia.name as impact_area,
                    p.duration_start_date, p.duration_end_date
                FROM projects p
                LEFT JOIN organizations o ON p.org_id = o.id
                LEFT JOIN grants g ON p.grant_id = g.id
                LEFT JOIN impact_areas ia ON p.impact_area_id = ia.id
                ORDER BY p.created_at DESC
            """)
            projects = []
            for row in cur.fetchall():
                p = dict(row)
                duration_str = ""
                if p['duration_start_date']:
                    duration_str = f"{p['duration_start_date']} to {p['duration_end_date'] or '?'}"

                projects.append({
                    "id": p['id'],
                    "ref_no": p['ref_no'],
                    "name": p['name'],
                    "description": p['description'],
                    "org_id": p['org_id'],
                    "grant_recipient_name": p['grant_recipient_name'],
                    "grant_id": p['grant_id'],
                    "impact_area_id": p['impact_area_id'],
                    "grant_value": float(p['grant_amount'] or 0),
                    "project_duration": duration_str,
                    "created_at": p['created_at'].isoformat() if p['created_at'] else None
                })
            return {"projects": projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects")
def create_project(req: ProjectCreateReq):
    proj_id = req.id if req.id else str(uuid.uuid4())
    try:
        with db_cursor() as cur:
            cur.execute("""
                INSERT INTO projects (
                    id, ref_no, name, description, 
                    org_id, grant_id, impact_area_id,
                    grant_amount, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (id) DO UPDATE SET
                    ref_no = EXCLUDED.ref_no,
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    org_id = EXCLUDED.org_id,
                    grant_id = EXCLUDED.grant_id,
                    impact_area_id = EXCLUDED.impact_area_id,
                    grant_amount = EXCLUDED.grant_amount
            """, (
                proj_id, req.ref_no, req.name, req.description,
                req.org_id, req.grant_id, req.impact_area_id,
                req.grant_amount
            ))
        return {"status": "success", "id": proj_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/project/{project_id}")
def delete_project(project_id: str):
    try:
        with db_cursor() as cur:
            cur.execute("DELETE FROM projects WHERE id = %s", (project_id,))

        # Clean disk
        proj_dir = UPLOAD_DIR / project_id
        if proj_dir.exists(): shutil.rmtree(proj_dir)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 4. DOCUMENTS (The missing one causing the 404!) ---

@app.get("/project/{project_id}/documents")
def get_project_documents(project_id: str):
    try:
        with db_cursor() as cur:
            cur.execute("""
                SELECT 
                    id, project_id, doc_type, name, file_name, 
                    file_extension, file_url, uploaded_at
                FROM project_documents 
                WHERE project_id = %s
                ORDER BY uploaded_at DESC
            """, (project_id,))

            docs = []
            for row in cur.fetchall():
                d = dict(row)
                d['uploaded_at'] = d['uploaded_at'].isoformat() if d['uploaded_at'] else None
                docs.append(d)
            return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/document/{document_id}")
def delete_document(document_id: str):
    try:
        with db_cursor() as cur:
            cur.execute("SELECT file_path FROM project_documents WHERE id = %s", (document_id,))
            row = cur.fetchone()
            if row and row['file_path']:
                fpath = Path(row['file_path'])
                if fpath.exists(): fpath.unlink()

            cur.execute("DELETE FROM project_documents WHERE id = %s", (document_id,))
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# TESTING ENDPOINTS (NO DB SIDE EFFECTS)
# ============================================

@app.post("/test/parse/grant-agreement")
async def test_parse_grant_agreement(file: UploadFile = File(...)):
    """TEST ONLY: Parses a Grant Agreement and returns the JSON."""
    temp_path = TEMP_DIR / f"test_{uuid.uuid4()}_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = ""
        if temp_path.suffix == ".pdf":
            text = extract_text_with_tables(str(temp_path))
        elif temp_path.suffix == ".docx":
            text = extract_text_from_docx(str(temp_path))

        data = extract_key_grant_fields(text)
        return {"status": "success", "parsed_data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists(): temp_path.unlink()


@app.post("/test/parse/milestones")
async def test_parse_milestones(file: UploadFile = File(...)):
    """TEST ONLY: Parses Schedule 1 (Milestones) and returns the JSON."""
    temp_path = TEMP_DIR / f"test_{uuid.uuid4()}_{file.filename}"
    csv_path = None
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        parser_input = str(temp_path)
        if temp_path.suffix in ['.xlsx', '.xls']:
            csv_path = Path(excel_to_csv(temp_path, sheet_name=0))
            parser_input = str(csv_path)

        data = parse_milestones(parser_input)
        return {"status": "success", "parsed_data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists(): temp_path.unlink()
        if csv_path and csv_path.exists(): csv_path.unlink()


@app.post("/test/parse/budget")
async def test_parse_budget(file: UploadFile = File(...)):
    """TEST ONLY: Parses Schedule 3 (Budget) and returns the JSON."""
    temp_path = TEMP_DIR / f"test_{uuid.uuid4()}_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        csv_content = ""
        if temp_path.suffix in ['.xlsx', '.xls']:
            csv_path = excel_to_csv(temp_path, sheet_name=0)
            with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
                csv_content = f.read()
            os.remove(csv_path)
        else:
            with open(temp_path, 'r', encoding='utf-8', errors='replace') as f:
                csv_content = f.read()

        data = parse_budget_production(csv_content)
        return {"status": "success", "parsed_data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists(): temp_path.unlink()


@app.post("/test/parse/proposal")
async def test_parse_proposal(file: UploadFile = File(...)):
    """TEST ONLY: Parses Schedule 5 (Proposal) and returns the JSON."""
    temp_path = TEMP_DIR / f"test_{uuid.uuid4()}_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if temp_path.suffix != ".docx":
            return {"status": "error", "message": "Only DOCX supported for proposals"}

        with open(temp_path, "rb") as f:
            content = f.read()
        text = extract_text_from_docx_bytes(content)

        data = parse_proposal_cloud(text)
        return {"status": "success", "parsed_data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists(): temp_path.unlink()


@app.post("/test/parse/progress-report")
async def test_parse_progress_report(file: UploadFile = File(...)):
    """TEST ONLY: Parses a Progress Report and returns the JSON."""
    temp_path = TEMP_DIR / f"test_{uuid.uuid4()}_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with open(temp_path, "rb") as f:
            content = f.read()
        text = extract_text_from_docx_bytes(content)

        data = parse_progress_report(text)
        return {"status": "success", "parsed_data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists(): temp_path.unlink()


# ============================================
# PRODUCTION PARSER ENDPOINTS (DB WRITES ENABLED)
# ============================================

@app.post("/project/{project_id}/parse/grant-agreement")
async def parse_grant_agreement_endpoint(project_id: str, file: UploadFile = File(...)):
    """Parses Grant Agreement -> updates Projects Table & Metadata."""
    temp_path = TEMP_DIR / f"{uuid.uuid4()}_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Register Document
        doc_id = str(uuid.uuid4())
        final_dir = UPLOAD_DIR / project_id
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / file.filename
        shutil.copy(temp_path, final_path)
        file_ext = Path(file.filename).suffix.lower()

        # 2. Extract Text
        text = ""
        if temp_path.suffix == ".pdf":
            text = extract_text_with_tables(str(temp_path))
        elif temp_path.suffix == ".docx":
            text = extract_text_from_docx(str(temp_path))

        # 3. Parse Data
        data = extract_key_grant_fields(text)

        # 4. Save to DB
        with db_cursor() as cur:
            _ensure_project_exists(cur, project_id)

            cur.execute("""
                INSERT INTO project_documents (
                    id, project_id, doc_type, file_path, 
                    name, file_name, file_extension, file_url, uploaded_at
                )
                VALUES (%s, %s, 'Grant Agreement', %s, %s, %s, %s, NULL, CURRENT_TIMESTAMP)
            """, (doc_id, project_id, str(final_path), file.filename, file.filename, file_ext))

            # Update Project Record (Standard Columns)
            cur.execute("""
                UPDATE projects SET 
                    grant_amount = COALESCE(%s, grant_amount),
                    duration_start_date = COALESCE(%s, duration_start_date),
                    duration_end_date = COALESCE(%s, duration_end_date),
                    grant_agreement_data = %s
                WHERE id = %s
            """, (
                data.get('grant_amount'),
                data.get('duration_start_date'),
                data.get('duration_end_date'),
                json.dumps(data),
                project_id
            ))

            # [NEW] Merge into Metadata
            _update_project_metadata(cur, project_id, data)

        # Index for RAG
        if text:
            index_document_for_rag(project_id, doc_id, file.filename, text)

        return {"status": "success", "data": data}
    except Exception as e:
        print(f"GA Parse Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists(): temp_path.unlink()


@app.post("/project/{project_id}/parse/milestones")
async def parse_milestones_endpoint(project_id: str, file: UploadFile = File(...)):
    """Parses Schedule 1 (KPIs) -> updates KPIs Table, Metadata & RAG Index."""
    temp_path = TEMP_DIR / f"{uuid.uuid4()}_{file.filename}"
    csv_path = None
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Register Doc
        doc_id = str(uuid.uuid4())
        final_dir = UPLOAD_DIR / project_id
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / file.filename
        shutil.copy(temp_path, final_path)
        file_ext = Path(file.filename).suffix.lower()

        # Convert to CSV if needed
        parser_input = str(temp_path)
        if temp_path.suffix in ['.xlsx', '.xls']:
            csv_path = Path(excel_to_csv(temp_path, sheet_name=0))
            parser_input = str(csv_path)

        # Parse Structured Data (for SQL)
        data = parse_milestones(parser_input)
        milestones = data.get('milestones', [])

        # RAG Text
        rag_text_content = ""
        try:
            with open(parser_input, "r", encoding="utf-8", errors="replace") as f:
                rag_text_content = f.read()
        except Exception as read_err:
            print(f"⚠️ Warning: Could not read content for RAG indexing: {read_err}")

        with db_cursor() as cur:
            _ensure_project_exists(cur, project_id)

            cur.execute("""
                INSERT INTO project_documents (
                    id, project_id, doc_type, file_path, 
                    name, file_name, file_extension, file_url, uploaded_at
                )
                VALUES (%s, %s, 'Schedule 1 - KPIs', %s, %s, %s, %s, NULL, CURRENT_TIMESTAMP)
            """, (doc_id, project_id, str(final_path), file.filename, file.filename, file_ext))

            # Overwrite KPIs (Structured Data)
            cur.execute("DELETE FROM kpis WHERE project_id = %s", (project_id,))
            for i, m in enumerate(milestones):
                amount = parse_currency_to_float(m.get('amount', '0'))
                desc = "; ".join([d['description'] for d in m.get('deliverables', [])])
                sods = "; ".join([d.get('sod', '') for d in m.get('deliverables', []) if d.get('sod')])
                cur.execute("""
                    INSERT INTO kpis (id, project_id, milestone_number, amount, description, sod, status)
                    VALUES (%s, %s, %s, %s, %s, %s, 'Pending')
                """, (str(uuid.uuid4()), project_id, i + 1, amount, desc, sods))

            # [NEW] Merge Metadata from Parser
            if 'metadata' in data:
                _update_project_metadata(cur, project_id, data['metadata'])

        if rag_text_content:
            index_document_for_rag(project_id, doc_id, file.filename, rag_text_content)

        return {"status": "success", "data": data}
    except Exception as e:
        print(f"Milestone Parse Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists(): temp_path.unlink()
        if csv_path and csv_path.exists(): csv_path.unlink()


@app.post("/project/{project_id}/parse/budget")
async def parse_budget_endpoint(project_id: str, file: UploadFile = File(...)):
    """Parses Schedule 3 (Budget) -> updates Budget Items Table, Metadata & RAG Index."""
    temp_path = TEMP_DIR / f"{uuid.uuid4()}_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Register Doc
        doc_id = str(uuid.uuid4())
        final_dir = UPLOAD_DIR / project_id
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / file.filename
        shutil.copy(temp_path, final_path)
        file_ext = Path(file.filename).suffix.lower()

        # Get CSV Content
        csv_content = ""
        if temp_path.suffix in ['.xlsx', '.xls']:
            csv_path = excel_to_csv(temp_path, sheet_name=0)
            with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
                csv_content = f.read()
            os.remove(csv_path)
        else:
            with open(temp_path, 'r', encoding='utf-8', errors='replace') as f:
                csv_content = f.read()

        # Parse with Gemini
        data = parse_budget_production(csv_content)
        items = data.get('budget_items', [])

        with db_cursor() as cur:
            _ensure_project_exists(cur, project_id)

            cur.execute("""
                INSERT INTO project_documents (
                    id, project_id, doc_type, file_path, 
                    name, file_name, file_extension, file_url, uploaded_at
                )
                VALUES (%s, %s, 'Schedule 3 - Budget', %s, %s, %s, %s, NULL, CURRENT_TIMESTAMP)
            """, (doc_id, project_id, str(final_path), file.filename, file.filename, file_ext))

            cur.execute("DELETE FROM budget_items WHERE project_id = %s", (project_id,))
            for item in items:
                cur.execute("""
                    INSERT INTO budget_items (id, project_id, category_l1, category_l2, description, requested_amount)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    str(uuid.uuid4()), project_id,
                    item.get('category_l1', 'Uncategorized'),
                    item.get('category_l2'),
                    item.get('description'),
                    item.get('requested_amount', 0)
                ))

            # [NEW] Merge Metadata from Parser
            if 'metadata' in data:
                _update_project_metadata(cur, project_id, data['metadata'])

        if csv_content:
            index_document_for_rag(project_id, doc_id, file.filename, csv_content)

        return {"status": "success", "data": data}
    except Exception as e:
        print(f"Budget Parse Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists(): temp_path.unlink()


@app.post("/project/{project_id}/parse/proposal")
async def parse_proposal_endpoint(project_id: str, file: UploadFile = File(...)):
    """Parses Schedule 5 (Proposal) -> updates Projects table (Locations/JSON), Metadata & RAG Index."""
    temp_path = TEMP_DIR / f"{uuid.uuid4()}_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Register Doc
        doc_id = str(uuid.uuid4())
        final_dir = UPLOAD_DIR / project_id
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / file.filename
        shutil.copy(temp_path, final_path)
        file_ext = Path(file.filename).suffix.lower()

        if temp_path.suffix != ".docx":
            return {"status": "error", "message": "Only DOCX supported for proposals"}

        with open(temp_path, "rb") as f:
            content = f.read()
        text = extract_text_from_docx_bytes(content)

        # Parse
        data = parse_proposal_cloud(text)

        with db_cursor() as cur:
            _ensure_project_exists(cur, project_id)

            cur.execute("""
                INSERT INTO project_documents (
                    id, project_id, doc_type, file_path, 
                    name, file_name, file_extension, file_url, uploaded_at
                )
                VALUES (%s, %s, 'Schedule 5 - Proposal', %s, %s, %s, %s, NULL, CURRENT_TIMESTAMP)
            """, (doc_id, project_id, str(final_path), file.filename, file.filename, file_ext))

            # Update Project with JSONB and Locations
            locations = data.get('project_location_list', [])
            cur.execute("""
                UPDATE projects SET
                    locations = %s,
                    proposal_data = %s
                WHERE id = %s
            """, (locations if locations else None, json.dumps(data), project_id))

            # [NEW] Merge into Metadata
            _update_project_metadata(cur, project_id, data)

        if text:
            index_document_for_rag(project_id, doc_id, file.filename, text)

        return {"status": "success", "data": data}
    except Exception as e:
        print(f"Proposal Parse Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists(): temp_path.unlink()


@app.post("/project/{project_id}/parse/progress-report")
async def parse_progress_report_endpoint(project_id: str, file: UploadFile = File(...)):
    """Parses Progress Report -> updates Tables, Metadata & RAG Index."""
    temp_path = TEMP_DIR / f"{uuid.uuid4()}_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Register Doc
        doc_id = str(uuid.uuid4())
        report_id = str(uuid.uuid4())
        final_dir = UPLOAD_DIR / project_id
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / file.filename
        shutil.copy(temp_path, final_path)
        file_ext = Path(file.filename).suffix.lower()

        # Extract Text
        with open(temp_path, "rb") as f:
            content = f.read()
        text_content = extract_text_from_docx_bytes(content)

        # Parse Structured Data
        data = parse_progress_report(text_content)

        with db_cursor() as cur:
            _ensure_project_exists(cur, project_id)

            cur.execute("""
                INSERT INTO project_documents (
                    id, project_id, doc_type, file_path, 
                    name, file_name, file_extension, file_url, uploaded_at
                )
                VALUES (%s, %s, 'progress_report', %s, %s, %s, %s, NULL, CURRENT_TIMESTAMP)
            """, (doc_id, project_id, str(final_path), file.filename, file.filename, file_ext))

            # Insert Report Header
            cur.execute("""
                INSERT INTO progress_reports (
                    id, project_id, project_document_id,
                    report_number, report_date, report_type,
                    funds_disbursed, funds_utilized, funds_unutilized,
                    executive_summary, challenges_summary, lessons_learned_summary,
                    full_data
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                report_id, project_id, doc_id,
                data.get('report_number'), data.get('report_date'), data.get('report_type'),
                data.get('funds_disbursed_to_date_myr'), data.get('funds_utilized_to_date_myr'),
                data.get('funds_unutilized_to_date_myr'),
                data.get('executive_summary'), data.get('challenges_summary'), data.get('lessons_learned_summary'),
                json.dumps(data)
            ))

            # Insert Deliverables
            for d in data.get('deliverables', []):
                cur.execute("""
                    INSERT INTO report_deliverables (id, progress_report_id, description, status, progress_update)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    str(uuid.uuid4()), report_id,
                    d.get('description'), d.get('status'), d.get('progress_update')
                ))

            # [NEW] Merge into Metadata
            meta_update = {
                "latest_challenges": data.get('challenges_summary'),
                "latest_lessons_learned": data.get('lessons_learned_summary'),
                "latest_report_date": data.get('report_date'),
                "latest_executive_summary": data.get('executive_summary')
            }
            _update_project_metadata(cur, project_id, meta_update)

        if text_content:
            index_document_for_rag(project_id, doc_id, file.filename, text_content)

        return {"status": "success", "data": data}
    except Exception as e:
        print(f"Progress Report Parse Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists(): temp_path.unlink()


# ============================================
# RAG ENDPOINTS (3-TIER LOGIC)
# ============================================

@app.post("/rag/advanced")
async def advanced_rag_endpoint(req: AdvancedRagReq):
    print(f"API: POST /rag/advanced | Q: {req.question}")

    try:
        # --- 1. AI ROUTER ---
        # First, check if this is Math/SQL, Metadata Lookup, or Vector Search
        intent = classify_intent(req.question)

        # Override: Force SQL for specific finance keywords to ensure math accuracy
        if any(x in req.question.lower() for x in ['utilization', 'spent', 'financial', 'budget left', 'how much']):
            intent = "SQL_ANALYSIS"

        print(f"  [Router] Classified as: {intent}")

        # =========================================================
        # TIER 1: SQL_ANALYSIS (Aggregations, Counts, Finance)
        # =========================================================
        if intent == "SQL_ANALYSIS":
            print("  [Mode] Executing SQL Aggregation.")

            sql_query = """
                SELECT 
                    p.id, p.name as project_name, 
                    p.grant_amount as total_grant_allocated,
                    o.name as recipient_org, ia.name as impact_area,
                    -- FINANCIAL DATA (From Latest Progress Report)
                    COALESCE(pr.report_date::text, 'N/A') as last_report_date,
                    COALESCE(pr.funds_utilized, 0) as funds_actually_spent,
                    COALESCE(pr.funds_unutilized, 0) as funds_remaining,
                    CASE 
                        WHEN p.grant_amount > 0 THEN ROUND((COALESCE(pr.funds_utilized, 0) / p.grant_amount * 100)::numeric, 2)
                        ELSE 0 
                    END as utilization_percentage
                FROM projects p
                LEFT JOIN organizations o ON p.org_id = o.id
                LEFT JOIN impact_areas ia ON p.impact_area_id = ia.id
                LEFT JOIN LATERAL (
                    SELECT * FROM progress_reports WHERE project_id = p.id ORDER BY report_date DESC LIMIT 1
                ) pr ON true
                WHERE 1=1
            """
            params = []
            if req.filter_project_ids:
                sql_query += " AND p.id = ANY(%s)"
                params.append(req.filter_project_ids)
            if req.filter_impact_areas:
                sql_query += " AND p.impact_area_id = ANY(%s)"
                params.append(req.filter_impact_areas)

            with db_cursor() as cur:
                cur.execute(sql_query, tuple(params))
                rows = cur.fetchall()

            projects_data = [dict(row) for row in rows]
            context_text = json.dumps(projects_data, indent=2, default=str)

            system_prompt = (
                "You are a Grant Data Analyst. "
                "I will provide a JSON dataset of projects including their latest financial utilization. "
                "Answer the user's question STRICTLY based on this data."
            )
            full_prompt = f"{system_prompt}\n\nUSER QUESTION: {req.question}\n\nDATASET:\n{context_text}"

            answer = ask_gemini(full_prompt)
            return {"answer": answer, "citations": ["SQL Database"], "mode": "sql_analysis"}

        # =========================================================
        # TIER 2: METADATA_LOOKUP (Specific Field Extraction)
        # =========================================================
        elif intent == "METADATA_LOOKUP":
            print("  [Mode] Executing Metadata Lookup.")

            # A. Get Available Keys (Sample from DB)
            # [FIX] Added alias 'meta_key' so we can access it by name in RealDictCursor
            keys_query = "SELECT DISTINCT jsonb_object_keys(metadata) AS meta_key FROM projects"
            params = []
            if req.filter_project_ids:
                keys_query += " WHERE id = ANY(%s)"
                params.append(req.filter_project_ids)
            else:
                keys_query += " LIMIT 200"  # Optimization

            available_keys = []
            with db_cursor() as cur:
                cur.execute(keys_query, tuple(params))
                # [FIX] Access by string key 'meta_key' instead of index 0
                available_keys = [r['meta_key'] for r in cur.fetchall()]

            # B. Ask LLM to Pick Relevant Keys
            target_keys = identify_relevant_metadata_keys(req.question, available_keys)
            print(f"  [Metadata] Target Keys: {target_keys}")

            if not target_keys:
                print("  [Metadata] No keys found. Falling back to Vector.")
                intent = "VECTOR_RAG"  # Fallback
            else:
                # C. Fetch Key Values
                select_parts = [f"metadata->>'{k}' as \"{k}\"" for k in target_keys]
                select_sql = ", ".join(select_parts)
                query = f"SELECT name, {select_sql} FROM projects WHERE 1=1"
                q_params = []

                if req.filter_project_ids:
                    query += " AND id = ANY(%s)"
                    q_params.append(req.filter_project_ids)
                query += " LIMIT 20"

                data_rows = []
                with db_cursor() as cur:
                    cur.execute(query, tuple(q_params))
                    data_rows = [dict(row) for row in cur.fetchall()]

                # D. Generate Answer
                context_text = json.dumps(data_rows, indent=2)
                full_prompt = (
                    f"You are a Project Data Assistant. Answer based on the METADATA provided.\n\n"
                    f"USER QUESTION: {req.question}\n\nMETADATA:\n{context_text}"
                )
                answer = ask_gemini(full_prompt)
                return {"answer": answer, "citations": ["Project Metadata"], "mode": "metadata_lookup"}

        # =========================================================
        # TIER 3: VECTOR_RAG (Narrative Search)
        # =========================================================
        # Note: 'intent' might be set here by Router OR by Fallback from Tier 2
        if intent == "VECTOR_RAG":
            print("  [Mode] Executing Semantic Vector Search.")

            query_embedding = embed_text(req.question)
            fetch_limit = min(req.top_k * 5, 150)

            base_query = """
                SELECT 
                    c.chunk_text, pd.name as doc_name, pd.file_name, p.name as project_name, 
                    1 - (c.embedding <=> %s::vector) as similarity
                FROM chunks c
                JOIN toc_nodes tn ON c.node_id = tn.id
                JOIN rag_documents rd ON tn.document_id = rd.id
                JOIN project_documents pd ON rd.id = pd.id
                JOIN projects p ON pd.project_id = p.id
                WHERE 1=1
            """
            params = [query_embedding]

            if req.filter_project_ids:
                base_query += " AND p.id = ANY(%s)"
                params.append(req.filter_project_ids)
            # Add other filters if needed...

            base_query += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
            params.append(query_embedding)
            params.append(fetch_limit)

            with db_cursor() as cur:
                cur.execute(base_query, tuple(params))
                raw_rows = cur.fetchall()

            # Reranking logic (simple top K for now)
            balanced_chunks = raw_rows[:req.top_k]

            if not balanced_chunks:
                return {"answer": "No relevant documents found.", "citations": []}

            context_text = "\n\n".join([
                f"SOURCE: {c['project_name']} - {c['file_name']}\nCONTENT: {c['chunk_text']}"
                for c in balanced_chunks
            ])

            full_prompt = (
                f"You are an AI assistant. Answer based ONLY on the context.\n\n"
                f"USER QUESTION: {req.question}\n\nCONTEXT:\n{context_text}"
            )
            answer = ask_gemini(full_prompt)

            unique_sources = list(set([f"{c['project_name']} - {c['file_name']}" for c in balanced_chunks]))
            return {"answer": answer, "citations": unique_sources, "mode": "vector_rag"}

    except Exception as e:
        print(f"Error in Advanced RAG: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/debug/reset-db")
def reset_database_schema():
    """
    WARNING: DELETES ALL DATA.
    Drops existing tables so they can be recreated.
    """
    try:
        with db_cursor() as cur:
            # Order matters due to foreign keys
            cur.execute("DROP TABLE IF EXISTS chunks CASCADE;")
            cur.execute("DROP TABLE IF EXISTS toc_nodes CASCADE;")
            cur.execute("DROP TABLE IF EXISTS rag_documents CASCADE;")
            cur.execute("DROP TABLE IF EXISTS project_documents CASCADE;")

            cur.execute("DROP TABLE IF EXISTS budget_items CASCADE;")
            cur.execute("DROP TABLE IF EXISTS kpis CASCADE;")
            cur.execute("DROP TABLE IF EXISTS report_deliverables CASCADE;")
            cur.execute("DROP TABLE IF EXISTS progress_reports CASCADE;")

            cur.execute("DROP TABLE IF EXISTS projects CASCADE;")
            cur.execute("DROP TABLE IF EXISTS grants CASCADE;")
            cur.execute("DROP TABLE IF EXISTS impact_areas CASCADE;")
            cur.execute("DROP TABLE IF EXISTS organizations CASCADE;")

        # Re-initialize immediately
        from .db_schema import initialize_database
        initialize_database()

        return {"status": "success", "message": "Database reset complete. Tables recreated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    import uvicorn
    import os

    # Get the PORT from the environment (Cloud Run sets this to 8080 automatically)
    port = int(os.environ.get("PORT", 8080))

    print(f"Starting YH GrantNav server on port {port}...")

    # CRITICAL: host must be "0.0.0.0" for Docker/Cloud Run
    uvicorn.run(
        "yh_rag_cloud_api.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Set reload=False for production
        workers=1  # Keep workers low for Cloud Run instances
    )