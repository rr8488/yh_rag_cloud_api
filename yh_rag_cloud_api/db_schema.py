# yh_rag_cloud_api/db_schema.py

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import contextlib
from .settings import settings


@contextlib.contextmanager
def db_cursor():
    """Provides a transactional database cursor."""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(settings.database_url)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        yield cur
        conn.commit()
    except Exception as e:
        print(f"FATAL: DB Error. {e}")
        if conn: conn.rollback()
        raise
    finally:
        if cur: cur.close()
        if conn: conn.close()


def initialize_database():
    """
    Creates the 'RAG-Ready' Relational Schema.
    INCLUDES AUTO-MIGRATION for the 'metadata' column.
    """
    print("Initializing Database Schema...")
    conn = None
    try:
        conn = psycopg2.connect(settings.database_url)
        conn.autocommit = True

        with conn.cursor() as cur:
            # 1. Enable Vector Extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # ==========================================
            # MODULE 1: LOOKUP TABLES
            # ==========================================
            print("  Creating Lookup Tables...")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS organizations (
                    id TEXT PRIMARY KEY,       
                    ref_no TEXT,               
                    name TEXT NOT NULL,        
                    description TEXT,          
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS impact_areas (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS grants (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT,
                    year INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # ==========================================
            # MODULE 2: CORE PROJECT TABLE
            # ==========================================
            print("  Creating 'projects' table...")

            # 1. Create Table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,       
                    ref_no TEXT,               

                    name TEXT NOT NULL,        
                    description TEXT,          

                    -- FOREIGN KEYS
                    org_id TEXT REFERENCES organizations(id),
                    grant_id TEXT REFERENCES grants(id),
                    impact_area_id TEXT REFERENCES impact_areas(id),

                    -- METRICS
                    grant_amount NUMERIC(15, 2),
                    duration_start_date DATE,
                    duration_end_date DATE,

                    -- GEOGRAPHY
                    locations TEXT[],

                    -- RAW DATA
                    proposal_data JSONB,
                    grant_agreement_data JSONB,

                    -- METADATA (Defined here for new DBs)
                    metadata JSONB DEFAULT '{}'::jsonb,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP
                );
            """)

            # 2. AUTO-MIGRATION: Ensure 'metadata' exists for OLD DBs
            # This is safe to run on every startup (Idempotent)
            print("  Verifying/Migrating 'metadata' column...")
            cur.execute("""
                ALTER TABLE projects 
                ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;
            """)

            # ==========================================
            # MODULE 3: DOCUMENTS & RAG
            # ==========================================
            print("  Creating 'project_documents' table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS project_documents (
                    id TEXT PRIMARY KEY,
                    project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
                    doc_type TEXT NOT NULL,

                    name TEXT NOT NULL,        
                    file_name TEXT NOT NULL,   
                    file_path TEXT NOT NULL,   

                    file_extension TEXT,       
                    file_url TEXT,             

                    file_size_bytes BIGINT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id TEXT PRIMARY KEY REFERENCES project_documents(id) ON DELETE CASCADE,
                    project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
                    toc_config JSONB,
                    total_pages INTEGER
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS toc_nodes (
                    id TEXT PRIMARY KEY,
                    document_id TEXT REFERENCES rag_documents(id) ON DELETE CASCADE,
                    title TEXT,
                    level INTEGER,
                    start_pp INTEGER,
                    end_pp INTEGER
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    node_id TEXT REFERENCES toc_nodes(id) ON DELETE CASCADE,
                    chunk_text TEXT,
                    embedding vector(768)
                );
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks 
                USING hnsw (embedding vector_l2_ops);
            """)

            # ==========================================
            # MODULE 4: THE "PLAN" (KPIs & Budget)
            # ==========================================
            print("  Creating Plan Tables...")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS kpis (
                    id TEXT PRIMARY KEY,
                    project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
                    milestone_number INTEGER,
                    description TEXT,
                    sod TEXT,
                    due_date DATE,
                    amount NUMERIC(15, 2),
                    status TEXT DEFAULT 'Pending'
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS budget_items (
                    id TEXT PRIMARY KEY,
                    project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
                    category_l1 TEXT,
                    category_l2 TEXT,
                    description TEXT,
                    requested_amount NUMERIC(15, 2)
                );
            """)

            # ==========================================
            # MODULE 5: THE "ACTUALS" (Reports)
            # ==========================================
            print("  Creating Progress Reporting Tables...")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS progress_reports (
                    id TEXT PRIMARY KEY,
                    project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
                    project_document_id TEXT REFERENCES project_documents(id) ON DELETE SET NULL,

                    report_number INTEGER,
                    report_date DATE,
                    report_type TEXT,

                    funds_disbursed NUMERIC(15, 2),
                    funds_utilized NUMERIC(15, 2),
                    funds_unutilized NUMERIC(15, 2),

                    executive_summary TEXT,
                    challenges_summary TEXT,
                    lessons_learned_summary TEXT,
                    success_stories_summary TEXT,

                    full_data JSONB, 
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_pr_beneficiaries 
                ON progress_reports USING GIN ((full_data->'beneficiaries'));
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS report_deliverables (
                    id TEXT PRIMARY KEY,
                    progress_report_id TEXT REFERENCES progress_reports(id) ON DELETE CASCADE,
                    description TEXT,
                    status TEXT,
                    progress_update TEXT
                );
            """)

        print("\n✅ Database Schema Initialized & Migrated.")

    except Exception as e:
        print(f"❌ DATABASE SETUP FAILED: {e}")
        raise
    finally:
        if conn: conn.close()


if __name__ == "__main__":
    initialize_database()