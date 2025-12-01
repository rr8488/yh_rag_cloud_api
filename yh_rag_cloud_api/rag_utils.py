# yh_rag_cloud_api/rag_utils.py

import os
import google.generativeai as genai
# [FIX] Explicit imports for strict type checking
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import json
from contextlib import contextmanager

# Load environment variables
load_dotenv()

# --- GEMINI CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Use a dummy key if not found to prevent import crashes,
    # but runtime calls will fail if not set.
    print("⚠️ WARN: GEMINI_API_KEY not found in env.")
    GEMINI_API_KEY = "dummy_key"

genai.configure(api_key=GEMINI_API_KEY)

# [FIX] Safety Settings using Enums
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

# --- MODELS (Gemini 2.5) ---
model_flash = genai.GenerativeModel(
    'gemini-2.5-flash',
    safety_settings=safety_settings
)

model_pro = genai.GenerativeModel(
    'gemini-2.5-pro',
    safety_settings=safety_settings
)


# --- DATABASE CONNECTION ---
def get_db_connection():
    """Returns a new database connection."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL not found in .env")
    return psycopg2.connect(db_url)


@contextmanager
def db_cursor():
    """Context manager for database cursors."""
    conn = get_db_connection()
    try:
        yield conn.cursor(cursor_factory=DictCursor)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


# --- GEMINI FUNCTIONS ---

def initialize_cloud_clients():
    """Placeholder for cloud init logic if needed."""
    pass


def embed_text(text: str):
    """Generates vector embedding for retrieval."""
    if not text:
        return []
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Embedding Error: {e}")
        return []


def ask_gemini(prompt: str, json_schema=None):
    """
    Sends a prompt to Gemini 2.5 Flash.
    """
    try:
        generation_config = {}
        if json_schema:
            generation_config = {
                "response_mime_type": "application/json",
                "response_schema": json_schema
            }

        response = model_flash.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        print(f"Gemini Generation Error: {e}")
        return f"Error communicating with AI: {e}"


# --- COMPATIBILITY FIX: ADD query_llm ---
def query_llm(prompt: str, model_name: str = None) -> str:
    """
    Legacy wrapper for RAG handler compatibility.
    Redirects to ask_gemini.
    """
    return ask_gemini(prompt)


# --- INTENT CLASSIFIER ---
def classify_intent(question: str) -> str:
    """
    Classifies User Query into 3 distinct buckets:
    SQL_ANALYSIS, METADATA_LOOKUP, or VECTOR_RAG.
    """
    prompt = f"""
    Classify the USER QUESTION into exactly one category: SQL_ANALYSIS, METADATA_LOOKUP, or VECTOR_RAG.

    Definitions:
    - SQL_ANALYSIS: Questions involving numbers, counting, summing, averages, or filtering lists of projects.
    - METADATA_LOOKUP: Questions asking for specific details, facts, or attributes about a project (e.g., "What is the objective?", "List the beneficiaries").
    - VECTOR_RAG: Questions asking for explanations, summaries of narratives, or thematic exploration.

    USER QUESTION: "{question}"

    Return ONLY the category name.
    """
    try:
        response = model_flash.generate_content(prompt)
        cleaned = response.text.strip().upper()
        if "SQL" in cleaned: return "SQL_ANALYSIS"
        if "METADATA" in cleaned: return "METADATA_LOOKUP"
        return "VECTOR_RAG"
    except:
        return "VECTOR_RAG"


# --- METADATA SELECTOR ---
def identify_relevant_metadata_keys(question: str, available_keys: list) -> list:
    """
    Asks the LLM which keys are most likely to contain the answer.
    """
    if not available_keys:
        return []

    keys_str = ", ".join(available_keys[:300])

    prompt = f"""
    You are a database assistant.
    USER QUESTION: "{question}"

    AVAILABLE METADATA KEYS:
    [{keys_str}]

    Which of these keys are most relevant to answering the question?
    Return a JSON array of strings. Example: ["project_goal", "beneficiaries"]
    If none are relevant, return [].
    """

    try:
        response = ask_gemini(prompt)
        text = response.replace("```json", "").replace("```", "").strip()
        selected_keys = json.loads(text)
        if isinstance(selected_keys, list):
            return selected_keys
        return []
    except Exception as e:
        print(f"Key selection failed: {e}")
        return []