# yh_rag_api/parsers/grant_agreement_parser_cloud.py

import json
import re
from datetime import datetime
from typing import Dict, Any, Optional
import unicodedata

try:
    # Use the real LLM wrappers in the main environment
    from ..rag_utils import ask_gemini
except ImportError:
    # Fallback for standalone script testing
    def ask_gemini(prompt: str, json_schema: Optional[Dict[str, Any]] = None) -> str:
        print("[MOCK GEMINI CALL for Cloud Grant Parser]")
        # Mock structured response for testing
        return json.dumps({
            "grant_recipient_name": "CHUMBAKA SDN. BHD.",
            "grant_recipient_reg_no": "1065257-D",
            "grant_recipient_address": "303-4-8, Block B, Krystal Point, Jalan Sultan Azlan Shah, 11900 Bayan Lepas, Penang, Malaysia",
            "project_name": "Transformation of Sarawak Rural Primary schools to School 4.0",
            "org_background": "The Organisation is involved in building life-skill through technology and to re-ignite children's passion for learning and advocates Transformation of Sarawak Rural Primary schools to School 4.0.",
            "grant_amount": 1000000.0,
            "duration_start_date": "01 February 2020",
        })


def clean_control_chars(s: str) -> str:
    """Removes common control characters except newline and tab from a string."""
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in ('\n', '\t'))


def _parse_flexible_date(date_str: Any) -> Optional[str]:
    """
    Tries to parse various date formats (including from OCR)
    and returns a YYYY-MM-DD string or None. (Copied from grant_agreement_parser)
    """
    if not isinstance(date_str, str):
        return None

    cleaned_date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str.strip())
    cleaned_date_str = re.sub(r'(\d{2}/\d{2}/\d{4})\d$', r'\1', cleaned_date_str)
    if re.match(r'^[/\s]+/\d{4}$', cleaned_date_str):
        return None

    dt = None
    fmts = [
        '%d/%m/%Y',  # 1/10/2022
        '%d %B %Y',  # 1 October 2022
        '%Y-%m-%d',  # 2022-10-01
        '%d %b %Y',  # 1 Oct 2022
        '%d %B, %Y',  # 1 October, 2022
    ]

    for fmt in fmts:
        try:
            dt = datetime.strptime(cleaned_date_str, fmt)
            break
        except ValueError:
            pass

    if dt:
        return dt.strftime('%Y-%m-%d')
    else:
        if re.match(r'^[A-Za-z]{3}\s\d{4}$', cleaned_date_str):
            try:
                dt = datetime.strptime(cleaned_date_str, '%b %Y')
                return dt.strftime('%Y-%m-01')
            except ValueError:
                pass
        return None


def extract_key_grant_fields_cloud(ga_text: str) -> dict:
    """
    Extracts key fields from the Grant Agreement text using a pure Cloud LLM (Gemini) approach
    for structured JSON output, which is robust against OCR artifacts.
    """
    print("  Extracting key grant fields via pure GEMINI Structured JSON approach...")

    cleaned_text = clean_control_chars(ga_text)

    # Define the JSON schema for structured extraction
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "grant_recipient_name": {"type": "STRING",
                                     "description": "The full, clean name of the Grant Recipient, excluding any preceding OCR noise or leading organization tags."},
            "grant_recipient_reg_no": {"type": "STRING", "description": "The Company Registration Number."},
            "grant_recipient_address": {"type": "STRING", "description": "The principal place of business address."},
            "project_name": {"type": "STRING",
                             "description": "The title of the project, found in Recital B and/or defined terms."},
            "org_background": {"type": "STRING",
                               "description": "The descriptive text about the Organisation's purpose, typically found in Recital B."},
            "grant_amount": {"type": "NUMBER",
                             "description": "The total grant amount in MYR, converted to a float/number (e.g., 1000000.0)."},
            "duration_start_date": {"type": "STRING",
                                    "description": "The project start date as a raw date string (e.g., '01 February 2020')."}
        },
        "required": ["grant_recipient_name", "grant_recipient_reg_no", "grant_recipient_address", "project_name",
                     "grant_amount", "duration_start_date"]
    }

    # Use a focused prompt to instruct the LLM on how to handle the OCR text
    prompt = f"""
    You are a precise document data extraction bot.
    Analyze the 'Grant Agreement Text' provided below. The text may contain OCR noise.
    Extract the required fields strictly from the text.
    - For 'grant_recipient_name', focus only on the main, clean name (e.g., 'CHUMBAKA SDN. BHD.'), ignore any leading OCR junk like '0098MLKOOMN1'.
    - For 'grant_amount', convert the value to a clean floating point number.
    - For dates, return the raw date string from the document (e.g., '01 February 2020').

    Grant Agreement Text:
    ---
    {cleaned_text}
    ---
    """

    llm_result = {}
    try:
        # Call Gemini with structured JSON requirement
        response_text = ask_gemini(prompt, json_schema=response_schema)

        # Robust JSON Parsing (as LLM response is guaranteed to be JSON string)
        llm_result = json.loads(response_text)
        print("  LLM structured extraction successful.")

    except Exception as e:
        print(f"  ERROR: LLM structured extraction failed: {e}")
        # Continue with empty result if LLM fails

    # --- POST-PROCESSING AND FINAL RETURN ---

    # 1. Sanitize/Format Date
    start_date_raw = llm_result.get("duration_start_date")
    formatted_start_date = _parse_flexible_date(start_date_raw)

    # 2. Convert amount to float (if the LLM didn't already, or if it's None)
    grant_amount_raw = llm_result.get("grant_amount")
    if isinstance(grant_amount_raw, str):
        cleaned_amount = re.sub(r'[RM,\s]', '', grant_amount_raw)
        try:
            llm_result["grant_amount"] = float(cleaned_amount)
        except (ValueError, TypeError):
            llm_result["grant_amount"] = None

    final_result = {
        "grant_recipient_name": llm_result.get("grant_recipient_name"),
        "grant_recipient_reg_no": llm_result.get("grant_recipient_reg_no"),
        "grant_recipient_address": llm_result.get("grant_recipient_address"),
        "project_name": llm_result.get("project_name"),
        "org_background": llm_result.get("org_background"),
        "description": None,

        "duration_start_date": formatted_start_date,  # Use the sanitized date
        "duration_end_date": None,  # Not requested/extracted
        "grant_amount": llm_result.get("grant_amount"),

        # Other fields required by the app schema but not extracted by this function
        "contact_name": None,
        "contact_title": None,
        "contact_phone": None,
        "contact_email": None,
        "schedule_4_milestones": [],
    }

    print(f"Extracted Organization (LLM): {final_result.get('grant_recipient_name')}")
    print(f"Extracted Project Title (LLM): {final_result.get('project_name')}")
    print(f"Extracted Start Date (LLM, Formatted): {final_result.get('duration_start_date')}")

    return final_result