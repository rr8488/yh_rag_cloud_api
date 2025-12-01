# yh_rag_api/parsers/grant_agreement_parser_production.py

import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import unicodedata

try:
    from ..rag_utils import _ask_ollama, ask_gemini
except ImportError:
    def _ask_ollama(prompt: str, model: str) -> str:
        return ""


    def ask_gemini(prompt: str) -> str:
        return ""


def clean_control_chars(s: str) -> str:
    """Removes common control characters except newline and tab from a string."""
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in ('\n', '\t'))


def _parse_flexible_date(date_str: Any) -> Optional[str]:
    """
    Tries to parse various date formats (including from OCR)
    and returns a YYYY-MM-DD string or None.
    """
    if not isinstance(date_str, str):
        return None

    # Clean up date string
    cleaned_date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str.strip())
    cleaned_date_str = re.sub(r'(\d{2}/\d{2}/\d{4})\d$', r'\1', cleaned_date_str)

    # Handle dates like '/ /2024' -> None
    if re.match(r'^[/\s]+/\d{4}$', cleaned_date_str):
        return None

    dt = None
    # List of formats to try
    fmts = [
        '%d %B %Y',  # 19 February 2020
        '%d %b %Y',  # 19 Feb 2020
        '%d/%m/%Y',  # 19/02/2020
        '%Y-%m-%d',  # 2020-02-19
        '%d-%m-%Y',  # 19-02-2020
        '%B %d, %Y',  # February 19, 2020
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
        return None


def extract_key_grant_fields(ga_text: str) -> dict:
    """
    Extract only essential grant fields from grant agreement text.
    """
    print("  Extracting key grant fields via precise regex patterns...")

    # Print the raw extracted text for debugging
    print(f"\n{'=' * 80}")
    print("DEBUG: RAW EXTRACTED TEXT (First 2000 characters):")
    print(f"{'=' * 80}")
    print(ga_text[:2000])
    print(f"{'=' * 80}\n")

    cleaned_text = clean_control_chars(ga_text)
    flags = re.DOTALL | re.IGNORECASE
    raw_data = {}

    # --- 1. ORGANIZATION NAME (From Cover Page) ---
    print("  Looking for organization name from cover page...")

    # Extract from the clean cover page section: BETWEEN [name] ("Organisation")
    cover_page_patterns = [
        r"BETWEEN\s+([^\n\(]+?)\s+\(\"Organisation\"\)",
        r"BETWEEN\s+([^\n]+?)\s+\(\"Organisation\"\)",
        r"BETWEEN\s+(.*?)\s+\(\"Organisation\"\)",
    ]

    for i, pattern in enumerate(cover_page_patterns):
        org_match = re.search(pattern, cleaned_text, flags)
        if org_match:
            org_name = org_match.group(1).strip()
            # Clean up the name - remove extra whitespace and artifacts
            org_name = re.sub(r'\s+', ' ', org_name)
            org_name = re.sub(r'[^\w\s\.\(\)&,\-]', '', org_name)
            raw_data["grant_recipient_name"] = org_name
            print(f"  ✓ Found organization name from cover page with pattern {i}: '{org_name}'")
            break
    else:
        print("  ✗ No organization name found from cover page")

    # --- 2. REGISTRATION NUMBER ---
    print("  Looking for registration number pattern...")
    reg_patterns = [
        r"Company No:\s*([^\)\s]+)",
        r"\(Company No:\s*([^\)]+?)\)",
        r"Company No:\s*([^\)]+?)\)",
    ]

    for i, pattern in enumerate(reg_patterns):
        reg_match = re.search(pattern, cleaned_text, flags)
        if reg_match:
            reg_no = reg_match.group(1).strip()
            reg_no = re.sub(r'[^\w\-]', '', reg_no)
            raw_data["grant_recipient_reg_no"] = reg_no
            print(f"  ✓ Found registration number with pattern {i}: '{reg_no}'")
            break
    else:
        print("  ✗ No registration number found with any pattern")

    # --- 3. ORGANIZATION ADDRESS ---
    print("  Looking for organization address pattern...")
    address_patterns = [
        r"principal place of business address at\s+([^\(]+?)\s+\(\"Organisation\"\)",
        r"business address at\s+([^\(]+?)\s+\(\"Organisation\"\)",
        r"address at\s+([^\(]+?)\s+\(\"Organisation\"\)",
    ]

    for i, pattern in enumerate(address_patterns):
        address_match = re.search(pattern, cleaned_text, flags)
        if address_match:
            address = address_match.group(1).strip()
            address = re.sub(r'\s+', ' ', address)
            address = re.sub(r'\n', ' ', address)
            address = re.sub(r'[\(\)\"]', '', address).strip()
            raw_data["grant_recipient_address"] = address
            print(f"  ✓ Found address with pattern {i}: '{address}'")
            break
    else:
        print("  ✗ No address found with any pattern")

    # --- 4. GRANT AMOUNT ---
    print("  Looking for grant amount pattern...")
    amount_patterns = [
        r"Ringgit Malaysia\s*([\d,]+\.?\d*)\s+or such other amount",
        r"up to Ringgit Malaysia\s*([\d,]+\.?\d*)\s+or such",
        r"Ringgit Malaysia\s*([\d,]+\.?\d*)",
        r"RM\s*([\d,]+\.?\d*)\s+or such other amount",
    ]

    for i, pattern in enumerate(amount_patterns):
        amount_match = re.search(pattern, cleaned_text, flags)
        if amount_match:
            amount_str = amount_match.group(1).strip()
            cleaned_amount = re.sub(r'[RM,\s]', '', amount_str)
            try:
                raw_data["grant_amount"] = float(cleaned_amount)
                print(f"  ✓ Found grant amount with pattern {i}: {raw_data['grant_amount']}")
                break
            except (ValueError, TypeError):
                print(f"  ✗ Failed to convert amount: '{amount_str}'")
                continue
    else:
        print("  ✗ No grant amount found with any pattern")

    # --- 5. PROJECT DURATION ---
    print("  Looking for duration dates pattern...")

    # First, find the DURATION section
    duration_section_patterns = [
        r"2\.\s*DURATION(.*?)(?=3\.|4\.|$)",
        r"DURATION\s*(.*?)(?=3\.|GRANT|$)",
        r"2\s*DURATION(.*?)(?=3|GRANT|$)",
    ]

    duration_text = None
    for pattern in duration_section_patterns:
        duration_section_match = re.search(pattern, cleaned_text, flags)
        if duration_section_match:
            duration_text = duration_section_match.group(1)
            print(f"  ✓ Found DURATION section")
            break

    if duration_text:
        # Look for date pattern in the duration section
        duration_patterns = [
            r"from\s*(\d{1,2}\s+\w+\s+\d{4})\s+to\s+(\d{1,2}\s+\w+\s+\d{4})\s*\(\"Grant Period\"\)",
            r"from\s*(\d{1,2}\s+\w+\s+\d{4})\s+to\s+(\d{1,2}\s+\w+\s+\d{4})\s*\(Grant Period\)",
            r"commencing from\s*(\d{1,2}\s+\w+\s+\d{4})\s+to\s+(\d{1,2}\s+\w+\s+\d{4})",
            r"from\s*(\d{1,2}\s+\w+\s+\d{4})\s+to\s+(\d{1,2}\s+\w+\s+\d{4})",
        ]

        for i, pattern in enumerate(duration_patterns):
            date_match = re.search(pattern, duration_text, flags)
            if date_match:
                start_date_str = date_match.group(1).strip()
                end_date_str = date_match.group(2).strip()

                raw_data["duration_start_date"] = _parse_flexible_date(start_date_str)
                raw_data["duration_end_date"] = _parse_flexible_date(end_date_str)
                print(f"  ✓ Found dates with pattern {i}: {start_date_str} to {end_date_str}")
                print(f"    Parsed as: {raw_data['duration_start_date']} to {raw_data['duration_end_date']}")
                break
        else:
            print("  ✗ No dates found in DURATION section")
    else:
        print("  ✗ No DURATION section found")

    # --- FINAL RESULT CONSTRUCTION ---
    result = {
        "grant_recipient_name": raw_data.get("grant_recipient_name"),
        "grant_recipient_reg_no": raw_data.get("grant_recipient_reg_no"),
        "grant_recipient_address": raw_data.get("grant_recipient_address"),
        "duration_start_date": raw_data.get("duration_start_date"),
        "duration_end_date": raw_data.get("duration_end_date"),
        "grant_amount": raw_data.get("grant_amount"),
    }

    # Print final results for structured fields
    print(f"\n{'=' * 80}")
    print("FINAL EXTRACTION RESULTS:")
    print(f"{'=' * 80}")
    for field, value in result.items():
        print(f"  {field}: {value}")
    print(f"{'=' * 80}")

    return result