# yh_rag_cloud_api/parsers/budget_parser_academic.py

import re
import json
from typing import Dict, Any
from .budget_parser_utils import (
    extract_json_from_response_enhanced,
    _validate_with_accuracy_checks_enhanced,
    safe_float,
    parse_currency_to_float
)


def parse_academic_year_budget(csv_content: str) -> Dict[str, Any]:
    """
    Parser for academic year templates (Oct-Sep quarters).
    """
    print(f"  [Academic Year Parser] Processing template...")

    try:
        from ..rag_utils import ask_gemini

        # Extract metadata for academic template
        metadata = extract_metadata_academic(csv_content)
        print(f"  [Academic Year Parser] Extracted metadata: {metadata}")

        # Define academic year structure
        academic_structure = {
            "type": "academic_year",
            "quarters": ["Q1 (Oct-Dec)", "Q2 (Jan-Mar)", "Q3 (Apr-Jun)", "Q4 (Jul-Sep)"],
            "start_month": "October",
            "total_quarters": 4,
            "year_span": "1 academic year",
            "quarter_labels": ["Q1", "Q2", "Q3", "Q4"],
            "note": "Uses academic year quarters (October to September)"
        }

        truncated_content = csv_content[:30000] if len(csv_content) > 30000 else csv_content
        metadata_total = parse_currency_to_float(metadata.get('total_amount_requested', '0'))

        prompt = f"""
        EXTRACT budget data from this ACADEMIC YEAR template.

        CRITICAL: This template uses ACADEMIC YEAR QUARTERS (Oct-Sep)!

        PROJECT DETAILS:
        - Project Title: {metadata.get('project_title', 'Unknown')}
        - Organization: {metadata.get('organisation_name', 'Unknown')} 
        - Total Amount Requested: {metadata.get('total_amount_requested', 'Unknown')}
        - Funding Period: {metadata.get('funding_period', 'Unknown')}

        QUARTER STRUCTURE (ACADEMIC YEAR):
        - Q1: October - December
        - Q2: January - March  
        - Q3: April - June
        - Q4: July - September

        BUDGET DATA:
        {truncated_content}

        Return JSON with academic year quarter structure...
        """

        print(f"  [Academic Year Parser] Calling Gemini...")
        response = ask_gemini(prompt)

        result = extract_json_from_response_enhanced(response)

        if "error" in result:
            return {"error": "AI extraction failed for academic year template"}

        # Ensure metadata is correct
        if "metadata" in result:
            result["metadata"] = {
                "project_title": metadata.get('project_title', 'Project Title Not Found'),
                "organisation_name": metadata.get('organisation_name', 'Organization Name Not Found'),
                "total_amount_requested": metadata.get('total_amount_requested', 'RM0'),
                "funding_period": metadata.get('funding_period', 'Funding Period Not Found'),
                "amount_approved": metadata.get('amount_approved', metadata.get('total_amount_requested', 'RM0')),
                "quarter_structure": academic_structure
            }
            result["sanity_checks"]["metadata_total"] = metadata_total

        return _validate_with_accuracy_checks_enhanced(result)

    except Exception as e:
        print(f"  [Academic Year Parser] ERROR: {e}")
        return {"error": f"Academic year parsing failed: {str(e)}"}


def extract_metadata_academic(csv_content: str) -> Dict[str, str]:
    """Universal metadata extraction for academic year templates."""
    lines = csv_content.split('\n')
    metadata = {}

    for i, line in enumerate(lines):
        line_clean = line.strip()

        if not line_clean or line_clean.startswith('(') or line_clean.startswith('|'):
            continue

        # Project Title - universal pattern
        if 'project title:' in line_clean.lower():
            parts = line_clean.split(':', 1)
            if len(parts) > 1:
                project_title = parts[1].strip()
                project_title = re.sub(r'\"\,\"\"\,.*$', '', project_title)
                project_title = re.sub(r'\"\,\".*$', '', project_title)
                metadata['project_title'] = project_title

        # Organization Name - universal pattern
        elif 'organisation name' in line_clean.lower() or 'organization name' in line_clean.lower():
            parts = line_clean.split(':', 1)
            if len(parts) > 1:
                org_name = parts[1].strip()
                org_name = re.sub(r'\"\,\"\"\,.*$', '', org_name)
                org_name = re.sub(r'\"\,\".*$', '', org_name)
                metadata['organisation_name'] = org_name

        # Total Amount - universal pattern
        elif 'total amount requested:' in line_clean.lower():
            parts = line_clean.split(':', 1)
            if len(parts) > 1:
                amount_text = parts[1].strip()
                amount_text = re.sub(r'\"\,\"\"\,.*$', '', amount_text)
                amount_text = re.sub(r'\"\,\".*$', '', amount_text)
                metadata['total_amount_requested'] = amount_text

        # Funding Period - universal pattern
        elif 'funding period:' in line_clean.lower():
            parts = line_clean.split(':', 1)
            if len(parts) > 1:
                period_text = parts[1].strip()
                period_text = re.sub(r'\"\,\"\"\,.*$', '', period_text)
                period_text = re.sub(r'\"\,\".*$', '', period_text)
                metadata['funding_period'] = period_text

        #