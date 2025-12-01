# yh_rag_cloud_api/parsers/budget_parser_standard.py

import re
import json
from typing import Dict, Any
from .budget_parser_utils import (
    extract_json_from_response_enhanced,
    _validate_with_accuracy_checks_enhanced,
    safe_float,
    parse_currency_to_float
)


def parse_standard_multi_year_budget(csv_content: str) -> Dict[str, Any]:
    """
    Parser for standard multi-year templates (calendar quarters with explicit years).
    """
    print(f"  [Standard Multi-Year Parser] Processing template...")

    try:
        # Extract metadata
        metadata = extract_metadata_standard(csv_content)
        print(f"  [Standard Multi-Year Parser] Extracted metadata: {metadata}")

        # Detect quarter structure
        quarter_structure = detect_quarter_structure_standard(csv_content, metadata.get('funding_period', ''))
        print(f"  [Standard Multi-Year Parser] Quarter structure: {quarter_structure}")

        # Use AI parsing
        from ..rag_utils import ask_gemini

        truncated_content = csv_content[:30000] if len(csv_content) > 30000 else csv_content
        metadata_total = parse_currency_to_float(metadata.get('total_amount_requested', '0'))

        prompt = f"""
        EXTRACT budget data from this STANDARD MULTI-YEAR template.

        ACTUAL PROJECT DETAILS:
        - Project Title: {metadata.get('project_title', 'Unknown')}
        - Organization: {metadata.get('organisation_name', 'Unknown')} 
        - Total Amount Requested: {metadata.get('total_amount_requested', 'Unknown')}
        - Funding Period: {metadata.get('funding_period', 'Unknown')}

        QUARTER STRUCTURE (STANDARD CALENDAR YEARS):
        {quarter_structure}

        BUDGET DATA:
        {truncated_content}

        IMPORTANT: This is a STANDARD template with calendar year quarters.
        Use the exact quarter structure above and extract ACTUAL amounts.
        Follow the distribution across quarters as shown in the budget table.
        Ensure sum(tranche_breakdown) = total_amount for each line item.
        """

        print(f"  [Standard Multi-Year Parser] Calling Gemini...")
        response = ask_gemini(prompt)

        result = extract_json_from_response_enhanced(response)

        if "error" in result:
            return {"error": "AI extraction failed for standard template"}

        # Ensure metadata is correct
        if "metadata" in result:
            result["metadata"] = {
                "project_title": metadata.get('project_title', 'Project Title Not Found'),
                "organisation_name": metadata.get('organisation_name', 'Organization Name Not Found'),
                "total_amount_requested": metadata.get('total_amount_requested', 'RM0'),
                "funding_period": metadata.get('funding_period', 'Funding Period Not Found'),
                "amount_approved": metadata.get('amount_approved', metadata.get('total_amount_requested', 'RM0')),
                "quarter_structure": quarter_structure
            }
            result["sanity_checks"]["metadata_total"] = metadata_total

        return _validate_with_accuracy_checks_enhanced(result)

    except Exception as e:
        print(f"  [Standard Multi-Year Parser] ERROR: {e}")
        return {"error": f"Standard multi-year parsing failed: {str(e)}"}


def extract_metadata_standard(csv_content: str) -> Dict[str, str]:
    """Universal metadata extraction for standard multi-year templates."""
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

        # Amount Approved - universal pattern
        elif 'amount approved:' in line_clean.lower() or 'approved amount:' in line_clean.lower():
            parts = line_clean.split(':', 1)
            if len(parts) > 1:
                amount_text = parts[1].strip()
                amount_text = re.sub(r'\"\,\"\"\,.*$', '', amount_text)
                amount_text = re.sub(r'\"\,\".*$', '', amount_text)
                metadata['amount_approved'] = amount_text

    # Set universal defaults - NO PROJECT-SPECIFIC VALUES
    if 'project_title' not in metadata:
        metadata['project_title'] = 'Project Title Not Found'
    if 'organisation_name' not in metadata:
        metadata['organisation_name'] = 'Organization Name Not Found'
    if 'total_amount_requested' not in metadata:
        metadata['total_amount_requested'] = 'RM0'
    if 'funding_period' not in metadata:
        metadata['funding_period'] = 'Funding Period Not Found'
    if 'amount_approved' not in metadata:
        metadata['amount_approved'] = metadata['total_amount_requested']

    return metadata


def detect_quarter_structure_standard(csv_content: str, funding_period: str) -> Dict[str, Any]:
    """Quarter structure detection for standard templates."""
    lines = csv_content.split('\n')

    for line in lines:
        if any(f' {year} ' in line for year in ['2020', '2021', '2022', '2023', '2024']):
            years_found = []
            for year in ['2020', '2021', '2022', '2023', '2024']:
                if year in line:
                    years_found.append(year)

            if years_found:
                year_count = len(years_found)
                start_year = min(int(y) for y in years_found)
                end_year = max(int(y) for y in years_found)

                quarter_labels = []
                for year in range(start_year, end_year + 1):
                    for quarter in range(1, 5):
                        quarter_labels.append(f"{year}_Q{quarter}")

                return {
                    "type": "multi_year_calendar",
                    "quarters": ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"],
                    "start_month": "January",
                    "total_quarters": year_count * 4,
                    "year_span": f"{year_count} years ({start_year}-{end_year})",
                    "quarter_labels": quarter_labels,
                    "years": [str(y) for y in range(start_year, end_year + 1)],
                    "start_year": start_year,
                    "end_year": end_year
                }

    # Default structure for single year
    return {
        "type": "calendar_year",
        "quarters": ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"],
        "start_month": "January",
        "total_quarters": 4,
        "year_span": "1 year",
        "quarter_labels": ["Q1", "Q2", "Q3", "Q4"]
    }