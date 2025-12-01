# yh_rag_cloud_api/parsers/budget_parser_cloud.py

import re
import json
from typing import Dict, Any
from . import budget_parser_standard, budget_parser_six_month, budget_parser_academic


def parse_budget_cloud(csv_content: str) -> Dict[str, Any]:
    """
    Main cloud budget parser - routes to appropriate template parser.
    """
    print(f"  [Budget Parser Router] Starting template-aware routing...")

    try:
        # Detect template type first
        template_type = detect_budget_template(csv_content)
        print(f"  [Budget Parser Router] Detected template: {template_type}")

        # Route to appropriate parser
        if template_type == "standard_multi_year":
            print(f"  [Budget Parser Router] Routing to Standard Multi-Year Parser...")
            return budget_parser_standard.parse_standard_multi_year_budget(csv_content)

        elif template_type == "six_month_periods":
            print(f"  [Budget Parser Router] Routing to Six-Month Period Parser...")
            return budget_parser_six_month.parse_six_month_period_budget(csv_content)

        elif template_type == "academic_year":
            print(f"  [Budget Parser Router] Routing to Academic Year Parser...")
            return budget_parser_academic.parse_academic_year_budget(csv_content)

        else:
            print(f"  [Budget Parser Router] Using Standard Parser as fallback...")
            return budget_parser_standard.parse_standard_multi_year_budget(csv_content)

    except Exception as e:
        print(f"  [Budget Parser Router] ERROR: {e}")
        return {"error": f"Budget parsing failed: {str(e)}"}


def detect_budget_template(csv_content: str) -> str:
    """
    Detect which budget template format we're dealing with.
    """
    lines = csv_content.split('\n')

    # Check for Chumbaka-style template (standard quarters with explicit year columns)
    if any('2020' in line and '2021' in line and '2022' in line for line in lines[:50]):
        return "standard_multi_year"

    # Check for Bajau Laut-style template (6-month periods: Jan-Jun, Jul-Dec)
    if any('Jan-Jun' in line and 'Jul-Dec' in line for line in lines[:50]):
        return "six_month_periods"

    # Check for GSS-style template (academic year quarters)
    if any(term in line for line in lines[:50] for term in ['Oct-Dec', 'Jan-Mar', 'Apr-Jun', 'Jul-Sep']):
        return "academic_year"

    # Check for formula-based templates (like Bajau Laut)
    if any('=2000*24' in line or '=C66' in line for line in lines):
        return "six_month_periods"

    return "standard_multi_year"  # Default to most common format


# Alias for backward compatibility
def parse_budget_cloud_ai(csv_content: str) -> Dict[str, Any]:
    """Direct AI parser alias."""
    return parse_budget_cloud(csv_content)