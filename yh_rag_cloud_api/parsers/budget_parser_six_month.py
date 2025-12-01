# yh_rag_cloud_api/parsers/budget_parser_six_month.py

import re
import json
from typing import Dict, Any, List
from .budget_parser_utils import (
    safe_float,
    parse_currency_to_float
)


def parse_six_month_period_budget(csv_content: str) -> Dict[str, Any]:
    """
    Rule-based parser for six-month period templates.
    Uses direct extraction instead of AI for better reliability.
    """
    print(f"  [Six-Month Period Parser] Using rule-based parsing...")

    try:
        # Extract metadata
        metadata = extract_metadata_six_month(csv_content)
        print(f"  [Six-Month Period Parser] Extracted metadata: {metadata}")

        # Define period structure
        period_structure = {
            "type": "six_month_periods",
            "periods": ["Jan-Jun Year 1", "Jul-Dec Year 1", "Jan-Jun Year 2", "Jul-Dec Year 2"],
            "start_month": "January",
            "total_periods": 4,
            "year_span": "2 years (24 months)",
            "period_labels": ["Year1_JanJun", "Year1_JulDec", "Year2_JanJun", "Year2_JulDec"],
            "years": ["Year 1", "Year 2"],
            "note": "Uses 6-month periods instead of quarters"
        }

        # Extract budget line items using rule-based approach
        budget_line_items = extract_budget_items_rule_based(csv_content)

        # Calculate totals
        metadata_total = parse_currency_to_float(metadata.get('total_amount_requested', '0'))
        sum_line_items = sum(item.get('total_amount', 0) for item in budget_line_items)

        result = {
            "metadata": {
                "project_title": metadata.get('project_title', ''),
                "organisation_name": metadata.get('organisation_name', ''),
                "total_amount_requested": metadata.get('total_amount_requested', ''),
                "funding_period": metadata.get('funding_period', ''),
                "amount_approved": metadata.get('amount_approved', metadata.get('total_amount_requested', '')),
                "quarter_structure": period_structure
            },
            "budget_line_items": budget_line_items,
            "sanity_checks": {
                "metadata_total": metadata_total,
                "sum_of_line_items": sum_line_items,
                "amounts_tally": abs(metadata_total - sum_line_items) < 10000,
                "line_item_count": len(budget_line_items),
                "message": f"Metadata: RM{metadata_total:,.2f}, Line Items: RM{sum_line_items:,.2f}, Items: {len(budget_line_items)}"
            }
        }

        return _validate_with_accuracy_checks_enhanced(result)

    except Exception as e:
        print(f"  [Six-Month Period Parser] ERROR: {e}")
        return {"error": f"Six-month period parsing failed: {str(e)}"}


def extract_budget_items_rule_based(csv_content: str) -> List[Dict[str, Any]]:
    """
    Rule-based extraction of budget items from six-month template.
    """
    lines = csv_content.split('\n')
    items = []
    current_category = ""
    current_subcategory = ""

    for i, line in enumerate(lines):
        line_clean = line.strip()

        # Skip empty lines and header lines
        if not line_clean or line_clean.startswith('|') or line_clean.startswith('('):
            continue

        # Detect categories
        if 'Project Implementation Cost' in line_clean:
            current_category = "Project Implementation Cost"
            current_subcategory = ""
        elif 'Project Objective 1:' in line_clean:
            current_subcategory = extract_subcategory(line_clean)
        elif 'Project Objective 2:' in line_clean:
            current_subcategory = extract_subcategory(line_clean)
        elif 'Project staff cost' in line_clean:
            current_category = "Project Staff Cost"
            current_subcategory = ""
        elif 'Administrative cost' in line_clean:
            current_category = "Administrative Cost"
            current_subcategory = ""
        elif 'Transportation cost' in line_clean:
            current_category = "Transportation Cost"
            current_subcategory = ""

        # Detect budget line items (lines starting with numbers)
        elif re.match(r'^\d+\s+', line_clean):
            item = extract_budget_line_item(line_clean, lines, i, current_category, current_subcategory)
            if item:
                items.append(item)

    return items


def extract_budget_line_item(line: str, all_lines: List[str], line_index: int, category: str, subcategory: str) -> Dict[
    str, Any]:
    """
    Extract a single budget line item using rule-based approach.
    """
    # Extract description and amounts
    parts = re.split(r'\s{2,}', line.strip())  # Split on multiple spaces
    if len(parts) < 2:
        return None

    # Description is usually the first part after the number
    description = parts[0]
    # Remove leading number
    description = re.sub(r'^\d+\s*', '', description)

    # Look for amounts in subsequent columns
    total_amount = 0.0
    period_amounts = {}

    # Check if this line has amount columns
    if len(parts) > 1:
        # The total amount might be in various columns - look for numeric values
        for part in parts[1:]:
            # Look for formula or numeric value
            amount_value = extract_amount_from_cell(part)
            if amount_value > 0:
                total_amount = amount_value
                break

    # If no amount found in current line, check next line for formulas
    if total_amount == 0 and line_index + 1 < len(all_lines):
        next_line = all_lines[line_index + 1].strip()
        total_amount = extract_amount_from_formula(next_line)

    # Distribute across periods based on template structure
    period_amounts = distribute_amounts_six_month(total_amount, description)

    return {
        "category": category,
        "subcategory": subcategory,
        "line_item_description": description,
        "total_amount": total_amount,
        "tranche_breakdown": period_amounts,
        "is_subtotal": False,
        "budget_type": "PMF"  # Default to PMF for this template
    }


def extract_amount_from_cell(cell_content: str) -> float:
    """Extract amount from a cell content."""
    # Remove formula symbols and extract numeric value
    clean_content = re.sub(r'^=.*\(|\)$', '', cell_content)
    clean_content = re.sub(r'\$[A-Z]+\$\d+', '', clean_content)  # Remove cell references
    return parse_currency_to_float(clean_content)


def extract_amount_from_formula(formula_line: str) -> float:
    """Extract amount from formula line."""
    formulas = re.findall(r'=([^,\n]+)', formula_line)
    for formula in formulas:
        try:
            # Clean and evaluate formula
            clean_formula = formula.strip()
            clean_formula = re.sub(r'\$[A-Z]+\$\d+', '1', clean_formula)
            clean_formula = re.sub(r'[A-Z]+\d+', '1', clean_formula)

            if all(c in '0123456789+-*/.() ' for c in clean_formula):
                result = eval(clean_formula)
                return float(result)
        except:
            continue
    return 0.0


def distribute_amounts_six_month(total_amount: float, description: str) -> Dict[str, float]:
    """
    Distribute amounts across six-month periods based on template column analysis.
    """
    # For now, use even distribution as fallback
    # In a more advanced version, we would analyze the actual column distributions
    per_period = total_amount / 4
    return {
        "Year1_JanJun": per_period,
        "Year1_JulDec": per_period,
        "Year2_JanJun": per_period,
        "Year2_JulDec": per_period
    }


def extract_subcategory(line: str) -> str:
    """Extract subcategory from objective line."""
    # Extract the main objective description
    if ':' in line:
        return line.split(':', 1)[1].strip()
    return line.strip()


def extract_metadata_six_month(csv_content: str) -> Dict[str, str]:
    """Enhanced metadata extraction with better pattern matching."""
    lines = csv_content.split('\n')
    metadata = {}

    for i, line in enumerate(lines):
        line_clean = line.strip()

        if not line_clean or line_clean.startswith('(') or line_clean.startswith('|'):
            continue

        # Project Title - improved pattern matching
        if 'project title' in line_clean.lower() or 'tajuk projek' in line_clean.lower():
            # Look for multi-line title
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not next_line.startswith('|') and len(next_line) > 10:
                    metadata['project_title'] = clean_text(next_line)

        # Organization Name
        elif 'organisation name' in line_clean.lower() or 'nama organisasi' in line_clean.lower():
            if ':' in line_clean:
                parts = line_clean.split(':', 1)
                if len(parts) > 1:
                    metadata['organisation_name'] = clean_text(parts[1])

        # Funding Period
        elif 'funding period' in line_clean.lower() or 'tempoh dana' in line_clean.lower():
            if ':' in line_clean:
                parts = line_clean.split(':', 1)
                if len(parts) > 1:
                    metadata['funding_period'] = clean_text(parts[1])

    # Calculate total from budget items
    calculated_total = calculate_total_from_budget(csv_content)
    if calculated_total > 0:
        metadata['total_amount_requested'] = f'RM{calculated_total:,.2f}'
        metadata['amount_approved'] = f'RM{calculated_total:,.2f}'

    return metadata


def clean_text(text: str) -> str:
    """Clean extracted text."""
    if not text:
        return ""
    cleaned = re.sub(r'\"\,\"\"\,.*$', '', text)
    cleaned = re.sub(r'\"\,\".*$', '', cleaned)
    return cleaned.strip()


def calculate_total_from_budget(csv_content: str) -> float:
    """Calculate total from budget formulas."""
    lines = csv_content.split('\n')
    total = 0.0

    for line in lines:
        formulas = re.findall(r'=([^,\n]+)', line)
        for formula in formulas:
            try:
                clean_formula = formula.strip()
                clean_formula = re.sub(r'\$[A-Z]+\$\d+', '1', clean_formula)
                clean_formula = re.sub(r'[A-Z]+\d+', '1', clean_formula)

                if all(c in '0123456789+-*/.() ' for c in clean_formula):
                    result = eval(clean_formula)
                    total += float(result)
            except:
                continue

    return total


def _validate_with_accuracy_checks_enhanced(result: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced validation for six-month periods."""
    # Import the shared validation function
    from .budget_parser_utils import _validate_with_accuracy_checks_enhanced as shared_validate
    return shared_validate(result)