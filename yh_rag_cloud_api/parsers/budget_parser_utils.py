# yh_rag_cloud_api/parsers/budget_parser_utils.py

import re
import json
from typing import Dict, Any, List


def extract_json_from_response_enhanced(response: str) -> Dict[str, Any]:
    """Enhanced JSON extraction - SHARED"""
    strategies = [
        lambda: json.loads(response),
        lambda: json.loads(re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL).group(1)),
        lambda: json.loads(re.search(r'```\s*(.*?)\s*```', response, re.DOTALL).group(1)),
        lambda: json.loads(re.search(r'\{[\s\S]*\}', response).group()),
        lambda: json.loads(clean_json_response_enhanced(response)),
    ]

    for i, strategy in enumerate(strategies):
        try:
            result = strategy()
            print(f"  [JSON Extraction] Successful with strategy {i + 1}")
            return result
        except Exception:
            continue

    return {"error": "Could not extract valid JSON from AI response"}


def clean_json_response_enhanced(text: str) -> str:
    """Enhanced JSON cleaning - SHARED"""
    if not text:
        return "{}"

    text = re.sub(r'^[^{]*', '', text)
    text = re.sub(r'[^}]*$', '', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = re.sub(r'(\w+)\s*:', r'"\1":', text)
    text = re.sub(r':\s*\'([^\']*)\'', r': "\1"', text)

    return text.strip()


def _validate_with_accuracy_checks_enhanced(result: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced accuracy validation - SHARED"""
    result = _validate_budget_result_enhanced(result)

    # Accuracy analysis
    mismatch_count = 0
    total_mismatch = 0
    items_with_issues = []

    for i, item in enumerate(result.get("budget_line_items", [])):
        if not item.get("is_subtotal", False):
            total_amount = item.get("total_amount", 0)
            breakdown_sum = sum(item.get("tranche_breakdown", {}).values())

            # Check breakdown vs total
            if abs(total_amount - breakdown_sum) > 1.0:
                mismatch_count += 1
                mismatch_amount = abs(total_amount - breakdown_sum)
                total_mismatch += mismatch_amount

                item["accuracy_note"] = f"Breakdown mismatch: total={total_amount}, sum={breakdown_sum}"
                items_with_issues.append({
                    "index": i,
                    "description": item.get("line_item_description", "")[:100],
                    "total_amount": total_amount,
                    "breakdown_sum": breakdown_sum,
                    "difference": mismatch_amount
                })
            else:
                item["accuracy_note"] = "Breakdown matches total"

    # Add comprehensive accuracy report
    result["accuracy_report"] = {
        "items_with_mismatch": mismatch_count,
        "total_mismatch_amount": total_mismatch,
        "validation_passed": mismatch_count == 0,
        "mismatch_percentage": f"{(mismatch_count / max(len(result['budget_line_items']), 1)) * 100:.1f}%",
        "total_items": len(result['budget_line_items']),
        "pmf_odf_breakdown": {
            "pmf_items": len([item for item in result["budget_line_items"] if item.get("budget_type") == "PMF"]),
            "odf_items": len([item for item in result["budget_line_items"] if item.get("budget_type") == "ODF"])
        },
        "items_with_issues": items_with_issues[:5]
    }

    print(f"  [Accuracy Check] {mismatch_count}/{len(result['budget_line_items'])} mismatches")
    return result


def _validate_budget_result_enhanced(result: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced validation with dynamic quarter awareness - SHARED"""
    if not isinstance(result, dict):
        result = {}

    result.setdefault("metadata", {})
    result.setdefault("budget_line_items", [])
    result.setdefault("sanity_checks", {})

    # Get expected quarters from quarter structure
    quarter_structure = result.get("metadata", {}).get("quarter_structure", {})
    expected_quarters = quarter_structure.get("quarter_labels", ["Q1", "Q2", "Q3", "Q4"])

    # Clean and validate items
    cleaned_items = []
    for item in result.get("budget_line_items", []):
        if not isinstance(item, dict):
            continue

        description = str(item.get("line_item_description", "")).strip().lower()
        if any(keyword in description for keyword in ["total", "jumlah", "sum", "overall", "approved", "breakdown"]):
            continue

        cleaned_item = {
            "category": str(item.get("category", "Unknown")).strip(),
            "subcategory": str(item.get("subcategory", "")).strip(),
            "line_item_description": str(item.get("line_item_description", "")).strip(),
            "total_amount": safe_float(item.get("total_amount")),
            "tranche_breakdown": _clean_tranche_breakdown_dynamic(item.get("tranche_breakdown", {}), expected_quarters),
            "is_subtotal": bool(item.get("is_subtotal", False)),
            "budget_type": str(item.get("budget_type", "PMF")).strip().upper()
        }

        if (cleaned_item["line_item_description"] or
                cleaned_item["total_amount"] > 0 or
                any(val > 0 for val in cleaned_item["tranche_breakdown"].values())):
            cleaned_items.append(cleaned_item)

    result["budget_line_items"] = cleaned_items

    # Calculate sanity checks
    metadata_total = parse_currency_to_float(result["metadata"].get("total_amount_requested", "0"))
    sum_line_items = sum(item.get("total_amount", 0) for item in result["budget_line_items"])

    result["sanity_checks"] = {
        "metadata_total": metadata_total,
        "sum_of_line_items": sum_line_items,
        "amounts_tally": abs(metadata_total - sum_line_items) < 10000,
        "line_item_count": len(result["budget_line_items"]),
        "message": f"Metadata: RM{metadata_total:,.2f}, Line Items: RM{sum_line_items:,.2f}, Items: {len(result['budget_line_items'])}, Quarters: {len(expected_quarters)}"
    }

    return result


def _clean_tranche_breakdown_dynamic(breakdown: Dict, expected_quarters: List[str]) -> Dict[str, float]:
    """Ensure breakdown matches expected quarter structure - SHARED"""
    if not isinstance(breakdown, dict):
        return {q: 0.0 for q in expected_quarters}

    cleaned = {}
    for quarter in expected_quarters:
        cleaned[quarter] = safe_float(breakdown.get(quarter, 0.0))

    return cleaned


def safe_float(value) -> float:
    """Safely convert any value to float - SHARED"""
    if value is None:
        return 0.0
    try:
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            return parse_currency_to_float(value)
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0


def parse_currency_to_float(currency_str: str) -> float:
    """Convert currency string to float - SHARED"""
    if not currency_str:
        return 0.0
    text = str(currency_str).strip()
    text = re.sub(r'^=.*\(|\)$', '', text)
    text = re.sub(r'[RM\$\£\€\¥\s,]', '', text, flags=re.IGNORECASE)
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        numbers = re.findall(r'\d+\.?\d*', text)
        return float(numbers[0]) if numbers else 0.0