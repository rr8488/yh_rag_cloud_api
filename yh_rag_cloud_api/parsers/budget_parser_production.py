# yh_rag_cloud_api/parsers/budget_parser_production.py

import logging
import json
import re
from typing import Dict, List, Any
from ..rag_utils import ask_gemini

logger = logging.getLogger(__name__)


class BudgetParserProduction:
    """Production budget parser using Gemini AI for reliable extraction."""

    def __init__(self):
        self.parser_name = "budget_parser_production"

    def parse(self, csv_content: str) -> Dict[str, Any]:
        """
        Parse budget using Gemini AI - the most reliable method.
        """
        try:
            # Limit content length to avoid token limits
            limited_content = csv_content[:15000]  # Conservative limit

            prompt = self._build_prompt(limited_content)
            response = ask_gemini(prompt)

            # Extract JSON from response
            result = self._extract_json_from_response(response)

            # Add parser info
            result["parser_version"] = "production_gemini"
            result["status"] = "success"

            return result

        except Exception as e:
            logger.error(f"Error in production budget parser: {e}")
            return self._create_error_response(str(e))

    def _build_prompt(self, csv_content: str) -> str:
        """Build the prompt for Gemini."""
        return f"""
        Extract budget information from this CSV content and return as valid JSON.

        CRITICAL: Extract these EXACT metadata fields:
        - project_title (string)
        - organization_name (string)  
        - funding_period (original string)
        - funding_period_start (YYYY-MM-DD or "n/a")
        - funding_period_end (YYYY-MM-DD or "n/a") 
        - total_amount_requested (number, extract from "Total amount requested")
        - organization_id (string from "Company ID")
        - project_id (string)

        THEN extract the budget line items.

        **CRITICAL INSTRUCTION FOR CATEGORIES:**
        - Budget items are hierarchically grouped under headers (e.g., "A. Program Costs", "B. Institutional Development").
        - For EACH item, you MUST identify its `category_l1` (the main header) and `category_l2` (the sub-header if exists).
        - Example: If "Flight Ticket" is under "B. Travel", then L1="Program Cost" (inferred) and L2="Travel".

        For each budget item include:
        - category_l1 (string - Main Category e.g. "Program Cost", "Manpower")
        - category_l2 (string - Sub Category e.g. "Travel", "Training" or null)
        - description (string)
        - requested_amount (number or null if no amount)
        - amount_formula (string if Excel formula like "=SUM(...)", otherwise null)

        **CRITICAL SANITY CHECK FIX:**
        - Calculate items_total by summing ALL requested_amount values from budget_items
        - DO NOT use quarterly distributions or column sums for this calculation
        - Only sum the requested_amount field from each budget item

        CSV Content:
        {csv_content}

        Return ONLY valid JSON with this EXACT structure:
        {{
            "metadata": {{
                "project_title": "string",
                "organization_name": "string", 
                "funding_period": "string",
                "funding_period_start": "YYYY-MM-DD or n/a",
                "funding_period_end": "YYYY-MM-DD or n/a", 
                "total_amount_requested": number,
                "organization_id": "string",
                "project_id": "string"
            }},
            "budget_items": [
                {{
                    "category_l1": "string",
                    "category_l2": "string",
                    "description": "string",
                    "requested_amount": number or null,
                    "amount_formula": "string or null"
                }}
            ],
            "sanity_check": {{
                "items_total": number,
                "metadata_total": number, 
                "match": boolean,
                "difference": number
            }}
        }}

        Important rules:
        1. Skip rows that are clearly section headers (contain "Subject:", "Project Objective", "Activity", "List funding") IF they are just wrappers. However, use them to derive the 'category' before skipping.
        2. For funding period "2020 - 2022", use start "2020-01-01" and end "2022-12-31"
        3. For amounts with commas like "RM1,000,000", convert to 1000000
        4. If a field is not found, use empty string "" for strings and null for numbers
        """

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract and validate JSON from Gemini response."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in Gemini response")

            result = json.loads(json_match.group())

            # Validate required structure
            if "metadata" not in result or "budget_items" not in result:
                raise ValueError("Invalid JSON structure from Gemini")

            # Ensure all required metadata fields exist
            required_metadata = [
                "project_title", "organization_name", "funding_period",
                "funding_period_start", "funding_period_end", "total_amount_requested",
                "organization_id", "project_id"
            ]

            for field in required_metadata:
                if field not in result["metadata"]:
                    result["metadata"][field] = "" if "id" not in field else ""

            # --- NEW: Ensure categories exist in items ---
            for item in result["budget_items"]:
                if "category_l1" not in item: item["category_l1"] = "Uncategorized"
                if "category_l2" not in item: item["category_l2"] = None
            # ---------------------------------------------

            # Clean up budget items - remove null amount items that are likely headers
            result["budget_items"] = self._filter_real_budget_items(result["budget_items"])

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise ValueError(f"Invalid JSON from Gemini: {e}")

    def _filter_real_budget_items(self, items: List[Dict]) -> List[Dict]:
        """Filter out items that are likely section headers rather than real budget items."""
        real_items = []
        header_keywords = [
            "subject:", "project objective", "activity", "list funding",
            "organizational development fund", "note:"
        ]

        for item in items:
            description = item.get("description", "").lower()
            amount = item.get("requested_amount")

            # Skip if it's clearly a header and has no amount
            if any(keyword in description for keyword in header_keywords) and amount is None:
                continue

            real_items.append(item)

        return real_items

    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "metadata": {},
            "budget_items": [],
            "parser_version": "production_gemini",
            "status": "error",
            "error": error_msg
        }


def parse_budget_production(csv_content: str) -> Dict[str, Any]:
    """
    Main production function for budget parsing.

    Args:
        csv_content: CSV content as string from converted Excel file

    Returns:
        Dictionary containing parsed metadata and budget items
    """
    parser = BudgetParserProduction()
    return parser.parse(csv_content)