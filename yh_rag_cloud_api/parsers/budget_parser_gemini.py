# yh_rag_cloud_api/parsers/budget_parser_gemini.py

import re
import logging
import json
from typing import Dict, List, Any
from ..rag_utils import ask_gemini

logger = logging.getLogger(__name__)


def parse_budget_with_gemini(csv_content: str) -> Dict[str, Any]:
    """
    Parse budget using Gemini AI for intelligent extraction.
    """
    try:
        prompt = f"""
        Extract budget information from this CSV content and return as JSON.

        Extract these metadata fields:
        - project_title
        - organization_name  
        - funding_period (convert to start_date and end_date in YYYY-MM-DD format)
        - total_amount_requested (as number)
        - organization_id
        - project_id

        Then extract budget items with:
        - description
        - requested_amount (as number)
        - amount_formula (if it contains Excel formula)

        Finally, perform a sanity check to ensure the sum of budget items matches total_amount_requested.

        CSV Content:
        {csv_content[:10000]}  # Limit content length

        Return valid JSON only with this structure:
        {{
            "metadata": {{
                "project_title": "string",
                "organization_name": "string", 
                "funding_period": "string",
                "funding_period_start": "YYYY-MM-DD",
                "funding_period_end": "YYYY-MM-DD", 
                "total_amount_requested": number,
                "organization_id": "string",
                "project_id": "string"
            }},
            "budget_items": [
                {{
                    "description": "string",
                    "requested_amount": number,
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
        """

        response = ask_gemini(prompt)

        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            result["parser_version"] = "gemini_ai"
            result["status"] = "success"
            return result
        else:
            raise ValueError("No JSON found in Gemini response")

    except Exception as e:
        logger.error(f"Error in Gemini budget parser: {e}")
        return {
            "metadata": {},
            "budget_items": [],
            "parser_version": "gemini_ai",
            "status": "error",
            "error": str(e)
        }