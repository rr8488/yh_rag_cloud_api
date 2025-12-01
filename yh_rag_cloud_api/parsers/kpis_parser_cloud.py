# yh_rag_cloud_api/parsers/kpis_parser_cloud.py

import logging
import json
import re
from typing import Dict, List, Any, Optional
from ..rag_utils import ask_gemini

logger = logging.getLogger(__name__)


class KPIsParserCloud:
    """Cloud-based KPIs parser with completely generic instructions."""

    def __init__(self):
        self.parser_name = "kpis_parser_cloud"

    def parse(self, csv_content: str) -> Dict[str, Any]:
        """
        Parse KPIs and milestones using completely generic instructions.
        """
        try:
            # Step 1: Minimal pre-processing
            processed_content = self._minimal_preprocess(csv_content)

            logger.info(
                f"Content prepared: {len(processed_content)} chars from original {len(csv_content)}")

            # Step 2: LLM parsing with completely generic instructions
            prompt = self._build_improved_prompt(processed_content)
            response = ask_gemini(prompt)

            # Step 3: Extract and validate JSON
            result = self._extract_json_from_response(response)

            # Step 4: Add parser metadata
            result["parser_version"] = "production_gemini"
            result["status"] = "success"

            return result

        except Exception as e:
            logger.error(f"Error in cloud KPIs parser: {e}")
            return self._create_error_response(str(e))

    def _minimal_preprocess(self, csv_content: str) -> str:
        """Minimal preprocessing - just remove excessive empty lines."""
        try:
            lines = csv_content.split('\n')
            # Remove completely empty lines but preserve all content
            processed_lines = [line for line in lines if line.strip()]

            # Keep reasonable limit but preserve structure
            if len(processed_lines) > 150:
                processed_lines = processed_lines[:150]
                logger.warning(f"Truncated to 150 lines for token management")

            result = '\n'.join(processed_lines)
            return result

        except Exception as e:
            logger.warning(f"Minimal pre-processing failed: {e}")
            return csv_content

    def _build_generic_prompt(self, processed_content: str) -> str:
        """Build completely generic prompt with no examples or specific guidance."""
        return f"""
        Analyze the provided content and extract information according to the specified JSON structure.

        Content to analyze:
        {processed_content}

        Extract all available information and return it in this exact JSON format:
        {{
            "metadata": {{
                "organization_name": "value or null",
                "company_id": "value or null", 
                "project_id": "value or null",
                "project_title": "value or null",
                "total_grant_amount": "number or null",
                "funding_period": "value or null",
                "funding_period_start": "value or null",
                "funding_period_end": "value or null"
            }},
            "milestones": [
                {{
                    "tranche_number": "number",
                    "disbursement_date": "value or null",
                    "tranche_amount": "number or null",
                    "percentage_of_total": "number or null",
                    "kpis": [
                        {{
                            "kpi_description": "value",
                            "deliverable_type": "value or null",
                            "target_beneficiaries": {{
                                "quantity": "number or null",
                                "unit": "value or null",
                                "description": "value or null"
                            }},
                            "success_criteria": "value or null"
                        }}
                    ]
                }}
            ]
        }}

        Instructions:
        - Extract information directly from the content
        - For beneficiary information, extract both numerical quantities and their corresponding units
        - Use null for any information not found in the content
        - Return only valid JSON without any additional text
        """

    def _build_improved_prompt(self, processed_content: str) -> str:
        """Build prompt with minimal structural guidance."""
        return f"""
        Analyze the provided content and extract information according to the specified JSON structure.

        Content to analyze:
        {processed_content}

        Extract all available information and return it in this exact JSON format:
        {{
            "metadata": {{
                "organization_name": "value or null",
                "company_id": "value or null", 
                "project_id": "value or null",
                "project_title": "value or null",
                "total_grant_amount": "number or null",
                "funding_period": "value or null",
                "funding_period_start": "value or null",
                "funding_period_end": "value or null"
            }},
            "milestones": [
                {{
                    "tranche_number": "number",
                    "disbursement_date": "value or null",
                    "tranche_amount": "number or null",
                    "percentage_of_total": "number or null",
                    "kpis": [
                        {{
                            "kpi_description": "value",
                            "deliverable_type": "classify the activity type",
                            "target_beneficiaries": {{
                                "quantity": "count of people/organizations (not percentages)",
                                "unit": "type of beneficiary",
                                "description": "additional details"
                            }},
                            "success_criteria": "performance targets or completion requirements"
                        }}
                    ]
                }}
            ]
        }}

        Data type guidance:
        - target_beneficiaries.quantity: Use actual counts (200, 48, 18) not percentages
        - target_beneficiaries.unit: Use the beneficiary type (students, teachers, schools, etc.)
        - deliverable_type: Classify the main activity, not the documentation requirements
        - success_criteria: Extract performance thresholds and completion requirements

        Instructions:
        - Extract information directly from the content
        - Use null for any information not found
        - Return only valid JSON without any additional text
        """

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract and validate JSON from Gemini response."""
        try:
            # Clean the response
            cleaned_response = re.sub(r'```json\s*|\s*```', '', response.strip())

            # Try to find JSON
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if not json_match:
                logger.error(f"No JSON found in response. Response: {response[:500]}...")
                raise ValueError("No JSON found in Gemini response")

            json_str = json_match.group()
            result = json.loads(json_str)

            # Log extraction results
            milestones_count = len(result.get('milestones', []))
            kpis_count = sum(len(m.get('kpis', [])) for m in result.get('milestones', []))

            logger.info(f"Extracted {milestones_count} milestones with {kpis_count} KPIs")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Problematic JSON: {json_str[:500] if 'json_str' in locals() else 'N/A'}")
            raise ValueError(f"Invalid JSON from Gemini: {e}")

    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "metadata": {},
            "milestones": [],
            "parser_version": "production_gemini",
            "status": "error",
            "error": error_msg
        }


def kpis_parser_cloud(csv_content: str) -> Dict[str, Any]:
    """
    Completely generic KPIs parser with no biased instructions.

    Args:
        csv_content: CSV content as string from converted Excel file

    Returns:
        Dictionary containing parsed metadata and milestones
    """
    parser = KPIsParserCloud()
    return parser.parse(csv_content)