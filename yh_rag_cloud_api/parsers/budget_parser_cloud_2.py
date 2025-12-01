# yh_rag_cloud_api/parsers/budget_parser_cloud_2.py

import re
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import io
from datetime import datetime

logger = logging.getLogger(__name__)


class BudgetParserCloud2:
    """Enhanced budget parser that extracts metadata and budget items from Excel files."""

    def __init__(self):
        self.parser_name = "budget_parser_cloud_2"

    def parse(self, csv_content: str) -> Dict[str, Any]:
        """
        Parse CSV content from budget Excel files to extract metadata and budget items.

        Args:
            csv_content: CSV content as string from converted Excel file

        Returns:
            Dictionary containing parsed metadata and budget items
        """
        try:
            # Read CSV into pandas for better structure handling
            df = pd.read_csv(io.StringIO(csv_content), header=None)

            # Extract metadata using regex from the entire CSV content
            metadata = self._extract_metadata_with_regex(csv_content)

            # Extract budget items from the structured table
            budget_items = self._extract_budget_items_from_dataframe(df)

            # Perform sanity check
            self._sanity_check(metadata, budget_items)

            return {
                "metadata": metadata,
                "budget_items": budget_items,
                "parser_version": "2.0",
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error in budget parser v2: {e}")
            return {
                "metadata": {},
                "budget_items": [],
                "parser_version": "2.0",
                "status": "error",
                "error": str(e)
            }

    def _extract_metadata_with_regex(self, csv_content: str) -> Dict[str, Any]:
        """Extract metadata using regex patterns from the CSV content."""
        metadata = {}

        # Clean the content by removing empty quoted strings
        cleaned_content = re.sub(r',\"\"', ',', csv_content)  # Remove empty quoted columns
        cleaned_content = re.sub(r',,+,', ',', cleaned_content)  # Remove multiple consecutive commas

        # Project Title pattern - stop at comma or newline
        project_title_match = re.search(
            r'Project\s*Title\s*[:\-]*\s*([^,\n\r]+)',
            cleaned_content,
            re.IGNORECASE
        )
        if project_title_match:
            metadata["project_title"] = self._clean_metadata_value(project_title_match.group(1))

        # Organisation Name pattern - stop at comma or newline
        org_name_match = re.search(
            r'Organi[sz]ation\s*Name\s*[:\-]*\s*([^,\n\r]+)',
            cleaned_content,
            re.IGNORECASE
        )
        if org_name_match:
            metadata["organization_name"] = self._clean_metadata_value(org_name_match.group(1))

        # Funding Period pattern - stop at comma or newline
        funding_period_match = re.search(
            r'Funding\s*Period\s*[:\-]*\s*([^,\n\r]+)',
            cleaned_content,
            re.IGNORECASE
        )
        if funding_period_match:
            period_text = self._clean_metadata_value(funding_period_match.group(1))
            start_date, end_date = self._parse_funding_period(period_text)
            metadata["funding_period"] = period_text
            metadata["funding_period_start"] = start_date
            metadata["funding_period_end"] = end_date

        # Total Amount Requested pattern - SPECIAL HANDLING FOR COMMAS IN AMOUNTS
        amount_requested_match = re.search(
            r'Total\s*amount\s*requested\s*[:\-]*\s*([^\n\r]+?)(?=,?\s*\n|,?\s*\r|,?\s*$|,?\s*[A-Za-z]+\s*:)',
            cleaned_content,
            re.IGNORECASE
        )
        if amount_requested_match:
            amount_str = self._clean_metadata_value(amount_requested_match.group(1))
            amount = self._extract_amount(amount_str)
            if amount is not None:
                metadata["total_amount_requested"] = amount

        # Amount Approved pattern (alternative) - SPECIAL HANDLING FOR COMMAS IN AMOUNTS
        amount_approved_match = re.search(
            r'Amount\s*Approved\s*[:\-]*\s*([^\n\r]+?)(?=,?\s*\n|,?\s*\r|,?\s*$|,?\s*[A-Za-z]+\s*:)',
            cleaned_content,
            re.IGNORECASE
        )
        if amount_approved_match:
            amount_str = self._clean_metadata_value(amount_approved_match.group(1))
            amount = self._extract_amount(amount_str)
            if amount is not None:
                metadata["amount_approved"] = amount

        # Company ID pattern - stop at comma or newline
        company_id_match = re.search(
            r'Company\s*ID\s*[:\-]*\s*([^,\n\r]+)',
            cleaned_content,
            re.IGNORECASE
        )
        if company_id_match:
            metadata["organization_id"] = self._clean_metadata_value(company_id_match.group(1))

        # Project ID pattern - stop at comma or newline
        project_id_match = re.search(
            r'Project\s*ID\s*[:\-]*\s*([^,\n\r]+)',
            cleaned_content,
            re.IGNORECASE
        )
        if project_id_match:
            metadata["project_id"] = self._clean_metadata_value(project_id_match.group(1))

        return metadata

    def _clean_metadata_value(self, value: str) -> str:
        """Clean metadata value by removing quotes, extra spaces, and empty CSV artifacts."""
        # Remove surrounding quotes
        cleaned = value.strip().strip('"')
        # Remove any trailing commas or empty CSV artifacts
        cleaned = re.sub(r',.*$', '', cleaned)  # Remove everything after first comma
        # Remove any remaining quotes
        cleaned = cleaned.replace('"', '')
        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _parse_funding_period(self, period_text: str) -> Tuple[str, str]:
        """Parse funding period text into start and end dates."""
        # ... (same as before)

    def _get_first_day_of_month(self, month: str, year: str) -> str:
        """Get first day of month in YYYY-MM-DD format."""
        # ... (same as before)

    def _get_last_day_of_month(self, month: str, year: str) -> str:
        """Get last day of month in YYYY-MM-DD format."""
        # ... (same as before)

    def _month_to_number(self, month: str) -> int:
        """Convert month name to number."""
        # ... (same as before)

    def _parse_month_year(self, text: str, last_day: bool = False) -> str:
        """Parse month year text into date string."""
        # ... (same as before)

    def _extract_budget_items_from_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract budget items from the structured table in DataFrame."""
        # ... (same as before)

    def _extract_amount(self, amount_str: str) -> Optional[float]:
        """Extract numeric amount from string, handling various formats."""
        if not amount_str or amount_str.isspace():
            return None

        # Remove currency symbols but KEEP COMMAS for now
        cleaned = re.sub(r'[RM\$]', '', amount_str.strip()).strip()

        # Handle Excel formulas
        if '=sum' in cleaned.lower() or '=' in cleaned:
            # Try to extract numbers from formulas like =SUM(D22:S22)
            numbers = re.findall(r'\d+', cleaned)
            if numbers:
                # Convert to integers and sum them for formulas with multiple numbers
                numbers_int = [int(num) for num in numbers]
                return float(sum(numbers_int))
            return None

        # Try to convert to float - now handle commas properly
        try:
            # Remove commas for conversion but preserve the number
            cleaned = cleaned.replace(',', '')
            return float(cleaned)
        except ValueError:
            return None

    def _sanity_check(self, metadata: Dict[str, Any], budget_items: List[Dict[str, Any]]) -> None:
        """Perform sanity check to ensure budget items total matches metadata total."""
        # ... (same as before)


def parse_budget_cloud_2(csv_content: str) -> Dict[str, Any]:
    """
    Main function to parse budget data using the enhanced cloud parser.

    Args:
        csv_content: CSV content as string from converted Excel file

    Returns:
        Dictionary containing parsed metadata and budget items
    """
    parser = BudgetParserCloud2()
    return parser.parse(csv_content)