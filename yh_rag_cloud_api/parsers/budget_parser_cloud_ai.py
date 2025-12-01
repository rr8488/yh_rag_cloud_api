# yh_rag_cloud_api/parsers/budget_parser_cloud_ai.py

import logging
from typing import Dict, List, Any
from ..parsing_utils import extract_text_with_document_ai
from .budget_parser_cloud_2 import BudgetParserCloud2

logger = logging.getLogger(__name__)


def parse_budget_with_document_ai(file_bytes: bytes,
                                  mime_type: str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet") -> \
Dict[str, Any]:
    """
    Parse budget using Google Document AI for better text extraction.
    """
    try:
        # Extract text using Document AI
        extracted_text = extract_text_with_document_ai(file_bytes, mime_type)

        if not extracted_text or extracted_text.startswith("Error"):
            return {
                "metadata": {},
                "budget_items": [],
                "parser_version": "document_ai",
                "status": "error",
                "error": "Document AI extraction failed"
            }

        # Use our existing parser but with the clean Document AI text
        parser = BudgetParserCloud2()

        # Convert the extracted text to a CSV-like format for the parser
        # Document AI preserves the structure better than CSV conversion
        lines = extracted_text.split('\n')
        csv_like_content = '\n'.join([','.join(line.split('\t')) for line in lines])

        return parser.parse(csv_like_content)

    except Exception as e:
        logger.error(f"Error in Document AI budget parser: {e}")
        return {
            "metadata": {},
            "budget_items": [],
            "parser_version": "document_ai",
            "status": "error",
            "error": str(e)
        }