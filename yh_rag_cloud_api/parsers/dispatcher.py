from typing import Callable, Dict, Any

# Import your specific parser functions from their respective modules
# Assuming they are in the same directory. Adjust the path if needed.
from .proposal_parser import parse as parse_proposal
from .budget_parser import parse as parse_budget
from .grant_agreement_parser_production import parse as parse_grant_agreement
from .kpis_parser_production import parse as parse_milestones
from .schedule4_parser import parse as parse_schedule4

# Define a type for our parser functions for better type hinting
# Each parser should accept a file path and return a dictionary of results.
ParserFunction = Callable[[str], Dict[str, Any]]

# 1. Create a mapping from a document type string to the parser function.
# This is the core of the dispatcher pattern. It's easy to add new parsers here.
PARSER_MAPPING: Dict[str, ParserFunction] = {
    "proposal": parse_proposal,
    "budget": parse_budget,
    "grant_agreement": parse_grant_agreement,
    "milestones": parse_milestones,
    "schedule4": parse_schedule4,
}


def dispatch_parser(document_type: str, file_path: str) -> Dict[str, Any]:
    """
    Selects and executes the correct parser based on the document type.

    Args:
        document_type: A string identifier for the type of document
                       (e.g., 'proposal', 'budget').
        file_path: The local path to the document to be parsed.

    Returns:
        A dictionary containing the parsed data from the document.

    Raises:
        ValueError: If the document_type is not supported.
    """
    print(f"  [Dispatcher] Received request for document type: '{document_type}'")
    parser_func = PARSER_MAPPING.get(document_type)

    if not parser_func:
        raise ValueError(f"Unsupported document type: '{document_type}'. No parser found.")

    print(f"  [Dispatcher] Invoking parser: {parser_func.__name__}")
    return parser_func(file_path)