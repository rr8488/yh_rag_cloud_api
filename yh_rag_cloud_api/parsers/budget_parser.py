# yh_rag_cloud_api/parsers/budget_parser.py

import csv
import re
from typing import List, Dict, Any, Optional, Tuple


# --- NEW: Helper to parse currency (from milestones_parser) ---
def parse_currency_to_float(text: str) -> float:
    """
    Converts a currency string (e.g., "RM 1,000.00") to a float.
    """
    if not text:
        return 0.0
    # Remove "RM", ",", whitespace, and any text after a newline
    cleaned_text = text.split('\n')[0].strip()
    cleaned_text = re.sub(r'[RM,\s]', '', cleaned_text, flags=re.IGNORECASE)
    try:
        # Handle potential empty strings after cleaning (e.g., "RM")
        if not cleaned_text:
            return 0.0
        return float(cleaned_text)
    except ValueError:
        return 0.0


def _is_numeric(cell: str) -> bool:
    """Check if a string can be interpreted as a number."""
    # Use parse_currency_to_float to check, as it handles "RM 1,000"
    try:
        parse_currency_to_float(cell)
        # Check if the original string contains at least one digit
        return any(char.isdigit() for char in cell)
    except ValueError:
        return False


def _is_header_row(row: List[str]) -> bool:
    """Heuristic to identify if a row is part of the complex header."""
    text_cells = [cell.lower().strip() for cell in row if cell.strip()]
    if not text_cells:
        return False

    keywords = ['q1', 'q2', 'q3', 'q4', 'tranche', 'total', 'amount', '2020', '2021', '2022', '2023', '2024']

    # --- FIX: Regex for date-like strings (Oct'22, Jan-Jun) ---
    date_regex = re.compile(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b')

    # Count how many *cells* contain *any* keyword
    cells_with_keywords = 0
    for cell in text_cells:
        if any(kw in cell for kw in keywords):
            cells_with_keywords += 1
        elif date_regex.search(cell):  # <-- Check for month names
            cells_with_keywords += 1

    # --- END FIX ---

    # A header row should have multiple cells with these keywords
    # (e.g., "Total", "Q1", "Q2"...)
    # This prevents matching "Project ID ... 2020 ..." (1 cell)
    if cells_with_keywords > 2:
        return True

    # Or, the row is *mostly* just keywords (e.g., the year row)
    # Count cells that are *only* keywords (or variations)
    almost_keyword_cells = 0
    for cell in text_cells:
        # Check if cell is just a year or Qx or Tranche
        cleaned_cell = re.sub(r'[^a-zA-Z0-9]', '', cell)
        if cleaned_cell in keywords or cleaned_cell.startswith('q'):
            almost_keyword_cells += 1
        # --- FIX: Also check for date-like cells here ---
        elif date_regex.search(cell):
            almost_keyword_cells += 1
        # --- END FIX ---

    # If > 50% of non-empty cells are simple keywords, it's probably a header
    if text_cells and almost_keyword_cells / len(text_cells) > 0.5:
        return True

    return False


def _is_data_row(row: List[str], header_cols: Dict[int, str], desc_col_idx: int) -> bool:
    """
    A row is a data row if it has a description in the description column
    AND at least one numeric value in a "tranche" or "total" column.
    """
    if not row or desc_col_idx >= len(row) or not row[desc_col_idx].strip():  # Must have a description
        return False

    # Check for numbers in columns that are known headers
    for idx, cell in enumerate(row):
        if idx in header_cols and _is_numeric(cell):
            return True
    return False


def _is_subtotal_row(row: List[str], desc_col_idx: int) -> bool:
    """Check if a row looks like a subtotal row."""
    if desc_col_idx >= len(row):
        return False
    desc = row[desc_col_idx].lower()
    return 'total' in desc or 'sub-total' in desc or 'sub total' in desc


def _parse_header_structure(header_rows: List[List[str]]) -> Tuple[Dict[int, str], int]:
    """
    Parses the multi-row header to create a flat dictionary of {col_idx: "Header Name"}.

    **FIX:** This now "fills" merged cells (like years) across empty columns to
    their right, ensuring keys are unique (e.g., "2020 Q1 Tranche").
    """
    if not header_rows:
        return {}, 0

    num_cols = max(len(r) for r in header_rows)

    # --- FIX: Fill merged cells ---
    filled_header_rows = []
    for row in header_rows:
        filled_row = [""] * num_cols
        last_val = ""
        # Ensure we iterate up to num_cols
        for i in range(num_cols):
            cell_val = row[i].strip() if i < len(row) else ""
            # Logic: if the cell has a value, use it. If not, use the last value.
            # This 'fills' merged cells forward.
            if cell_val:
                last_val = cell_val
            filled_row[i] = last_val
        filled_header_rows.append(filled_row)
    # --- END FIX ---

    # Combine filled header rows into column names
    combined_headers = [""] * num_cols
    for i in range(num_cols):
        parts = []
        for row in filled_header_rows:
            part = row[i]
            # Avoid duplicate parts (e.g., "Tranche" "Tranche")
            if part and part not in parts:
                parts.append(part)
        combined_headers[i] = " ".join(parts)

    # Create the final {col_idx: header_name} map
    header_map = {}
    total_amount_col_idx = -1

    # Find the "description" and "total amount" columns first
    desc_col_idx = 0  # Assume 0 by default

    for i, header_name in enumerate(combined_headers):
        name_lower = header_name.lower()
        if ('total' in name_lower and ('requested' in name_lower or 'amount' in name_lower)) or name_lower == 'total':
            total_amount_col_idx = i
        elif (
                'item' in name_lower or 'description' in name_lower or 'budget' in name_lower or 'proposed' in name_lower) and i < 2:
            desc_col_idx = i

    # Fallback for total amount column if not found
    if total_amount_col_idx == -1:
        for i, header_name in enumerate(combined_headers):
            if 'total' in header_name.lower() and i > desc_col_idx:
                total_amount_col_idx = i
                break

    # Map all columns that are not the description
    for i, header_name in enumerate(combined_headers):
        if i == desc_col_idx:
            continue
        # Special key for total amount
        if i == total_amount_col_idx:
            header_map[i] = "TOTAL_AMOUNT"
            continue

        # Only map columns that look like tranches (have a number or 'Q')
        if re.search(r'(\d|Q\d|Tranche)', header_name, re.IGNORECASE):
            # Clean up the name
            name_parts = header_name.strip().split()
            # Remove duplicates while preserving order
            cleaned_parts = []
            for part in name_parts:
                if part not in cleaned_parts:
                    cleaned_parts.append(part)
            header_map[i] = " ".join(cleaned_parts)

    print(f"  [Parser] Found Description Col: {desc_col_idx}")
    print(f"  [Parser] Found Total Amount Col: {total_amount_col_idx}")
    print(f"  [Parser] Generated Header Map: {header_map}")

    return header_map, desc_col_idx


def parse_budget(csv_path: str) -> Dict[str, Any]:
    """
    Parses a pre-converted CSV file of a Schedule 3 Budget.
    """
    print(f"  [Parser] Starting budget parse for: {csv_path}")

    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        print(f"  [Parser] ERROR: Could not read CSV file: {e}")
        return {"error": f"Failed to read CSV: {e}"}

    metadata = {}
    budget_line_items = []

    # --- 1. Find Metadata (simplified for now) ---
    for row in rows[:15]:  # Search top 15 rows
        for cell in row:
            if "Project Title" in cell:
                metadata["project_title"] = " ".join(cell.split(":")[1:]).strip()
            if "Organisation Name" in cell:
                metadata["organisation_name"] = " ".join(cell.split(":")[1:]).strip()
            if "Total amount requested" in cell:
                metadata["total_amount_requested"] = " ".join(cell.split(":")[1:]).strip()

    # --- 2. Find Header ---
    header_start_row_idx = -1
    header_rows = []

    # Start search from row 10 (to skip metadata)
    for i, row in enumerate(rows[10:], start=10):
        if _is_header_row(row):
            if header_start_row_idx == -1:
                header_start_row_idx = i
            header_rows.append(row)
        elif header_start_row_idx != -1:
            # We found the end of the header block
            break

    if not header_rows:
        print("  [Parser] ERROR: Could not find any header rows.")
        return {"error": "Could not find header rows (heuristic failed)."}

    print(f"  [Parser] Found {len(header_rows)} header rows starting at index {header_start_row_idx}.")

    # --- 3. Parse Header Structure ---
    header_cols, desc_col_idx = _parse_header_structure(header_rows)
    if not header_cols:
        print("  [Parser] ERROR: Failed to parse header structure.")
        return {"error": "Failed to parse header structure."}

    # --- 4. Find and Parse Data Rows ---
    category_l1 = None
    category_l2 = None

    # Start searching for data rows *after* the header block
    data_start_row_idx = header_start_row_idx + len(header_rows)

    for i, row in enumerate(rows[data_start_row_idx:], start=data_start_row_idx):
        if not any(cell.strip() for cell in row) or desc_col_idx >= len(row):
            continue  # Skip empty row

        description = row[desc_col_idx].strip()
        if not description:
            continue  # Skip row with no description

        is_data = _is_data_row(row, header_cols, desc_col_idx)

        # Check if this is a category row
        if not is_data:
            # Check if it's a category (e.g., "Project Implementation Cost")
            # Categories are usually in the first col, by themselves, and not numeric
            if description and all(not cell.strip() for cell in row[desc_col_idx + 1:]):

                # --- REVISED HIERARCHICAL LOGIC ---

                # Check for L2 keywords first, as they are more specific
                # (GSS file uses A. B., others use 'Project Objective')
                if category_l1 and (description.startswith('Project Objective') or \
                                    (description.startswith('A.') or description.startswith('B.'))):
                    category_l2 = description
                    print(f"  [Parser] Set L2: {category_l2}")

                # Check for L1 keywords
                elif (description.isupper() and len(description) < 100) or \
                        description.startswith('Organizational Development Fund') or \
                        "Project Implementation Cost" in description:

                    category_l1 = description
                    category_l2 = None  # Reset L2
                    print(f"  [Parser] Set L1: {category_l1}")

                # Fallback for other L2-like things
                elif category_l1:
                    # Avoid picking up long note text as a category
                    if len(description) < 150:
                        category_l2 = description
                        print(f"  [Parser] Set L2 (Fallback): {category_l2}")
                # --- END REVISED LOGIC ---

            continue  # Not a data row, move on

        # --- Process Data Row ---
        line_item = {
            "category_l1": category_l1,
            "category_l2": category_l2,
            "line_item_description": description,
            "total_amount": 0.0,
            "tranche_breakdown": {},
            "is_subtotal": _is_subtotal_row(row, desc_col_idx),
            "raw_row_index": i
        }

        for col_idx, header_name in header_cols.items():
            if col_idx >= len(row):
                continue

            cell_value_str = row[col_idx]
            cell_value_float = parse_currency_to_float(cell_value_str)

            if cell_value_float == 0.0:
                continue

            if header_name == "TOTAL_AMOUNT":
                line_item["total_amount"] = cell_value_float
            else:
                line_item["tranche_breakdown"][header_name] = cell_value_float

        # Only add if it has some value
        if line_item["total_amount"] > 0 or line_item["tranche_breakdown"]:
            budget_line_items.append(line_item)

    # --- 5. Final Sanity Check ---
    metadata_total = parse_currency_to_float(metadata.get("total_amount_requested", "0"))

    # Try to find the *last* subtotal row, which is usually the grand total
    sum_of_line_items = 0.0
    all_totals = [item["total_amount"] for item in budget_line_items if item["is_subtotal"]]
    if all_totals:
        sum_of_line_items = all_totals[-1]  # Assume last total is grand total
    else:
        # If no subtotal rows, sum non-subtotal items
        sum_of_line_items = sum(
            item["total_amount"] for item in budget_line_items
            if not item["is_subtotal"]
        )

    sanity_checks = {
        "metadata_total": metadata_total,
        "sum_of_line_items": sum_of_line_items,
        "amounts_tally": False,
        "message": ""
    }

    if metadata_total > 0 and abs(metadata_total - sum_of_line_items) < 0.01:
        sanity_checks["amounts_tally"] = True
        sanity_checks["message"] = f"Totals tally. Metadata: {metadata_total}, Sum of parsed items: {sum_of_line_items}"
    else:
        sanity_checks["amounts_tally"] = False
        sanity_checks[
            "message"] = f"Totals MISMATCH. Metadata: {metadata_total}, Sum of parsed items: {sum_of_line_items}"

    return {
        "metadata": metadata,
        "budget_line_items": budget_line_items,
        "sanity_checks": sanity_checks
    }

