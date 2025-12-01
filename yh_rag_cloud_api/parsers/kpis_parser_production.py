# parsers/kpis_parser_production.py
import csv
import re
from typing import List, Dict, Any


# --- NEW: Helper to parse currency ---
def parse_currency_to_float(text: str) -> float:
    if not text:
        return 0.0
    # Remove "RM", ",", and any text after a newline
    cleaned_text = text.split('\n')[0].strip()
    cleaned_text = re.sub(r'[RM,\s]', '', cleaned_text, flags=re.IGNORECASE)
    try:
        return float(cleaned_text)
    except ValueError:
        return 0.0


# Regex to find (SOD: ...)
# --- FIX: Made the final closing parenthesis optional `\)?` ---
# This handles truncated data where the (SOD:...) block is not closed.
SOD_REGEX = re.compile(r'\((?:[*\s]*)?SOD\s*:\s*([^()]*(\([^()]*\)[^()]*)*)\s*\)?', re.DOTALL | re.IGNORECASE)
# --- END FIX ---

# Regex to find date-like cells
DATE_REGEX = re.compile(
    r'^\d{4}-\d{2}-\d{2}'  # 2020-02-01...
    r'|^\w{3}-\d{2}'  # Feb-20...
    r'|^(January|February|March|April|May|June|July|August|September|October|November|December)'  # October
    , re.IGNORECASE)

# Regex to find a year (e.g., '2022' in '2022 Budget')
YEAR_REGEX = re.compile(r'\b(20\d{2})\b')  # e.g., 2020, 2021

# --- NEW: Regex for Duration and Dates ---
DURATION_REGEX = re.compile(r'(\d+)\s+months', re.IGNORECASE)
DATES_REGEX = re.compile(r'\(([\w\s]+ \d{4})\s*-\s*([\w\s\n]+ \d{4})\)', re.IGNORECASE)


def is_date_like(cell: str) -> bool:
    """Checks if a cell string matches known date patterns."""
    return bool(DATE_REGEX.match(cell.strip()))


def find_anchor_row(rows: List[List[str]], content_check: callable, start_row=5, end_row=20) -> int:
    """
    Finds a row index based on which row has the *most* cells matching a check.
    """
    best_row_index = -1
    max_count = 0
    search_end = min(len(rows), end_row)
    search_start = min(start_row, search_end)

    for i, row in enumerate(rows[search_start:search_end]):
        row_index = search_start + i
        count = sum(1 for cell in row if content_check(cell))

        if count > max_count and count >= 2:
            max_count = count
            best_row_index = row_index

    return best_row_index


def parse_milestones(csv_file_path: str) -> Dict[str, Any]:
    """
    Parses a specific CSV (Schedule 1) file format dynamically by
    finding anchor rows for dates, headers, and extracting metadata.

    Returns a dictionary with metadata, sanity checks, and milestones.
    """
    try:
        with open(csv_file_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: Parser could not find file {csv_file_path}")
        return {}
    except Exception as e:
        print(f"Error: Parser failed to read CSV {csv_file_path}: {e}")
        return {}

    # --- 1. Extract Basic Metadata (Org Name, Company ID) ---
    metadata = {
        "organization_name": "N/A",
        "company_id": "N/A",
        "total_grant_value": "N/A",
        "total_grant_numeric": 0.0,
        "duration_months": "N/A",
        "start_date": "N/A",
        "end_date": "N/A"
    }

    # --- FIX: Look for key:value in the *same cell* ---
    for row in rows[:15]:  # Search top 15 rows
        if not row or len(row) < 1:  # Only need to check first cell
            continue

        cell_a = row[0].strip()
        if ":" in cell_a:
            parts = cell_a.split(':', 1)
            key = parts[0].strip().lower()
            value = parts[1].strip()

            if key == "organisation name":
                metadata["organization_name"] = value
            elif key == "company id":
                metadata["company_id"] = value
    # --- END FIX ---

    # --- 2. Find Anchor Rows Dynamically ---
    date_row_index = find_anchor_row(rows, is_date_like, start_row=5, end_row=25)
    if date_row_index == -1:
        print("Milestone parser: Could not find a valid 'Date Row'.")
        return {}

    header_row_index = find_anchor_row(rows, lambda cell: cell.strip().startswith("Tranche Amount"),
                                       start_row=date_row_index, end_row=date_row_index + 5)
    if header_row_index == -1:
        print("Milestone parser: Could not find a valid 'Header Row'.")
        return {}

    year_row_index = find_anchor_row(rows, lambda cell: bool(YEAR_REGEX.search(cell)),
                                     start_row=max(0, date_row_index - 5), end_row=date_row_index)
    year_row = rows[year_row_index] if year_row_index != -1 else []

    data_start_row_index = header_row_index + 1
    if len(rows) <= data_start_row_index:
        print("Milestone parser: No data rows found after header row.")
        return {}

    print(f"  [Parser] Found Date Row: {date_row_index + 1}")
    if year_row:
        print(f"  [Parser] Found Year Row: {year_row_index + 1}")
    print(f"  [Parser] Found Header Row: {header_row_index + 1}")
    print(f"  [Parser] Found Data Start Row: {data_start_row_index + 1}")

    date_row = rows[date_row_index]
    header_row = rows[header_row_index]

    # --- 3. Extract Grant Total, Duration, and Dates ---
    first_data_row = rows[data_start_row_index]
    if len(first_data_row) > 1:
        # Get Total Grant Value
        # This assumes total grant is in column B of the first data row
        total_grant_str = first_data_row[1].split('\n')[0].strip()
        metadata["total_grant_value"] = total_grant_str
        metadata["total_grant_numeric"] = parse_currency_to_float(total_grant_str)

        # --- FIX: Search for Duration/Dates in a more robust way ---
        # The duration/dates can be in EITHER the first data row (A13) or the second (A14)

        # Cell 1: Try first_data_row[0]
        duration_cell_1 = first_data_row[0]

        # Cell 2: Try the row *below* it, if it exists
        duration_cell_2 = ""
        if len(rows) > data_start_row_index + 1 and len(rows[data_start_row_index + 1]) > 0:
            duration_cell_2 = rows[data_start_row_index + 1][0]

        # Combine them to search
        # This handles cases where "36 months" is in one cell and "(Feb 2020...)" is in the cell below it.
        combined_duration_text = (duration_cell_1 + " " + duration_cell_2).replace('\n', ' ')

        # Try to find duration in the combined text
        duration_match = DURATION_REGEX.search(combined_duration_text)
        if duration_match:
            metadata["duration_months"] = duration_match.group(1)

        # Try to find dates in the combined text
        dates_match = DATES_REGEX.search(combined_duration_text)
        if dates_match:
            metadata["start_date"] = dates_match.group(1).strip()
            # Clean up potential newlines in the end date
            end_date_str = re.sub(r'\s+', ' ', dates_match.group(2)).strip()

            # --- NEW OVERRIDE LOGIC ---
            # Check for a stray year *after* the closing parenthesis
            # e.g., "(Feb 2020 - Jan 2023) 2024"
            end_index_of_match = dates_match.end()
            remaining_text = combined_duration_text[end_index_of_match:]

            # Search for a year (e.g., "2024") immediately following
            # We strip() and check the first 10 chars to be safe.
            override_year_match = YEAR_REGEX.search(remaining_text.strip()[:10])

            if override_year_match:
                override_year = override_year_match.group(1)
                # Replace the year in the *original* end date
                # e.g., "Jan 2023" -> "Jan 2024"
                metadata["end_date"] = re.sub(YEAR_REGEX, override_year, end_date_str)
                print(f"  [Parser] Found end date override. Changed '{end_date_str}' to '{metadata['end_date']}'")
            else:
                metadata["end_date"] = end_date_str
            # --- END OVERRIDE LOGIC ---

        # --- END FIX ---

    milestone_cols = []

    # --- 4. Find all Milestone Columns ---
    for j, header_cell in enumerate(header_row):
        if header_cell.strip().startswith("Tranche Amount"):
            date = "Date N/A"
            if j < len(date_row):
                date_cell_val = date_row[j].strip()
                if is_date_like(date_cell_val):
                    if not re.search(r'\d', date_cell_val):
                        found_year = ""
                        if year_row:
                            for k in range(j, -1, -1):
                                if k < len(year_row):
                                    year_match = YEAR_REGEX.search(year_row[k])
                                    if year_match:
                                        found_year = year_match.group(1)
                                        break
                        if found_year:
                            date = f"{date_cell_val} {found_year}"
                        else:
                            date = date_cell_val
                    else:
                        date = date_cell_val

            milestone_cols.append({
                "date": date,
                "amount_col": j,
                "deliverable_col": j + 1,
                "amount": "",
                "deliverables": []
            })

    if not milestone_cols:
        print("Milestone parser: No milestone columns found based on headers.")
        return {}

    # --- 5. Process Data Rows ---

    total_tranche_amount = 0.0

    # Get Amounts from the FIRST data row
    for m_info in milestone_cols:
        col_idx = m_info["amount_col"]
        if col_idx < len(first_data_row):
            amount_cell = first_data_row[col_idx].strip()
            if amount_cell:
                amount_str = amount_cell.split('\n')[0]
                m_info["amount"] = amount_str
                # Add to total for sanity check
                total_tranche_amount += parse_currency_to_float(amount_str)

    # Get Deliverables from ALL data rows
    for i, row in enumerate(rows):
        if i < data_start_row_index:
            continue

        # --- NEW: Stop processing if we hit the footer table ---
        if row and len(row) > 0 and row[0].strip().lower() == "tranches":
            print("  [Parser] Footer 'Tranches' row detected. Stopping deliverable processing.")
            break
        # --- END NEW ---

        for m_info in milestone_cols:
            col_idx = m_info["deliverable_col"]

            if col_idx < len(row):
                deliverable_text = row[col_idx].strip()

                if deliverable_text:
                    sod = "N/A"
                    sod_match = SOD_REGEX.search(deliverable_text)
                    if sod_match:
                        sod = sod_match.group(1).strip()
                        deliverable_text = SOD_REGEX.sub('', deliverable_text).strip()

                    if not any(d['description'] == deliverable_text for d in m_info["deliverables"]):
                        m_info["deliverables"].append({
                            "description": deliverable_text,
                            "sod": sod
                        })

    final_milestones = [
        m for m in milestone_cols if
        (m["amount"] and m.get("amount").strip() != "Tranche Amount (RM)") or m["deliverables"]
    ]

    # --- 6. Final Sanity Check ---
    sanity_checks = {
        "total_tranche_amount": total_tranche_amount,
        "amounts_tally": False,
        "message": "N/A"
    }

    total_grant_numeric = metadata["total_grant_numeric"]

    # Use a small tolerance for float comparison
    if abs(total_grant_numeric - total_tranche_amount) < 0.01:
        sanity_checks["amounts_tally"] = True
        sanity_checks[
            "message"] = f"Amounts tally. Total Grant: {total_grant_numeric}, Sum of Tranches: {total_tranche_amount}"
    else:
        sanity_checks["amounts_tally"] = False
        if total_grant_numeric == 0.0 and total_tranche_amount > 0.0:
            sanity_checks["message"] = f"Could not parse Total Grant Value. Sum of Tranches: {total_tranche_amount}"
        else:
            sanity_checks[
                "message"] = f"Amounts MISMATCH. Total Grant: {total_grant_numeric}, Sum of Tranches: {total_tranche_amount}"

    # --- 7. Return Final Object ---
    return {
        "metadata": metadata,
        "sanity_checks": sanity_checks,
        "milestones": final_milestones
    }

