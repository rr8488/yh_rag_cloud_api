# yh_rag_cloud_api/parsers/excel_parser.py

import pandas as pd
import traceback
from typing import List, Dict, Optional

# Import the general utility from the parent directory's parsing_utils
try:
    from ..parsing_utils import find_header_row
except ImportError:
    # Fallback for standalone script testing
    def find_header_row(excel_path: str, keywords: List[str]) -> Optional[int]:
        print(f"[MOCK] find_header_row for {excel_path}")
        return 0 # Assume header is row 0 for test

def parse_kpi_schedule(excel_path: str) -> list[dict]:
    """ Parses Schedule 1 (KPIs) - Robust Header Finding """
    print(f"  Parsing KPI Excel: {excel_path}")
    try:
        # --- NEW: Find Header Row ---
        kpi_keywords = ['milestone', 'month', 'rm', 'amount', 'kpi', 'output']
        header_index = find_header_row(excel_path, kpi_keywords)
        if header_index is None:
            print("  WARNING: Could not reliably identify KPI header row. Skipping.")
            return []
        print(f"  Identified KPI header row at index: {header_index}")
        # --- END NEW ---

        # Read again, specifying the correct header row (0-based index)
        df = pd.read_excel(excel_path, sheet_name=0, header=header_index)

        # --- Column Mapping Logic (Keep previous improvements) ---
        column_map = {}
        found_kpi = False
        found_amount = False
        for col in df.columns:
            # Important: Skip unnamed columns pandas might create from merged cells
            if 'unnamed:' in str(col).lower(): continue

            col_str = str(col)
            col_lower = col_str.lower()

            if 'milestone' in col_lower: column_map['milestone_number'] = col_str
            elif 'month' in col_lower and 'due' in col_lower: column_map['disbursement_month'] = col_str
            elif col_str.strip() == 'RM' or ('amount' in col_lower or 'disbursement' in col_lower):
                 if 'disbursement_amount' not in column_map or col_str.strip() == 'RM':
                     column_map['disbursement_amount'] = col_str
                     found_amount = True
            elif 'kpi' in col_lower or 'output' in col_lower or 'indicator' in col_lower:
                column_map['kpi_description'] = col_str
                found_kpi = True

        if not found_kpi or not found_amount:
            print(f"  WARNING: KPI Excel - Header row {header_index} identified, but still missing required columns (KPI/Outputs, RM/Amount). Found: {list(column_map.keys())}. Skipping.")
            print(f"    Columns found by pandas at header={header_index}: {list(df.columns)}")
            return []
        # --- END Column Mapping ---

        # --- Data Cleaning (Keep previous improvements) ---
        df_renamed = df.rename(columns={v: k for k, v in column_map.items()})
        found_cols = list(column_map.keys())
        try:
             records = df_renamed[found_cols].to_dict('records')
        except KeyError as e:
             print(f"  ERROR: Column mapping failed after rename. Missing key: {e}")
             print(f"    Available columns after rename: {list(df_renamed.columns)}")
             return []

        cleaned_records = []
        for r in records:
            # Forward fill milestone number for merged cells
            if pd.isna(r.get('milestone_number')) and cleaned_records:
                 r['milestone_number'] = cleaned_records[-1].get('milestone_number')

            if not r.get('kpi_description') or pd.isna(r.get('kpi_description')):
                continue
            cleaned_records.append({
                k: (v if pd.notna(v) else None) for k, v in r.items()
            })
        # --- END Data Cleaning ---

        print(f"  Successfully parsed {len(cleaned_records)} KPI records.")
        return cleaned_records

    except Exception as e:
        print(f"  ERROR: KPI Excel parser failed: {e}")
        traceback.print_exc()
        return []


def parse_budget_schedule(excel_path: str) -> list[dict]:
    """ Parses Schedule 3 (Budget) - Manually finds header row """
    print(f"  Parsing Budget Excel (Manual Header Find): {excel_path}")
    try:
        # --- Step 1: Read sheet without assuming header ---
        df_no_header = pd.read_excel(excel_path, sheet_name=0, header=None)

        # --- Step 2: Manually find the header row ---
        header_row_index = -1
        actual_headers = []
        required_keywords = ['category', 'line item', 'total', 'rm'] # Keywords to identify the row

        for i, row in df_no_header.head(15).iterrows(): # Check first 15 rows
            row_values = [str(cell).lower().strip() for cell in row if pd.notna(cell)]
            matches = sum(1 for keyword in required_keywords
                          if any(keyword in cell_val for cell_val in row_values))

            if matches >= 3:
                header_row_index = i
                actual_headers = [str(cell).strip() for cell in row] # Get headers from this row
                print(f"  Found potential budget header at row index: {header_row_index}")
                print(f"    Headers found: {actual_headers}")
                break # Stop searching once found

        if header_row_index == -1:
            print("  WARNING: Could not find budget header row containing 'CATEGORY', 'LINE ITEM', 'TOTAL (RM)'. Skipping.")
            return []

        # --- Step 3: Read data specifying header row AND skip rows above data ---
        df = pd.read_excel(excel_path, sheet_name=0, header=header_row_index, skiprows=range(header_row_index + 1))
        df.columns = actual_headers[:len(df.columns)] # Use only as many headers as pandas found columns

        print(f"    Columns read by pandas using header={header_row_index}: {list(df.columns)}")

        # --- Step 4: Map columns (similar to previous version) ---
        column_map = {}
        found_item = False
        found_amount = False
        for col in df.columns:
            if col is None or pd.isna(col): continue # Skip empty column headers
            col_str = str(col)
            col_lower = col_str.lower().strip()

            if 'category' in col_lower: column_map['category'] = col_str
            elif 'line item' in col_lower or 'description' in col_lower:
                column_map['line_item'] = col_str
                found_item = True
            elif 'total' in col_lower and 'rm' in col_lower:
                column_map['budgeted_amount'] = col_str
                found_amount = True
            elif not found_amount and ('budget' in col_lower or 'amount' in col_lower or 'rm' in col_lower):
                 column_map['budgeted_amount'] = col_str
                 found_amount = True

        if not found_item or not found_amount:
            print(f"  WARNING: Budget Excel - Found header row {header_row_index}, but still missing required columns ('LINE ITEM', 'TOTAL (RM)'/Amount) after mapping. Found Keys: {list(column_map.keys())}. Skipping.")
            return []
        # --- Step 5: Process Records (same as previous version) ---
        df_renamed = df.rename(columns={v: k for k, v in column_map.items()})
        db_cols = ['category', 'line_item', 'budgeted_amount']
        found_db_cols = [col for col in db_cols if col in column_map]

        try:
             records_df = df_renamed[found_db_cols]
             records = records_df.to_dict('records')
        except KeyError as e:
             print(f"  ERROR: Column mapping failed. Missing key: {e}")
             print(f"    Available columns after rename: {list(df_renamed.columns)}")
             return []

        cleaned_records = []
        last_category = None
        for r in records:
             current_category = r.get('category')
             if pd.notna(current_category): last_category = current_category
             elif pd.isna(current_category) and last_category: r['category'] = last_category
             if not r.get('line_item') or pd.isna(r.get('line_item')): continue

             amount = r.get('budgeted_amount')
             if pd.notna(amount):
                 try: r['budgeted_amount'] = float(str(amount).replace(',',''))
                 except (ValueError, TypeError): r['budgeted_amount'] = None
             else: r['budgeted_amount'] = None

             cleaned_records.append({
                 'project_id': None,
                 'category': str(r.get('category')).strip() if pd.notna(r.get('category')) else None,
                 'line_item': str(r.get('line_item')).strip() if pd.notna(r.get('line_item')) else None,
                 'budgeted_amount': r.get('budgeted_amount')
             })

        print(f"  Successfully parsed {len(cleaned_records)} Budget records.")
        return cleaned_records

    except Exception as e:
        print(f"  ERROR: Budget Excel parser failed: {e}")
        traceback.print_exc()
        return []
