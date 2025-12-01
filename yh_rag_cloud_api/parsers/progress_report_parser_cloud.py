# yh_rag_cloud_api/parsers/progress_report_parser_cloud.py

import json
import re
import unicodedata
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, ValidationError

# ---
# 1. IMPORT THE CLOUD LLM
# ---
try:
    from ..rag_utils import ask_gemini
except ImportError:
    print("! WARN: 'from ..rag_utils...' failed. Using mock ask_gemini.")


    # Fallback for standalone script testing
    def ask_gemini(prompt: str, model: str = "gemini-pro") -> str:
        print("[MOCK GEMINI CALL - Progress Report Parser]")
        return "{}"


# ---
# 2. DEFINE THE CANONICAL SCHEMA
# (Unchanged)
# ---

class Beneficiary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: str = Field(description="The category of beneficiary (e.g., 'Students', 'Teachers', 'Parents').")
    count: Optional[int] = Field(
        alias="number",
        default=None,
        description="The number of beneficiaries in this category for this reporting period."
    )
    metrics_tracked: Optional[str] = Field(
        alias="tracking_data",
        default=None,
        description="A brief description of what is being tracked for this group (e.g., 'Learning skills', 'Pre and post evaluations')."
    )


class Deliverable(BaseModel):
    description: str = Field(description="The description of the deliverable or milestone as stated in the report.")
    status: str = Field(
        description="The inferred status, normalized to one of: ['Met', 'Partially Met', 'Not Met', 'Ongoing', 'Delayed', 'Not Applicable']")
    progress_update: str = Field(description="The full text of the progress update provided for this deliverable.")


class CanonicalProgressReport(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    report_number: Optional[int] = Field(
        default=None,
        description="The sequential number of the report (e.g., 3). **Look for 'This report is #'.**"
    )
    report_date: Optional[str] = Field(
        description="The single date the report was submitted, normalized to 'YYYY-MM-DD'. Find 'Report Date (Do Not Delete)' or similar fields.")
    reporting_period_start: Optional[str] = Field(default=None,
                                                  description="The start date of the monitoring period this report covers, normalized to 'YYYY-MM-DD'.")
    reporting_period_end: Optional[str] = Field(default=None,
                                                description="The end date of the monitoring period this report covers, normalized to 'YYYY-MM-DD'.")
    project_title: Optional[str] = Field(default=None, description="The official title of the project.")
    organisation_name: Optional[str] = Field(default=None,
                                             description="The name of the organisation submitting the report.")
    grant_amount_myr: Optional[float] = Field(
        default=None,
        description="The total grant amount (RM). **Look for the line 'Grant amount (RM) [Jumlah Geran (RM)]'.**"
    )
    funds_unutilized_to_date_myr: Optional[float] = Field(
        default=None,
        description="The total 'Fund unutilised to date (RM)'. **Look for the line 'Unutilised fund (from the amount disbursed (RM))'.**"
    )
    number_of_disbursement_to_date: Optional[int] = Field(
        default=None,
        description="The 'Number of disbursement to-date'. **Look for the question 'What number disbursement is this?' and use its answer.**"
    )
    month_year_of_disbursement: Optional[str] = Field(
        default=None,
        description="The 'Month and year of disbursement'."
    )
    project_locations: List[str] = Field(
        default_factory=list,
        description="List of project locations (Malaysian states)."
    )
    number_of_deliverables: Optional[int] = Field(
        default=None,
        description="Number of deliverables for this report. **Look for 'number of deliverable(s)'.**"
    )
    report_type: Optional[str] = Field(
        default="mid-progress",
        description="Detected report type: 'mid-progress' or 'final'."
    )
    executive_summary: Optional[str] = Field(
        default=None,
        description="Full text from the '1. Executive Summary' section."
    )
    effectiveness_summary: Optional[str] = Field(
        default=None,
        description="Full text from the '2. Effectiveness' section."
    )
    methodology_that_work_and_not_work: Optional[str] = Field(
        default=None,
        description="Answer to project methodology sub-question a: What methods and approaches worked..."
    )
    methodology_right_intervention_and_target_or_done_differently: Optional[str] = Field(
        default=None,
        description="Answer to project methodology sub-question b: Was the project the right intervention..."
    )
    methodology_successes_and_challenges: Optional[str] = Field(
        default=None,
        description="Answer to project methodology sub-question c: What successes and challenges did you encounter..."
    )
    sustainability_scalability_summary: Optional[str] = Field(
        default=None,
        description="Full text from the '5. Sustainability and Scalability' section."
    )
    funds_disbursed_to_date_myr: Optional[float] = Field(
        default=None,
        description="The total funds 'Disbursement to-date (RM)'. **Look for the line 'Fund disbursed to-date (RM) [Jumlah dana...'.**"
    )
    funds_utilized_to_date_myr: Optional[float] = Field(
        default=None,
        description="The total 'Fund utilised to date (RM)'. **Look for the line 'Fund utilised to-date (RM) [Jumlah dana...'.**"
    )
    deliverables: List[Deliverable] = Field(default_factory=list,
                                            description="A list of all deliverables and their progress updates mentioned in the 'PROGRESS UPDATE' section.")
    lessons_learned_summary: Optional[str] = Field(
        alias="lessons_learnt",
        default=None,
        description="The full, raw text from the 'LESSONS LEARNT' section (or 4a for Final Reports)."
    )
    challenges_summary: Optional[str] = Field(
        alias="challenges",
        default=None,
        description="The full, raw text from the 'CHALLENGES' section (or 4b for Final Reports)."
    )
    success_stories_summary: Optional[str] = Field(
        alias="success_stories",
        default=None,
        description="The full, raw text from the 'SUCCESS STORIES' section."
    )
    beneficiaries: List[Beneficiary] = Field(default_factory=list,
                                             description="A list of all beneficiary types from the 'BENEFICIARIES' section.")


# ---
# 3. DEFINE HELPER FUNCTIONS
# ---

def clean_control_chars(s: str) -> str:
    """Removes common control characters except newline and tab from a string."""
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in ('\n', '\t'))


def _parse_flexible_date(date_str: Any) -> Optional[str]:
    """
    Tries to parse various date formats and returns a YYYY-MM-DD string or None.
    """
    if not isinstance(date_str, str):
        return None

    cleaned_date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str.strip())

    # Handle "Nov 2022 - Feb 2023" case
    if ' - ' in cleaned_date_str:
        # We need to parse start and end separately
        parts = re.split(r'\s+-\s+', cleaned_date_str)
        if len(parts) == 2:
            start_str = parts[0].strip()
            end_str = parts[1].strip()
            # Try parsing start date
            start_dt = _parse_single_date(start_str)
            end_dt = _parse_single_date(end_str)
            # This function is used for multiple fields, so we can't return two.
            # For the *field* "reporting_period_start", we'll return the start date.
            # The _pre_parse_metadata function splits them, so this works.
            return start_dt
        else:
            # Fallback to just parsing the first part
            cleaned_date_str = parts[0].strip()

    return _parse_single_date(cleaned_date_str)


def _parse_single_date(date_str: str) -> Optional[str]:
    """Helper to parse a single date string."""
    dt = None
    fmts = [
        '%d/%m/%Y',  # 15/04/2020
        '%d %B %Y',  # 01 February 2020
        '%d %b %Y',  # 01 Feb 2020
        '%Y-%m-%d',  # 2020-02-01
        '%b %Y',  # Feb 2020 (Will default to day 1)
        '%B %Y',  # February 2020 (Will default to day 1)
    ]

    for fmt in fmts:
        try:
            dt = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            pass

    if dt:
        return dt.strftime('%Y-%m-%d')
    else:
        print(f"  WARN: Could not parse flexible date string: '{date_str}'")
        return None


def _get_float(value_str: Optional[str]) -> Optional[float]:
    """Converts currency string to float."""
    if value_str is None:
        return None
    try:
        # Remove currency, commas, and whitespace
        return float(re.sub(r"[^0-9.-]", "", value_str))
    except (ValueError, TypeError):
        return None


def _get_int(value_str: Optional[str]) -> Optional[int]:
    """Converts a string to an int, stripping non-numeric chars."""
    if value_str is None:
        return None
    try:
        # Get only digits
        cleaned_str = re.sub(r"[^0-9]", "", value_str)
        if not cleaned_str:
            return None
        return int(cleaned_str)
    except (ValueError, TypeError):
        return None


# ---
# 4. REBUILT HYBRID HELPER: REGEX PRE-PARSER (Corrected)
# ---
def _pre_parse_metadata(report_text: str) -> Dict[str, Any]:
    """
    Uses robust regex to find key-value pairs that the LLM struggles with.
    This version searches the full text for a key, then finds the *next*
    colon (:) and extracts the value from there.
    This handles keys and values on different lines.
    It uses the robust, general patterns from the successful on-prem parser.

    Returns a dictionary of *raw string values* to be processed later.
    """
    data = {}

    # Define a map of schema_key -> regex_pattern
    # We use re.DOTALL so '.' matches newlines, helping find keys split across lines
    # These patterns are sourced from the successful on-prem parser
    patterns = {
        "report_number": r"This report is #",
        "reporting_period_start_end": r"monitoring period",  # Special key for "Nov 2022 - Feb 2023"
        "grant_amount_myr": r"Grant amount",  # General pattern
        "funds_disbursed_to_date_myr": r"disburse.*to.*date",  # General pattern
        "funds_utilized_to_date_myr": r"utilised.*to.*date",  # General pattern
        "funds_unutilized_to_date_myr": r"unutilised fund",  # General pattern
        "number_of_disbursement_to_date": r"number.*disbursement",  # General pattern
        "number_of_deliverables": r"number of deliverable\(s\)"
    }

    for key, pattern in patterns.items():
        try:
            # Find the key pattern
            # We remove bilingual hints from the text to make matching cleaner
            clean_text = re.sub(r"\[.*?\]", "", report_text)
            match = re.search(pattern, clean_text, re.IGNORECASE | re.DOTALL)
            if not match:
                continue

            # Find the *next* colon after the key
            colon_index = clean_text.find(':', match.end())
            if colon_index == -1:
                continue

            # Find the end of that line
            eol_index = clean_text.find('\n', colon_index)
            if eol_index == -1:
                # If no newline, just take to the end of the string
                value_str = clean_text[colon_index + 1:]
            else:
                value_str = clean_text[colon_index + 1: eol_index]

            # Clean and store the raw value
            value = value_str.strip()
            if value:
                data[key] = value

        except Exception as e:
            print(f"  WARN: Regex pre-parser failed for key '{key}': {e}")

    # Post-process the found data
    if "reporting_period_start_end" in data:
        value = data.pop("reporting_period_start_end")
        # Value is "Nov 2022 - Feb 2023"
        parts = re.split(r'\s+-\s+', value)  # Split on " - "
        if len(parts) == 2:
            data["reporting_period_start"] = parts[0].strip()
            data["reporting_period_end"] = parts[1].strip()

    return data


# ---
# 5. DEFINE THE MAIN CLOUD PARSER FUNCTION (HYBRID)
# ---

def parse_progress_report_cloud(report_text: str) -> Dict[str, Any]:
    """
    Parses plain text from a progress report (extracted by Document AI)
    into a structured dictionary using a HYBRID approach:

    1.  A regex pre-parser finds simple key-value fields deterministically.
    2.  An LLM (Gemini) extracts all fields, excelling at large unstructured blocks.
    3.  The results are merged, with the deterministic regex values OVERWRITING
        the LLM's values for those specific fields.
    """
    print(f"[parse_progress_report_cloud] Starting HYBRID parsing... (text length: {len(report_text)})")

    # ---
    # 1. HYBRID STEP: Run deterministic regex parser first
    # ---
    print("  Running regex pre-parser...")
    pre_parsed_data = _pre_parse_metadata(report_text)
    print(f"  ...pre-parser found: {pre_parsed_data}")

    # Create the schema for the LLM
    try:
        schema = CanonicalProgressReport.model_json_schema()
    except Exception as e:
        print(f"  ERROR: Could not generate schema from Pydantic model: {e}")
        return {"error": "Internal server error: Could not generate schema."}

    # ---
    # 2. CREATE THE PROMPT
    # ---
    prompt = f"""
    You are an experienced Grant manager for Yayasan Hasanah in Malaysia.
    You are reviewing a progress report by a grant recipient required to receive the next disbursement
    Analyze the `Report Text` below.
    Your task is to find all the data fields listed in the `JSON Schema` and return them
    as a SINGLE, valid JSON object.
    It is very important to capture values related to the grant money (total, received, disbursed. etc).

    CRITICAL RULES:
    1.  Respond with ONLY the JSON object. Do not add any commentary, markdown, or other text.
    2.  If a value is not found, return `null` for that key.
    3.  For dates, extract them as found (e.g., "Feb 2024").
    4.  For money/numbers, extract only the number (e.g., "10,000.00").
    5.  `report_type`: Analyze the text. If it mentions "Final Report" or "Executive Summary",
        set to "final". Otherwise, default to "mid-progress".
    6.  `beneficiaries` and `deliverables`: These must be JSON arrays of objects,
        matching the schema. Extract all items you can find.
    7.  All string values MUST be properly escaped (e.g., newlines as \\n, quotes as \\").

    JSON SCHEMA:
    ```json
    {json.dumps(schema, indent=2)}
    ```

    Report Text:
    ---
    {report_text}
    ---

    JSON response:
    """

    try:
        # 3. Call Gemini
        print("  Calling Gemini for data extraction...")
        response_text = ask_gemini(prompt)
        print("  ...Gemini call complete.")

        # 4. Clean and Parse JSON from LLM
        json_str_raw = response_text.strip().lstrip("```json").rstrip("```").strip()
        json_str_cleaned = clean_control_chars(json_str_raw)

        result_raw = None
        try:
            result_raw = json.loads(json_str_cleaned)
        except json.JSONDecodeError as e:
            print(f"  WARN: Initial JSON parse failed on cleaned string ({e}). Retrying with fixes...")
            # Fallback logic
            fixed_json_str = ""
            in_string = False
            escape = False
            for char in json_str_cleaned:
                if char == '"' and not escape:
                    in_string = not in_string
                    fixed_json_str += char
                elif char == '\\':
                    escape = True
                    fixed_json_str += char
                elif char == '\n' and in_string:
                    fixed_json_str += r'\\n'
                elif char in ('\n', '\t') and not in_string:
                    pass
                else:
                    escape = False
                    fixed_json_str += char

            try:
                result_raw = json.loads(fixed_json_str)
                print("  Successfully parsed with newline/tab fix.")
            except json.JSONDecodeError as e2:
                print(f"  ERROR: JSON parse failed even after fixes: {e2}")
                return {"error": f"LLM returned invalid JSON: {e2}", "raw_response": response_text}

        if not isinstance(result_raw, dict):
            print(f"  ERROR: LLM did not return a JSON object. Got type: {type(result_raw)}")
            return {"error": "LLM did not return a JSON object", "raw_response": response_text}

        # ---
        # 5. HYBRID STEP: Merge results
        # ---
        # The LLM's result is the 'base'.
        # The pre-parsed data's values will OVERWRITE the LLM's values.
        # This is because the regex is more reliable for these specific fields.
        print("  Merging LLM results with pre-parsed data (regex wins)...")
        merged_data = {**result_raw, **pre_parsed_data}

        # We will use merged_data for the rest of the process
        result_raw = merged_data

        # 6. Post-process and Validate
        print("  Post-processing and validating data...")

        # --- Post-processing ---
        # This section will now run on the merged data.
        # The _get_float/int functions will correctly parse the
        # raw strings provided by the _pre_parse_metadata function.
        result_raw['report_date'] = _parse_flexible_date(result_raw.get('report_date'))
        result_raw['reporting_period_start'] = _parse_flexible_date(result_raw.get('reporting_period_start'))
        result_raw['reporting_period_end'] = _parse_flexible_date(result_raw.get('reporting_period_end'))

        result_raw['grant_amount_myr'] = _get_float(result_raw.get('grant_amount_myr'))
        result_raw['funds_unutilized_to_date_myr'] = _get_float(result_raw.get('funds_unutilized_to_date_myr'))
        result_raw['funds_disbursed_to_date_myr'] = _get_float(result_raw.get('funds_disbursed_to_date_myr'))
        result_raw['funds_utilized_to_date_myr'] = _get_float(result_raw.get('funds_utilized_to_date_myr'))

        result_raw['report_number'] = _get_int(result_raw.get('report_number'))
        result_raw['number_of_disbursement_to_date'] = _get_int(result_raw.get('number_of_disbursement_to_date'))
        result_raw['number_of_deliverables'] = _get_int(result_raw.get('number_of_deliverables'))

        # --- End Post-processing ---

        # Validate with the Pydantic model
        report_model = CanonicalProgressReport(**result_raw)
        validated_data = report_model.model_dump()  # Use .model_dump() for Pydantic v2

        print(
            f"[parse_progress_report_cloud] Cloud HYBRID parsing and validation successful. "
            f"Type: {validated_data.get('report_type')}. "
            f"Org: {validated_data.get('organisation_name')}"
        )
        return validated_data

    except ValidationError as pydantic_err:
        print(f"ERROR: Failed to validate extracted data: {pydantic_err}")
        return {"error": f"Schema validation failed: {pydantic_err}", "raw_extracted_data": result_raw}
    except Exception as e:
        print(f"  ERROR: Cloud parser failed during processing: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"An unexpected error occurred: {e}"}