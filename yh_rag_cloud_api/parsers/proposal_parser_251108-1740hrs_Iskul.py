# yh_rag_cloud_api/parsers/proposal_parser.py

import re
import json
import unicodedata
from typing import Optional, Tuple
from bs4 import BeautifulSoup  # <-- Used for metadata

try:
    from ..rag_utils import _ask_ollama, query_llm
except ImportError:
    # Fallback for standalone script testing
    def _ask_ollama(prompt: str, model: str) -> str:
        print("[MOCK OLLAMA CALL - Classifier]")
        # (This mock is now UPDATED based on your successful JSON output)
        if "project goal" in prompt.lower():
            return "project_goal_long_term"
        if "brief summary" in prompt.lower():
            return "project_summary"
        if "please describe project objective #1" in prompt.lower():
            return "project_objectives"
        if "list and provide an estimate number" in prompt.lower():
            return "beneficiaries"
        if "organisation name" in prompt.lower():
            return "organization_name"
        if "company id" in prompt.lower():
            return "company_id"
        if "project title" in prompt.lower():
            return "project_title"
        if "requested funding" in prompt.lower():
            return "funding_requested_total"
        if "organisation have the sector" in prompt.lower():
            return "organization_expertise"
        if "monitoring and evaluation" in prompt.lower() or "please present (a)" in prompt.lower():
            return "monitoring_and_evaluation_plan"
        if "implementing this project with another" in prompt.lower():
            return "implementing_with_other_org"
        if "total duration" in prompt.lower():
            return "project_duration_months"
        if "location(s)" in prompt.lower():
            return "project_location_state"
        if "specify village" in prompt.lower():
            return "project_location_details"
        if "number of team members" in prompt.lower():
            return "project_team_size"
        if "number of volunteers" in prompt.lower():
            return "project_volunteer_size"
        if "beneficiaries will be involved" in prompt.lower():
            return "beneficiary_involvement"
        if "beneficiaries will directly benefit" in prompt.lower():
            return "beneficiary_direct_benefits"
        if "brief situation" in prompt.lower():
            return "problem_statement"
        if "sustainable development goals" in prompt.lower():
            return "sdg_goals"
        if "if this project is part of a larger" in prompt.lower():
            return "project_phase_in_larger_program"
        if "how will the project address this issue" in prompt.lower():
            return "project_address_issue"
        if "how have you identified this issue" in prompt.lower():
            return "problem_identification_source"
        if "why did you choose this approach" in prompt.lower():
            return "chosen_approach_reason"
        if "at the policy level" in prompt.lower():
            return "sustainability_policy"
        if "financial –" in prompt.lower():
            return "sustainability_financial"
        if "institutional –" in prompt.lower():
            return "sustainability_institutional"
        if "programme management fund" in prompt.lower():
            return "funding_pmf"
        if "organisational development fund" in prompt.lower():
            return "funding_odf"
        if "why your organisation requires odf" in prompt.lower():
            return "odf_justification"
        if "are you a past / current partner" in prompt.lower():
            return "is_past_partner"
        if "how your proposed project is aligned" in prompt.lower():
            return "impact_area_alignment"
        if "please select the box corresponding" in prompt.lower():
            return "impact_area"

        # This handles the stray ":" tag
        if re.fullmatch(r"^\s*:\s*$", prompt.strip()):
            return "unclassified_answer_marker"

        return "unclassified"


# ---
# V39.5 PARSER IMPLEMENTATION
# ---

def clean_control_chars(s: str) -> str:
    """Removes common control characters except newline and tab from a string."""
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in ('\n', '\t'))


def _clean_content(text: str, remove_bilingual=True, remove_parentheticals=True) -> str:
    """
    Cleans HTML artifacts *after* parsing.
    V39.5: Decoupled remove_bilingual from remove_parentheticals.
    """
    if not text:
        return ""

    # 1. Strip all HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2. Decode HTML entities
    try:
        import html
        text = html.unescape(text)
    except ImportError:
        pass  # Fallback if html lib not available

    # 3. Remove all bilingual headers (e.g., [Tajuk Projek])
    if remove_bilingual:
        text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)

    # 4. Remove all parenthetical asides (e.g., (e.g: tables, charts etc))
    # --- FIX V39.5 ---
    # This rule is now separate from bilingual removal.
    if remove_parentheticals:
        text = re.sub(r'\([^)]*\)', '', text, flags=re.DOTALL)
    # --- END FIX ---

    # 5. Remove artifact lines/separators like '---' or '***'
    text = re.sub(r'^\s*[\-_\*]{3,}\s*$', '', text, flags=re.MULTILINE)

    # 6. Remove markdown (which shouldn't be here, but as a safeguard)
    text = text.replace('**', '').replace('##', '').replace('>', '')

    # 7. Collapse newlines and whitespace
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        # --- FIX V39.5 ---
        # This logic must also respect the new flag
        if remove_parentheticals:
            if not stripped_line or re.fullmatch(r'^\s*[:\.,]\s*$', stripped_line):
                continue
        elif not stripped_line:
            continue
        cleaned_lines.append(stripped_line)
        # --- END FIX ---

    text = '\n'.join(cleaned_lines)

    # 8. Final cleanup
    text = re.sub(r"(\n\s*){2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    # --- FIX V39.5 ---
    # This logic must also respect the new flag
    if remove_parentheticals:
        text = text.rstrip(':')
    # --- END FIX ---

    return text


# --- END _clean_content ---


# This is the "canonical schema" of keys the LLM is allowed to classify headers into.
CLASSIFICATION_SCHEMA = [
    "organization_name", "company_id", "is_past_partner", "project_title",
    "implementing_with_other_org", "impact_area", "impact_area_priority",
    "impact_area_alignment", "sdg_goals", "project_duration_months",
    "project_location_state", "project_location_details", "project_team_size",
    "project_volunteer_size", "project_goal_long_term", "project_summary",
    "project_objectives", "project_phase_in_larger_program", "beneficiaries",
    "beneficiary_involvement", "beneficiary_direct_benefits", "problem_statement",
    "project_address_issue", "problem_identification_source",
    "chosen_approach_reason", "organization_expertise",
    "monitoring_and_evaluation_plan", "sustainability_financial",
    "sustainability_institutional", "sustainability_policy",
    "funding_requested_total", "funding_pmf", "funding_odf",
    "odf_justification", "covid_safety_measures", "baseline_info",
    "org_structure_upload",
    "unclassified",
    "unclassified_answer_marker"  # Special key for stray ":"
]

# --- V39.5: Header classification map (Final) ---
HEADER_CLASSIFICATION_MAP = {
    # --- Real Questions ---
    "are you a past / current partner": "is_past_partner",
    "project title": "project_title",
    "implementing this project": "implementing_with_other_org",
    "please select the box corresponding": "impact_area",
    "specific impact area for which": "impact_area",
    "specific programme for which": "impact_area",
    "how your proposed project is aligned": "impact_area_alignment",
    "sustainable development goals": "sdg_goals",
    "total duration": "project_duration_months",
    "location(s)": "project_location_state",
    "specify village": "project_location_details",
    "number of team members": "project_team_size",
    "number of staffs": "project_team_size",
    "number of volunteers": "project_volunteer_size",
    "project goal": "project_goal_long_term",
    "brief summary": "project_summary",
    "brief description": "project_summary",
    "if this project is part of a larger ongoing programme": "project_phase_in_larger_program",
    "if this project is part of a larger on-going programme": "project_phase_in_larger_program",
    "please describe project objective #1": "project_objectives",
    "please list and provide an estimate number": "beneficiaries",
    "15a. please list and provide": "beneficiaries",
    "please describe if and how": "beneficiary_involvement",
    "15b. please describe if and how": "beneficiary_involvement",
    "please demonstrate how": "beneficiary_direct_benefits",
    "15c. please demonstrate how": "beneficiary_direct_benefits",
    "please provide a brief situation": "problem_statement",
    "how will the project address this issue": "project_address_issue",
    "how have you identified this issue": "problem_identification_source",
    "why did you choose this approach": "chosen_approach_reason",
    "please share to what extent": "organization_expertise",
    "please present (a) the type of information": "monitoring_and_evaluation_plan",
    "financial –": "sustainability_financial",
    "22a. financial –": "sustainability_financial",
    "institutional –": "sustainability_institutional",
    "22b. institutional –": "sustainability_institutional",
    "at the policy level": "sustainability_policy",
    "22c. at the policy level": "sustainability_policy",
    "requested funding": "funding_requested_total",
    "programme management fund": "funding_pmf",
    "organisational development fund": "funding_odf",
    "please explain why your organisation requires odf": "odf_justification",
    "upload your organisational structure": "org_structure_upload",
    "organisational structure": "org_structure_upload",
    "do you have any baseline information": "baseline_info",
    "given the potential prolonged of covid-19": "covid_safety_measures",

    # --- Junk / Section Headers (Map to unclassified) ---
    "project objectives, activities": "unclassified",
    "number of objectives for this project": "unclassified",
    "number of activities for project objective": "unclassified",
    "please describe activity #1": "unclassified",
    "please describe activity #2": "unclassified",
    "please describe the estimated output": "unclassified",
    "description of the project": "unclassified",
    "background of the project": "unclassified",
    "monitoring and evaluation (m&e)": "unclassified",
    "sustainability of the project": "unclassified",
    "implementation plan - please upload": "unclassified",
    "completed implementation plan": "unclassified",
    "budget template - please upload": "unclassified",
    "completed budget template": "unclassified",
    "optional: additional documents": "unclassified",
    "audited financial accounts": "unclassified",
    "details of the proposed project": "unclassified",
    "source of materials / research": "unclassified",
    "additional financial documentation": "unclassified",
    "last year’s / current year’s balance sheet": "unclassified",
    "the current operating budget breakdown": "unclassified",
    "optional documentation": "unclassified",
    "optional document 1": "unclassified",
    "record label": "unclassified",
    "linked to partner profile": "unclassified",

    # --- V39.5 FIX: Add rules to stop LLM fallback on these headers ---
    "the project": "unclassified",
    "the beneficiaries": "unclassified",
    "the location": "unclassified"
    # --- END V39.5 FIX ---
}


def _classify_header(header_text: str) -> str:
    """
    (V39.5)
    Uses a fast string-matching map first.
    Expects CLEANED TEXT (with parentheticals) as input.
    """
    # Sanitize the header for matching
    clean_header_text = re.sub(r'\s+', ' ', header_text).strip().lower()[:500]

    # --- Fast Path (Rule-Based) ---

    # Handle the special ":" tag case
    if re.fullmatch(r"^\s*:\s*$", header_text.strip()):
        return "unclassified_answer_marker"

    for keyword, key in HEADER_CLASSIFICATION_MAP.items():
        if keyword in clean_header_text:
            return key

    # --- Fallback Path (LLM) ---
    print(f"    [Parser] No fast-path match. Falling back to LLM for: {header_text[:70].strip()}...")
    prompt = f"""
        You are a document classification bot.
        Your job is to categorize the <header_text> into ONE of the categories
        from the <schema>.
        Respond with ONLY the matching category name.

        <schema>
        {json.dumps(CLASSIFICATION_SCHEMA)}
        </schema

        <header_text>
        {clean_header_text}
        </header_text>
    """
    try:
        raw_response = query_llm(prompt)
        clean_response = raw_response.strip().replace("`", "").replace('"', "")
        if clean_response in CLASSIFICATION_SCHEMA:
            return clean_response
        else:
            for key in CLASSIFICATION_SCHEMA:
                if key in clean_response:
                    return key
            return "unclassified"
    except Exception as e:
        print(f"  [Parser] ERROR: LLM Classifier failed: {e}")
        return "unclassified"


# --- METADATA PARSER (V27.0 - STABLE) ---
def _parse_metadata_table(table_html: str) -> dict:
    """
    (V27.0 - Pass 1)
    Parses the initial HTML table for the 4 metadata keys.
    This version handles keys/values being in the *same cell*.
    """
    metadata = {}
    print("  [Parser] V27.0 (Pass 1) Parsing Metadata <table> (HTML)...")

    key_map = {
        # "Key literal as seen in HTML" : "canonical_json_key"
        "Organisation name": "organization_name",
        "Company ID": "company_id",
        "Grant Application ID": "grant_application_id",
        "Project ID": "project_id"
    }

    # Initialize all keys to empty strings
    for key_canonical in key_map.values():
        metadata[key_canonical] = ""

    try:
        # Find ALL cells, both th and td, and get their content
        cells = re.findall(r"<(?:th|td).*?>(.*?)<\/(?:th|td)>", table_html, re.DOTALL | re.IGNORECASE)

        for cell_html in cells:
            # Clean all HTML tags (like <strong>) from the cell
            cell_text = re.sub(r'<[^>]+>', '', cell_html).strip()

            if not cell_text:
                continue  # Skip empty cells

            # Check if this cell's text starts with one of our known keys
            for key_literal, key_canonical in key_map.items():
                if cell_text.startswith(key_literal):
                    # Found a match. Split the string at the first colon
                    parts = cell_text.split(":", 1)

                    if len(parts) > 1:
                        # We have a value part
                        value = parts[1].strip()
                        # Use a simple cleaner just for bilingual, not _clean_content
                        cleaned_value = re.sub(r'\[.*?\]', '', value, flags=re.DOTALL).strip()
                        if cleaned_value:
                            metadata[key_canonical] = cleaned_value

                    # Break from the inner loop (key_map) and move to the next cell
                    break

    except Exception as e:
        print(f"  [Parser] V27.0 (Pass 1) ERROR parsing HTML table: {e}")

    print(f"  [Parser] V27.0 (Pass 1) Metadata found: {metadata}")
    return metadata


# --- END OF METADATA FUNCTION ---


# --- FUNCTION TO HANDLE COMBINED Q:A (V39.6 - STABLE) ---
def _split_at_colon(cleaned_text: str) -> Tuple[str, str, bool]:
    """
    (V39.6)
    Implements the user's Rule #2.
    Checks for the "Combined" Q:A format (e.g., "Q: A")
    Expects pre-cleaned text (no HTML, but WITH bilingual text).
    Returns (Question, Answer, found_answer_in_tag)
    V39.6: Added logic to ignore colons following "e.g", "eg", or "for example".
    """

    text = re.sub(r'\s+', ' ', cleaned_text).strip()

    if ":" not in text:
        return cleaned_text, "", False  # No colon found

    parts = text.rsplit(":", 1)  # Split on the *last* colon

    # This should be theoretically impossible if ":" is in text, but good safety check
    if len(parts) < 2:
        return cleaned_text, "", False

    question_part = parts[0].strip()
    answer_part = parts[1].strip()

    # --- V39.6 FIX ---
    # Check if the text *ending* the question_part is an example keyword.
    # We check a small slice from the end for efficiency.
    question_lower_chunk = question_part.lower()[-20:]

    if (
            question_lower_chunk.endswith("e.g") or
            question_lower_chunk.endswith("eg") or
            question_lower_chunk.endswith("for example") or
            question_lower_chunk.endswith("sebagai contoh")  # Added Malay
    ):
        # This colon is part of an example. The *entire* string is the question.
        # Treat it as if no answer was found in the tag.
        print(f"    [Parser] V39.6: Ignoring example colon in: {cleaned_text[:50]}...")
        return cleaned_text, "", False
    # --- END V39.6 FIX ---

    # --- Original logic continues ---
    # This is the key: did we find a *real* answer, or just a colon?
    if answer_part:
        # It's a real split (like in the PMF/ODF headers)
        return question_part, answer_part, True  # Found answer *inside* tag
    else:
        # It's a colon-terminated header (like project_goal)
        return question_part, "", False  # Found a colon, but no answer


# --- END NEW FUNCTION ---

# --- V39.0: Regex for Q&A (HTML) ---
# This is the finditer logic from V-1300hrs
HTML_SPLITTER_REGEX = re.compile(
    r"""
    (                   # Group 1: The entire Header block
        <h2.*?>.*?</h2>  # An h2 block (non-greedy)
        |               # OR
        <strong>.*?</strong> # A strong block (non-greedy)
    )
    """,
    re.VERBOSE | re.DOTALL | re.IGNORECASE
)


def parse_proposal(proposal_text: str) -> dict:
    """
    (V39.5 - Final Corrected Logic)
    """
    print("  [Parser] Running V39.5 (Final Logic)...")
    parsed_data = {}

    # --- Step 1: (Pass 1 - Metadata) ---
    metadata = {}
    qa_content_html = ""

    # This regex finds the *first* table
    metadata_table_match = re.search(r"(<table.*?>.*?</table>)", proposal_text, re.DOTALL | re.IGNORECASE)

    if metadata_table_match:
        metadata_html = metadata_table_match.group(1)
        metadata = _parse_metadata_table(metadata_html)  # <-- This calls V27.0
        parsed_data.update(metadata)
        qa_content_html = proposal_text[metadata_table_match.end():]
    else:
        print("  [Parser] V39.5 (Pass 1) WARNING: No <table> found. Parsing all content as Q&A.")
        qa_content_html = proposal_text

    metadata_keys = set(metadata.keys())

    # --- Step 2: (Pass 2 - finditer Q&A Extraction) ---
    print("  [Parser] V39.5 (Pass 2) Parsing Q&A content...")

    matches = list(HTML_SPLITTER_REGEX.finditer(qa_content_html))

    if not matches:
        print("  [Parser] V39.5 (Pass 2) No <h2> or <strong> tags found. No Q&A extracted.")
        return parsed_data  # Return metadata only

    print(f"  [Parser] V39.5 (Pass 2) Found {len(matches)} Q&A headers.")

    # --- "Last Question" logic ---
    last_valid_key = None
    last_valid_question_text = None
    # ---

    # --- V39.5: Main parsing loop ---
    i = 0
    while i < len(matches):
        current_match = matches[i]

        # 1. Get the Question (the tag itself)
        tag_html = current_match.group(1)

        # --- V39.5 FIX for Location(s) ---
        # Keep parentheticals for classification, remove bilingual
        question_text = _clean_content(tag_html, remove_bilingual=True, remove_parentheticals=False)

        # --- V39.4 LOGIC: Get the "in-between" answer FIRST ---
        content_start_index = current_match.end()
        content_end_index = len(qa_content_html)
        if (i + 1) < len(matches):
            content_end_index = matches[i + 1].start()

        answer_html_between = qa_content_html[content_start_index:content_end_index]
        # Clean the "in-between" text fully (remove parens)
        answer_text_between = _clean_content(answer_html_between, remove_bilingual=True, remove_parentheticals=True)
        # --- END V39.4 LOGIC ---

        # --- V39.4: Combine Split Header Logic (MODIFIED) ---
        combined_tag_match = current_match
        final_question_text = question_text

        # --- V39.4 USER'S FIX ---
        # ONLY run "combine" logic IF:
        # 1. Current question lacks a colon
        # 2. There is a *next* header
        # 3. The space between this header and the next is EMPTY
        if ":" not in question_text and (i + 1) < len(matches) and not answer_text_between:

            # Check the next header
            # --- V39.5 FIX ---
            # Keep parentheticals for classification
            next_question_text = _clean_content(matches[i + 1].group(1), remove_bilingual=True,
                                                remove_parentheticals=False)
            _nq_q, _nq_a, nq_found_answer_in_tag = _split_at_colon(next_question_text)

            # If it has a colon, *then* we combine
            if ":" in next_question_text and nq_found_answer_in_tag:
                print(
                    f"    [Parser] V39.5: Combining split question (no answer found): '{question_text[:30]}...' + '{next_question_text[:30]}...'")
                final_question_text = question_text + " " + next_question_text

                combined_tag_match = matches[i + 1]
                i += 1

                # --- V39.4: RE-FETCH ANSWER ---
                content_start_index = combined_tag_match.end()
                content_end_index = len(qa_content_html)
                if (i + 1) < len(matches):
                    content_end_index = matches[i + 1].start()

                answer_html_between = qa_content_html[content_start_index:content_end_index]
                # Clean the "in-between" text fully
                answer_text_between = _clean_content(answer_html_between, remove_bilingual=True,
                                                     remove_parentheticals=True)
                # --- END V39.4 RE-FETCH ---
        # --- END V39.4 FIX ---

        # --- Rule #2: Check for "Combined Q&A" (Answer *inside* the tag) ---
        # We use final_question_text (which has parens)
        final_question, answer_from_header, found_answer_in_tag = _split_at_colon(final_question_text)

        final_answer = ""

        if answer_text_between and found_answer_in_tag:
            # Clean the answer_from_header (remove parens)
            final_answer = _clean_content(answer_from_header, True, True) + "\n" + answer_text_between
        elif answer_text_between:
            # This is what "MYR 300,000" will trigger
            final_answer = answer_text_between
        elif found_answer_in_tag:
            # Clean the answer_from_header (remove parens)
            final_answer = _clean_content(answer_from_header, True, True)
        else:
            final_answer = ""
        # --- END ---

        # --- V39.5: Classify the header (which still has parens) ---
        header_key = _classify_header(final_question)

        # --- "Last Question" Heuristic ---
        if not final_answer:
            if header_key != "unclassified" and header_key != "unclassified_answer_marker":
                if last_valid_key is None:
                    last_valid_key = header_key
                    last_valid_question_text = final_question
                    print(f"    [Parser] V39.5: Holding key '{header_key}'")
                else:
                    print(f"    [Parser] V39.5: Chaining question to held key '{last_valid_key}'")
            i += 1
            continue

        # If we are here, final_answer IS NOT EMPTY.

        current_key = header_key
        current_question = final_question

        if last_valid_key and (header_key == "unclassified" or header_key == "unclassified_answer_marker"):
            current_key = last_valid_key
            current_question = last_valid_question_text
            print(f"    [Parser] V39.5: Re-assigning answer to '{current_key}'")
        # --- END Heuristic ---

        # --- Save the data ---
        # --- V39.5 ---
        # Now we fully clean the question text, removing parentheticals
        current_question = _clean_content(current_question, True, True)
        # Final answer is already clean

        if current_key == "unclassified":
            if "unclassified_chunks" not in parsed_data:
                parsed_data["unclassified_chunks"] = []
            parsed_data["unclassified_chunks"].append({
                "header": current_question,
                "content": final_answer
            })
        elif current_key not in metadata_keys:
            # --- V39.5 FIX ---
            # Do not overwrite a key that's already been set!
            # This stops "The project:" from overwriting "project_title"
            if current_key not in parsed_data:
                parsed_data[current_key] = final_answer
                print(f"    [Parser] V39.5: Saved '{current_key}'")
            else:
                print(
                    f"    [Parser] V39.5: Skipping save for '{current_key}', key already exists (e.g., project_title regression).")
        else:
            print(f"    [Parser] V39.5 (Pass 3) skipping '{current_key}', set by metadata pass.")

        # Reset the "hold" variables
        last_valid_key = None
        last_valid_question_text = None

        # Increment loop counter
        i += 1

    print(f"  [Parser] V39.5 Parsing complete.")

    # Post-process currency/number fields
    for key in ['project_duration_months', 'project_team_size', 'project_volunteer_size',
                'funding_requested_total', 'funding_pmf', 'funding_odf']:
        if key in parsed_data and isinstance(parsed_data[key], str):
            # Remove all non-numeric characters (except a decimal point)
            cleaned_num = re.sub(r'[^\d\.]', '', parsed_data[key]).strip()
            cleaned_num = cleaned_num.split('.')[0]  # Get integer part
            if cleaned_num:
                parsed_data[key] = cleaned_num

    return parsed_data