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
# V39.0 PARSER IMPLEMENTATION (Corrected Hybrid Logic)
# ---

def clean_control_chars(s: str) -> str:
    """Removes common control characters except newline and tab from a string."""
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in ('\n', '\t'))


# --- This is the _clean_content from V-1300hrs ---
def _clean_content(text: str, remove_bilingual=True) -> str:
    """
    Cleans HTML artifacts *after* parsing.
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
    # V-1300hrs had this, so we keep it.
    text = re.sub(r'\([^)]*\)', '', text, flags=re.DOTALL)

    # 5. Remove artifact lines/separators like '---' or '***'
    text = re.sub(r'^\s*[\-_\*]{3,}\s*$', '', text, flags=re.MULTILINE)

    # 6. Remove markdown (which shouldn't be here, but as a safeguard)
    text = text.replace('**', '').replace('##', '').replace('>', '')

    # 7. Collapse newlines and whitespace
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        # (FIX) This regex correctly skips stray colons/punctuation
        if not stripped_line or re.fullmatch(r'^\s*[:\.,]\s*$', stripped_line):
            continue
        cleaned_lines.append(stripped_line)

    text = '\n'.join(cleaned_lines)

    # 8. Final cleanup
    text = re.sub(r"(\n\s*){2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip().rstrip(':')

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

# --- V39.0: Header classification map (Merged) ---
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
    "linked to partner profile": "unclassified"
}


def _classify_header(header_text: str) -> str:
    """
    (V39.0)
    Uses a fast string-matching map first.
    Expects CLEANED TEXT as input, not HTML.
    """
    # Sanitize the header for matching
    clean_header_text = re.sub(r'\s+', ' ', header_text).strip().lower()[:500]

    # We leave bilingual text in for classification
    # clean_header_text = re.sub(r'\[.*?\]', '', clean_header_text)  # Remove [...]

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


# --- FUNCTION TO HANDLE COMBINED Q:A (V39.0 - STABLE) ---
def _split_at_colon(cleaned_text: str) -> Tuple[str, str, bool]:
    """
    (V39.0)
    Implements the user's Rule #2.
    Checks for the "Combined" Q:A format (e.g., "Q: A")
    Expects pre-cleaned text (no HTML, but WITH bilingual text).
    Returns (Question, Answer, found_answer_in_tag)
    """

    # We do NOT clean bilingual text here, because the classifier needs it
    # to find the right key (e.g., "financial –")

    text = re.sub(r'\s+', ' ', cleaned_text).strip()

    if ":" in text:
        parts = text.rsplit(":", 1)  # Split on the *last* colon
        if len(parts) > 1:
            question = parts[0].strip()
            answer = parts[1].strip()

            # This is the key: did we find a *real* answer, or just a colon?
            if answer:
                return question, answer, True  # Found answer *inside* tag
            else:
                return question, "", False  # Found a colon, but no answer

    # No colon found, or no answer after colon
    return cleaned_text, "", False


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
    (V39.0 - Corrected Hybrid Logic)
    Implements the user's explicit rules by combining:
    1. (Pass 1) V27.0 Metadata parser (stable)
    2. (Pass 2) V-1300hrs `finditer` logic (stable for Impact Area/SDG)
    3. (Pass 3) V39.0 `_split_at_colon` logic (stable for Funding/PMF/ODF)
    """
    print("  [Parser] Running V39.0 (Corrected Hybrid Logic)...")
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
        print("  [Parser] V39.0 (Pass 1) WARNING: No <table> found. Parsing all content as Q&A.")
        qa_content_html = proposal_text

    metadata_keys = set(metadata.keys())

    # --- Step 2: (Pass 2 - finditer Q&A Extraction) ---
    print("  [Parser] V39.0 (Pass 2) Parsing Q&A content...")

    matches = list(HTML_SPLITTER_REGEX.finditer(qa_content_html))

    if not matches:
        print("  [Parser] V39.0 (Pass 2) No <h2> or <strong> tags found. No Q&A extracted.")
        return parsed_data  # Return metadata only

    print(f"  [Parser] V39.0 (Pass 2) Found {len(matches)} Q&A headers.")

    # --- V39.0 "Last Question" logic ---
    last_valid_key = None
    last_valid_question_text = None
    # ---

    for i in range(len(matches)):
        current_match = matches[i]

        # 1. Get the Question (the tag itself)
        tag_html = current_match.group(1)
        # Get raw text, but keep bilingual for classification
        question_text = _clean_content(tag_html, remove_bilingual=False)

        # 2. Get the Answer (the text *between* this tag and the next)
        content_start_index = current_match.end()
        content_end_index = -1

        if (i + 1) < len(matches):
            content_end_index = matches[i + 1].start()
        else:
            content_end_index = len(qa_content_html)

        answer_html_between = qa_content_html[content_start_index:content_end_index]
        answer_text_between = _clean_content(answer_html_between)

        # --- Rule #2: Check for "Combined Q&A" (Answer *inside* the tag) ---
        final_question, answer_from_header, found_answer_in_tag = _split_at_colon(question_text)

        final_answer = ""
        # --- NEW V39.0 LOGIC ---
        if answer_text_between:
            # Rule 3 (Answer Between) ALWAYS takes priority.
            # This fixes PMF, ODF, Impact Area, SDG Goals.
            final_answer = answer_text_between
        elif found_answer_in_tag:
            # Rule 2 (Answer Inside) is the fallback.
            # This fixes Requested Funding.
            final_answer = answer_from_header
        else:
            # No answer found anywhere.
            final_answer = ""
        # --- END NEW V39.0 LOGIC ---

        header_key = _classify_header(final_question)

        # --- "Last Question" Heuristic ---
        if not final_answer:
            # This is a header with no answer (e.g., "Project Goal")
            if header_key != "unclassified" and header_key != "unclassified_answer_marker":
                # It's a *valid* header, so we "hold" it.
                last_valid_key = header_key
                last_valid_question_text = final_question
                print(f"    [Parser] V39.0: Holding key '{header_key}'")
            # If it's unclassified (or a stray ':'), we just ignore it.
            continue

        # If we are here, final_answer IS NOT EMPTY.

        current_key = header_key
        current_question = final_question

        if header_key == "unclassified_answer_marker":
            # This is an "answer" (like the stray ":")
            # Check if we are "holding" a valid key from the previous tag.
            if last_valid_key:
                # Yes. This answer belongs to the *previous* question.
                current_key = last_valid_key
                current_question = last_valid_question_text
                print(f"    [Parser] V39.0: Re-assigning answer to '{current_key}'")
            else:
                # This is an orphan ':', treat as unclassified
                current_key = "unclassified"

        # --- Save the data ---
        # Clean the final Q&A *after* all logic, before saving
        current_question = _clean_content(current_question)  # Now remove bilingual
        final_answer = _clean_content(final_answer)  # Clean the answer

        if current_key == "unclassified":
            if "unclassified_chunks" not in parsed_data:
                parsed_data["unclassified_chunks"] = []
            parsed_data["unclassified_chunks"].append({
                "header": current_question,  # Use the clean question
                "content": final_answer
            })
        elif current_key not in metadata_keys:
            parsed_data[current_key] = final_answer
            print(f"    [Parser] V39.0: Saved '{current_key}'")
        else:
            # This handles the case where the *metadata table* was also
            # parsed in Pass 2 (e.g., "Organisation name..."). We
            # ignore it here because Pass 1 already got it.
            print(f"    [Parser] V39.0 (Pass 3) skipping '{current_key}', set by metadata pass.")

        # Finally, reset the "hold" variables, as this answer "closes" the question.
        last_valid_key = None
        last_valid_question_text = None
        # --- END V39.0 LOGIC ---

    print(f"  [Parser] V39.0 Parsing complete.")

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