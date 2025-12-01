# yh_rag_cloud_api/parsers/progress_report_parser_production.py

import json
import re
import unicodedata
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, ValidationError
import bs4

# ---
# IMPORT THE CLOUD LLM FOR DELIVERABLES AND BENEFICIARIES
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
# 1. DEFINE THE CANONICAL SCHEMA (Unchanged)
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

    # --- FIELD UPDATED: Changed from str to int ---
    report_number: Optional[int] = Field(default=None,
                                         description="The sequential number of the report (e.g., 3).")
    # --- END UPDATE ---

    report_date: Optional[str] = Field(
        description="The single date the report was submitted, normalized to 'YYYY-MM-DD'. Find 'Report Date (Do Not Delete)' or similar fields.")
    reporting_period_start: Optional[str] = Field(default=None,
                                                  description="The start date of the monitoring period this report covers, normalized to 'YYYY-MM-DD'.")
    reporting_period_end: Optional[str] = Field(default=None,
                                                description="The end date of the monitoring period this report covers, normalized to 'YYYY-MM-DD'.")
    project_title: Optional[str] = Field(default=None, description="The official title of the project.")
    organisation_name: Optional[str] = Field(default=None,
                                             description="The name of the organisation submitting the report.")

    # --- NEW FIELDS (REQUEST 3) ---
    grant_amount_myr: Optional[float] = Field(
        default=None,
        description="The total grant amount (RM)."
    )
    funds_unutilized_to_date_myr: Optional[float] = Field(
        default=None,
        description="The total 'Fund unutilised to date (RM)'."
    )
    number_of_disbursement_to_date: Optional[int] = Field(
        default=None,
        description="The 'Number of disbursement to-date'."
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
        description="Number of deliverables for this report"
    )
    # --- END NEW FIELDS ---

    # --- NEW FIELDS FOR FINAL REPORT ---
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

    # --- MODIFIED: Section 3 is now broken into sub-questions ---
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
    # --- END MODIFICATION ---

    sustainability_scalability_summary: Optional[str] = Field(
        default=None,
        description="Full text from the '5. Sustainability and Scalability' section."
    )
    # --- END NEW FIELDS ---

    funds_disbursed_to_date_myr: Optional[float] = Field(default=None,
                                                         description="The total funds 'Disbursement to-date (RM)'.")
    funds_utilized_to_date_myr: Optional[float] = Field(default=None,
                                                        description="The total 'Fund utilised to date (RM)'.")
    deliverables: List[Deliverable] = Field(default_factory=list,
                                            description="A list of all deliverables and their progress updates mentioned in the 'PROGRESS UPDATE' section.")

    # --- NOTE: These fields are re-purposed for Final Reports to hold 4a and 4b ---
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
    # --- END NOTE ---

    success_stories_summary: Optional[str] = Field(
        alias="success_stories",
        default=None,
        description="The full, raw text from the 'SUCCESS STORIES' section."
    )
    beneficiaries: List[Beneficiary] = Field(default_factory=list,
                                             description="A list of all beneficiary types from the 'BENEFICIARIES' section.")


# ---
# 2. DEFINE NEW HTML-BASED HELPER
# ---

def _build_qa_map(raw_html: str) -> Dict[str, str]:
    """
    Builds a map of {question: answer} from the HTML content.
    - Questions are identified by <b> tags.
    - Answers are the text between one <b> tag and the next.
    """

    # Step 1: Remove bilingual prompts first
    cleaned_html = re.sub(r'\[\s*.*?\]', '', raw_html)

    soup = bs4.BeautifulSoup(cleaned_html, 'html.parser')

    # Find all <b> tags, which we assume are questions
    all_b_tags = soup.find_all('b')

    qa_map = {}
    current_q_text = ""

    for i, b_tag in enumerate(all_b_tags):

        # --- 1. Find the Question ---
        # Accumulate <b> tags to handle multi-line questions
        q_text = b_tag.get_text(separator=' ', strip=True)
        current_q_text += " " + q_text
        current_q_text = current_q_text.strip()

        # If question doesn't end with ':', it's multi-line. Keep accumulating.
        if not current_q_text.endswith(':'):
            continue

        # Clean the final question text
        current_q_text = current_q_text.replace(':', '').strip()

        # --- 2. Find the Answer ---
        # --- MODIFICATION: New logic to prevent including question text in answer ---

        # Find the end node (the next <b> tag)
        end_node = all_b_tags[i + 1] if i + 1 < len(all_b_tags) else None
        answer_nodes = []

        # Start iterating from the <b> tag itself
        current_element = b_tag

        while current_element:
            # Get the next element in the document
            current_element = current_element.next_element

            # Stop conditions
            if current_element is None:
                break
            if current_element == end_node:
                break

            # Check if this element is a text node
            if isinstance(current_element, bs4.NavigableString):
                # Check if this text is a child of the START <b> tag (ignore it)
                is_child_of_start_node = False
                for parent in current_element.parents:
                    if parent == b_tag:
                        is_child_of_start_node = True
                        break
                if is_child_of_start_node:
                    continue

                # Check if this text is a child of the END <b> tag (stop)
                if end_node:
                    is_child_of_end_node = False
                    for parent in current_element.parents:
                        if parent == end_node:
                            is_child_of_end_node = True
                            break
                    if is_child_of_end_node:
                        break  # Stop collecting, we've hit the next question's text

                # If it's not part of the start or end tag, it's answer text
                stripped_text = current_element.strip()
                if stripped_text:
                    answer_nodes.append(stripped_text)

        # Combine, avoiding double-spacing
        answer_text = " ".join(answer_nodes)
        answer_text = re.sub(r'\s+', ' ', answer_text).strip()
        # --- END MODIFICATION ---

        # Add to map and reset for next question
        final_answer = answer_text.strip()

        if current_q_text not in qa_map:
            # 1. It's a new question. Add it, even if the answer is empty.
            qa_map[current_q_text] = final_answer
        else:
            # 2. The question already exists in the map.
            existing_answer = qa_map[current_q_text]
            if not existing_answer and final_answer:
                # The existing answer was empty, but this new one is not. Update it.
                qa_map[current_q_text] = final_answer
            # (Else) The existing answer already has a value. We follow the "first-one-wins"
            # rule, so we do *nothing* and keep the original (non-empty) value.

        current_q_text = ""

    return qa_map


# ---
# 3. CLOUD-STYLE EXTRACTION FOR DELIVERABLES AND BENEFICIARIES
# ---

def clean_control_chars(s: str) -> str:
    """Removes common control characters except newline and tab from a string."""
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in ('\n', '\t'))


def _extract_complex_sections_with_llm(report_text: str, section_type: str) -> List[Dict[str, Any]]:
    """
    Uses LLM to extract complex sections like deliverables and beneficiaries.
    This replaces the regex-based extraction with a more robust approach.
    """
    print(f"  Using LLM to extract {section_type}...")

    if section_type == "deliverables":
        schema_example = '''
        [
            {
                "description": "Description of deliverable 1",
                "status": "Met",
                "progress_update": "Update text for deliverable 1"
            },
            {
                "description": "Description of deliverable 2", 
                "status": "Ongoing",
                "progress_update": "Update text for deliverable 2"
            }
        ]
        '''
        extraction_prompt = f"""
        Extract all deliverables from the progress report text below.
        For each deliverable, provide:
        - description: The description of the deliverable or milestone
        - status: Normalized to one of: ['Met', 'Partially Met', 'Not Met', 'Ongoing', 'Delayed', 'Not Applicable']
        - progress_update: The full text of the progress update provided

        Return ONLY a valid JSON array of deliverable objects. If no deliverables are found, return an empty array.

        Example format:
        {schema_example}

        Report Text:
        ---
        {report_text}
        ---

        JSON array:
        """
    else:  # beneficiaries
        schema_example = '''
        [
            {
                "type": "Students",
                "count": 150,
                "metrics_tracked": "Learning outcomes and attendance"
            },
            {
                "type": "Teachers",
                "count": 10, 
                "metrics_tracked": "Training participation and feedback"
            }
        ]
        '''
        extraction_prompt = f"""
        Extract all beneficiary types from the progress report text below.
        For each beneficiary type, provide:
        - type: The category of beneficiary (e.g., 'Students', 'Teachers', 'Parents')
        - count: The number of beneficiaries in this category (extract as integer)
        - metrics_tracked: What is being tracked for this group

        Return ONLY a valid JSON array of beneficiary objects. If no beneficiaries are found, return an empty array.

        Example format:
        {schema_example}

        Report Text:
        ---
        {report_text}
        ---

        JSON array:
        """

    try:
        response_text = ask_gemini(extraction_prompt)
        json_str_raw = response_text.strip().lstrip("```json").rstrip("```").strip()
        json_str_cleaned = clean_control_chars(json_str_raw)

        extracted_data = json.loads(json_str_cleaned)
        if isinstance(extracted_data, list):
            print(f"  Successfully extracted {len(extracted_data)} {section_type}")
            return extracted_data
        else:
            print(f"  WARN: LLM did not return array for {section_type}")
            return []

    except Exception as e:
        print(f"  ERROR: Failed to extract {section_type} with LLM: {e}")
        return []


# ---
# 4. DEFINE PARSER-SPECIFIC HELPERS (UPDATED)
# ---

def _parse_deliverables_section(report_data: Dict, text_report_content: str) -> Dict:
    """
    Parses the deliverables section using LLM extraction instead of regex.
    """
    print("  Extracting deliverables using LLM...")
    deliverables_list = []

    # Extract using LLM
    extracted_deliverables = _extract_complex_sections_with_llm(text_report_content, "deliverables")

    for item in extracted_deliverables:
        try:
            deliverable = Deliverable(
                description=item.get("description", "Description not found"),
                status=item.get("status", "Ongoing"),
                progress_update=item.get("progress_update", "Update not found")
            )
            deliverables_list.append(deliverable)
        except Exception as e:
            print(f"    WARN: Failed to create deliverable from LLM output: {e}")

    report_data["deliverables"] = deliverables_list
    report_data["number_of_deliverables"] = len(deliverables_list)

    return report_data


def _parse_mid_progress_sections(report_data: Dict, find_answer: callable, text_report_content: str) -> Dict:
    """
    Parses the sections specific to a MID-PROGRESS report
    (i.e., Beneficiaries Q&A, old qualitative sections).
    """
    print("[parse_progress_report] Parsing as MID-PROGRESS report.")
    report_data["report_type"] = "mid-progress"

    # 6. Parse Section 2: Beneficiaries using LLM instead of regex
    print("  Extracting beneficiaries using LLM...")
    beneficiaries_list = []

    extracted_beneficiaries = _extract_complex_sections_with_llm(text_report_content, "beneficiaries")

    for item in extracted_beneficiaries:
        try:
            beneficiary = Beneficiary(
                type=item.get("type", "Unknown"),
                count=item.get("count"),
                metrics_tracked=item.get("metrics_tracked", "Not specified")
            )
            beneficiaries_list.append(beneficiary)
        except Exception as e:
            print(f"    WARN: Failed to create beneficiary from LLM output: {e}")

    report_data["beneficiaries"] = beneficiaries_list

    # 7. Parse Sections 3, 4, 5 (Qualitative) - keeping regex for these
    report_data["lessons_learned_summary"] = find_answer(r"3\.\s*LESSONS LEARNT")
    report_data["challenges_summary"] = find_answer(r"4\.\s*CHALLENGES")
    report_data["success_stories_summary"] = find_answer(r"5\.\s*SUCCESS STORIES")

    return report_data


def _parse_final_report_sections(report_data: Dict, find_answer: callable) -> Dict:
    """
    Parses the sections specific to a FINAL report
    (i.e., Exec Summary, Effectiveness, new qualitative sections).

    This version populates sub-questions for sections 3 and 4.
    """
    print("[parse_progress_report] Parsing as FINAL report.")
    report_data["report_type"] = "final"

    # Parse new sections
    report_data["executive_summary"] = find_answer(r"1\.\s*Executive Summary")
    report_data["effectiveness_summary"] = find_answer(r"2\.\s*Effectiveness")

    # --- UPDATED LOGIC FOR SECTION 3 (Sub-questions) ---
    # Find 3a (which includes main question 3)
    report_data["methodology_that_work_and_not_work"] = find_answer(
        r"3\.\s*Project methodology.*a\)\s*What methods and approaches worked")
    # Find 3b (anchored to start)
    report_data["methodology_right_intervention_and_target_or_done_differently"] = find_answer(
        r"^b\)\s*Was the project the right intervention")
    # Find 3c (anchored to start)
    report_data["methodology_successes_and_challenges"] = find_answer(
        r"^c\)\s*What successes and challenges.*monitoring")

    # --- UPDATED LOGIC FOR SECTION 4 (Sub-questions) ---
    # Find 4a (which includes main question 4) and put in lessons_learned
    report_data["lessons_learned_summary"] = find_answer(r"4\.\s*Lessons Learnt.*a\)\s*Please share lessons")
    # Find 4b (anchored to start) and put in challenges
    report_data["challenges_summary"] = find_answer(r"^b\)\s*Elaborate on the challenges")
    # --- END UPDATED LOGIC ---

    # ISKUL has "5. Success Stories" / GSS has "6. Success Stories"
    report_data["success_stories_summary"] = find_answer(r"\d\.\s*Success Stories")

    # GSS has "5. Sustainability and Scalability"
    report_data["sustainability_scalability_summary"] = find_answer(
        r"5\.\s*Sustainability and Scalability"
    )

    return report_data


# ---
# 5. DEFINE THE MAIN PARSER FUNCTION (HTML-BASED)
# ---

def parse_progress_report_hybrid(html_report_content: str, text_report_content: str) -> Dict[str, Any]:
    """
    Parses a raw progress report (HTML) into a structured dictionary
    using HTML tags and flexible regex.

    This function now detects whether it is a "mid-progress" or "final"
    report and routes parsing logic accordingly.

    Args:
        text_report_content: The raw text string content from the file
        html_report_content: The raw HTML string content from the file.

    Returns:
        A dictionary conforming to the CanonicalProgressReport schema.
    """
    print(f"[parse_progress_report_hybrid] Starting HTML-based parsing... (length: {len(html_report_content)})")

    # --- DEBUG PRINT FOR RAW HTML ---
    # print("\n" + "=" * 20 + " BEGIN RAW HTML FOR PARSER " + "=" * 20)
    # print(raw_report_content)
    # print("=" * 20 + "  END RAW HTML FOR PARSER  " + "=" * 20 + "\n")
    # --- END DEBUG PRINT ---

    # 1. Build the Question-Answer Map
    try:
        qa_map = _build_qa_map(html_report_content)

        # --- DEBUG: Print the generated Q&A Map ---
        # print("\n" + "=" * 20 + " BEGIN GENERATED Q&A MAP " + "=" * 20)
        # print(json.dumps(qa_map, indent=2))
        # print("=" * 20 + "  END GENERATED Q&A MAP  " + "=" * 20 + "\n")
        # --- END DEBUG PRINT ---

    except Exception as e:
        print(f"CRITICAL: Failed to build Q&A map: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to parse HTML structure: {e}"}

    # 2. Define helper functions
    def find_answer(key_regex: str) -> Optional[str]:
        """Finds an answer in the map using a flexible regex for the key."""
        for q_text, a_text in qa_map.items():
            if re.search(key_regex, q_text, re.IGNORECASE):
                return a_text
        return None

    def get_float(value_str: Optional[str]) -> Optional[float]:
        """Converts currency string to float."""
        if value_str is None:
            return None
        try:
            return float(re.sub(r"[^0-9.-]", "", value_str))
        except (ValueError, TypeError):
            return None

    def get_int(value_str: Optional[str]) -> Optional[int]:
        """Converts a string to an int, stripping non-numeric chars."""
        if value_str is None:
            return None
        try:
            return int(re.sub(r"[^0-9]", "", value_str))
        except (ValueError, TypeError):
            return None

    # 3. Initialize data dictionary
    report_data = {}

    # 4. Parse COMMON Metadata
    # (This section is common to both mid-progress and final reports)
    report_num_str = find_answer(r"This report is #")
    report_data["report_number"] = get_int(report_num_str)

    # Use "Date report submitted" for final reports, fallback to "Report Date"
    report_data["report_date"] = find_answer(r"Date report submitted")
    if not report_data["report_date"]:
        report_data["report_date"] = find_answer(r"Report Date \(Do Not Delete\)")

    report_data["project_title"] = find_answer(r"Project title|Project Name")
    report_data["organisation_name"] = find_answer(r"Name of Organisation")

    # Use " "monitoring period"
    period_str = find_answer(r"monitoring period")
    print(f"[DEBUG] Monitoring period found: '{period_str}'")

    # If monitoring period is not found, then try project duration as fallback
    # if not period_str:
    #     period_str = find_answer(r"Project duration")
    #     print(f"[DEBUG] Project duration found: '{period_str}'")

    if period_str:
        print(f"[DEBUG] Processing period string: '{period_str}'")
        try:
            # Handle "Nov 2022 - Feb 2023" format by preserving the dash
            if ' - ' in period_str:
                print(f"[DEBUG] Found dash format, splitting on ' - '")
                parts = period_str.split(' - ')
                print(f"[DEBUG] Parts after dash split: {parts}")
                if len(parts) == 2:
                    start_str = parts[0].strip()
                    end_str = parts[1].strip()
                    print(f"[DEBUG] Start: '{start_str}', End: '{end_str}'")
                else:
                    start_str = period_str
                    end_str = period_str
                    print(f"[DEBUG] Unexpected parts count, using original")
            else:
                # Fallback to original logic for other formats
                print(f"[DEBUG] No dash found, using space splitting")
                period_str = period_str.replace(' - ', ' ')
                parts = period_str.split()
                print(f"[DEBUG] Parts after space split: {parts}")
                if len(parts) >= 4:
                    start_str = f"{parts[0]} {parts[1]}"
                    end_str = f"{parts[2]} {parts[3]}"
                elif len(parts) == 2:
                    start_str, end_str = parts
                else:
                    start_str = period_str
                    end_str = period_str

                start_str = start_str.strip()
                end_str = end_str.strip()
                print(f"[DEBUG] Start: '{start_str}', End: '{end_str}'")

            # Ensure both dates have years
            end_year_match = re.search(r'(\b\d{4}\b)', end_str)
            end_year = end_year_match.group(1) if end_year_match else None
            print(f"[DEBUG] End year found: '{end_year}'")

            if end_year:
                start_year_match = re.search(r'(\b\d{4}\b)', start_str)
                if not start_year_match:
                    start_str = f"{start_str} {end_year}"
                    print(f"[DEBUG] Added end year to start: '{start_str}'")

            report_data["reporting_period_start"] = start_str
            report_data["reporting_period_end"] = end_str
            print(f"[DEBUG] Final - Start: '{start_str}', End: '{end_str}'")

        except (ValueError, AttributeError) as e:
            print(f"[DEBUG] Error processing period: {e}")
            report_data["reporting_period_start"] = period_str
            report_data["reporting_period_end"] = period_str
    else:
        print("[DEBUG] No period string found")

    report_data["grant_amount_myr"] = get_float(find_answer(r"Grant amount"))
    report_data["funds_disbursed_to_date_myr"] = get_float(find_answer(r"disburse.*to.*date"))
    report_data["funds_utilized_to_date_myr"] = get_float(find_answer(r"utilised.*to.*date"))
    report_data["funds_unutilized_to_date_myr"] = get_float(find_answer(r"unutilised fund"))
    report_data["number_of_disbursement_to_date"] = get_int(find_answer(r"number.*disbursement"))
    report_data["month_year_of_disbursement"] = find_answer(r"month.*year.*disbursement")

    # Handle project locations (list)
    locations_str = find_answer(r"Location")  # "Location:" is simpler
    if not locations_str:
        locations_str = find_answer(r"location.*of.*project")  # Fallback

    if locations_str:
        locations_str = re.split(r'1\.\s*(PROGRESS UPDATE|EXECUTIVE SUMMARY)', locations_str, flags=re.IGNORECASE)[0]
        location_list = re.split(r'[^a-zA-Z0-9\s-]+', locations_str)
        report_data["project_locations"] = [
            loc.strip() for loc in location_list if loc.strip()
        ]
    else:
        report_data["project_locations"] = []

    # ---
    # 5. DETECT REPORT TYPE & ROUTE TO SPECIFIC PARSERS
    # ---

    is_final_report = False
    for q_text in qa_map.keys():
        if re.search(r"1\.\s*Executive Summary", q_text, re.IGNORECASE):
            is_final_report = True
            break

    # ---
    # 6. PARSE DIVERGENT SECTIONS
    # ---

    if is_final_report:
        report_data = _parse_final_report_sections(report_data, find_answer)
    else:
        report_data = _parse_mid_progress_sections(report_data, find_answer, text_report_content)

    # ---
    # 7. PARSE COMMON SECTIONS (Deliverables) - NOW USING LLM
    # ---

    # The deliverables section is common to both report types, now using LLM
    report_data = _parse_deliverables_section(report_data, text_report_content)

    # 8. Validate and Return
    try:
        # Validate the data we just parsed
        report_model = CanonicalProgressReport(**report_data)
        validated_data = report_model.model_dump()

        print(
            f"[parse_progress_report_hybrid] HTML Q&A parsing and validation successful. "
            f"Type: {validated_data.get('report_type')}. "
            f"Org: {validated_data.get('organisation_name')}"
        )
        return validated_data

    except ValidationError as pydantic_err:
        print(f"ERROR: Failed to validate extracted data: {pydantic_err}")
        # Return the raw extracted data for debugging
        return {"error": f"Schema validation failed: {pydantic_err}",
                "raw_extracted_data": report_data}
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during parsing: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"An unexpected error occurred: {e}"}