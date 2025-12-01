# yh_rag_cloud_api/parsers/progress_report_parser.py

import json
import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, ValidationError
import bs4


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
    funds_disbursed_to_date_myr: Optional[float] = Field(default=None,
                                                         description="The total funds 'Disbursement to-date (RM)'.")
    funds_utilized_to_date_myr: Optional[float] = Field(default=None,
                                                        description="The total 'Fund utilised to date (RM)'.")
    deliverables: List[Deliverable] = Field(default_factory=list,
                                            description="A list of all deliverables and their progress updates mentioned in the 'PROGRESS UPDATE' section.")
    lessons_learned_summary: Optional[str] = Field(
        alias="lessons_learnt",
        default=None,
        description="The full, raw text from the 'LESSONS LEARNT' section."
    )
    challenges_summary: Optional[str] = Field(
        alias="challenges",
        default=None,
        description="The full, raw text from the 'CHALLENGES' section."
    )
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
        # The answer is all content between this <b> tag and the next <b> tag

        # Find the end node (the next <b> tag)
        end_node = all_b_tags[i + 1] if i + 1 < len(all_b_tags) else None

        answer_nodes = []
        current_node = b_tag

        # Get content from the <b> tag's parent (e.g., the rest of the <p>)
        while current_node:
            if current_node == end_node:
                break

            # Find next sibling
            if current_node.next_sibling:
                current_node = current_node.next_sibling
            # If no more siblings, go to the parent's next sibling
            elif current_node.parent:
                current_node = current_node.parent.next_sibling
            else:
                break

            # If we hit the end_node, stop
            if current_node == end_node:
                break

            # Check if the node is a Tag before checking descendants
            if end_node and isinstance(current_node, bs4.Tag) and end_node in current_node.descendants:
                # The end_node is *inside* this node, so we shouldn't just append the whole node.
                # This is too complex for now, so we just stop.
                break

            answer_nodes.append(current_node)

        # Combine all answer nodes into a single string
        answer_text = " ".join(
            node.get_text(separator=' ', strip=True) for node in answer_nodes
        )

        # Add to map and reset for next question
        qa_map[current_q_text] = answer_text.strip()
        current_q_text = ""

    return qa_map


# ---
# 3. DEFINE THE MAIN PARSER FUNCTION (HTML-BASED)
# ---

def parse_progress_report(raw_report_content: str) -> Dict[str, Any]:
    """
    Parses a raw progress report (HTML) into a structured dictionary
    using HTML tags and flexible regex.

    Args:
        raw_report_content: The raw HTML string content from the file.

    Returns:
        A dictionary conforming to the CanonicalProgressReport schema.
    """
    print(f"[parse_progress_report] Starting HTML-based parsing... (length: {len(raw_report_content)})")

    # --- DEBUG PRINT FOR RAW HTML ---
    print("\n" + "=" * 20 + " BEGIN RAW HTML FOR PARSER " + "=" * 20)
    print(raw_report_content)
    print("=" * 20 + "  END RAW HTML FOR PARSER  " + "=" * 20 + "\n")
    # --- END DEBUG PRINT ---

    # 1. Build the Question-Answer Map
    try:
        qa_map = _build_qa_map(raw_report_content)

        # --- DEBUG: Print the generated Q&A Map ---
        print("\n" + "=" * 20 + " BEGIN GENERATED Q&A MAP " + "=" * 20)
        print(json.dumps(qa_map, indent=2))
        print("=" * 20 + "  END GENERATED Q&A MAP  " + "=" * 20 + "\n")
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

    # --- NEW HELPER ---
    def get_int(value_str: Optional[str]) -> Optional[int]:
        """Converts a string to an int, stripping non-numeric chars."""
        if value_str is None:
            return None
        try:
            # Strips "Report #"  or any other text, leaving only the number
            return int(re.sub(r"[^0-9]", "", value_str))
        except (ValueError, TypeError):
            return None

    # --- END NEW HELPER ---

    # 3. Initialize data dictionary
    report_data = {}

    # 4. Parse Metadata
    # --- UPDATED: Use get_int helper ---
    report_num_str = find_answer(r"This report is #")
    report_data["report_number"] = get_int(report_num_str)
    # --- END UPDATE ---

    report_data["report_date"] = find_answer(r"Report Date \(Do Not Delete\)")
    report_data["project_title"] = find_answer(r"Project title")
    report_data["organisation_name"] = find_answer(r"Name of Organisation")

    period_str = find_answer(r"monitoring period")
    if period_str:
        try:
            # Simple split, can be improved
            start_str, end_str = period_str.split(' - ')
            report_data["reporting_period_start"] = start_str.strip()
            report_data["reporting_period_end"] = end_str.strip()
        except (ValueError, AttributeError):
            report_data["reporting_period_start"] = period_str
            report_data["reporting_period_end"] = period_str

    report_data["funds_disbursed_to_date_myr"] = get_float(find_answer(r"Disbursement to-date"))
    report_data["funds_utilized_to_date_myr"] = get_float(find_answer(r"Fund utilised to date"))

    # 5. Parse Section 1: Deliverables
    deliverables_list = []
    num_deliverables_str = find_answer(r"Number of deliverable\(s\)")

    if num_deliverables_str:
        try:
            num_deliverables = int(re.sub(r'[^0-9]', '', num_deliverables_str))
            for i in range(1, num_deliverables + 1):
                desc = find_answer(fr"state deliverable \s*#{i}")

                # --- FIX: Make regex flexible for "deliverable" vs "deliveravle" ---
                update = find_answer(fr"Update for delivera[bv]le(?:s)? \s*#{i}")

                if not desc and not update:
                    continue  # Skip if both are missing

                # Infer Status from update text
                status = "Ongoing"  # Default
                if update:
                    if re.search(r"#\s*This deliverable has been met", update, re.I):
                        status = "Met"
                    elif re.search(r"#\s*This deliverable has been partially met", update, re.I):
                        status = "Partially Met"
                    elif re.search(r"#\s*This deliverable has not been met", update, re.I):
                        status = "Not Met"

                deliverables_list.append(Deliverable(
                    description=desc if desc else "Description not found",
                    status=status,
                    progress_update=update if update else "Update not found"
                ))
        except Exception as e:
            print(f"Error parsing deliverables: {e}")

    report_data["deliverables"] = deliverables_list

    # 6. Parse Section 2: Beneficiaries
    beneficiaries_list = []
    num_beneficiaries_str = find_answer(r"How many types of beneficiaries")

    if num_beneficiaries_str:
        try:
            num_beneficiaries = int(re.sub(r'[^0-9]', '', num_beneficiaries_str))
            for i in range(1, num_beneficiaries + 1):
                b_type = find_answer(fr"Type of beneficiaries \s*#{i}")
                b_count_str = find_answer(fr"No\. of beneficiaries \s*#{i}")
                b_metrics = find_answer(fr"additional data.*beneficiary.*#{i}")

                if not b_type:
                    continue  # Skip if type is missing

                b_count = 0
                if b_count_str:
                    try:
                        b_count = int(re.sub(r'[^0-9]', '', b_count_str))
                    except ValueError:
                        b_count = 0

                beneficiaries_list.append(Beneficiary(
                    type=b_type,
                    count=b_count,
                    metrics_tracked=b_metrics if b_metrics else "Not specified"
                ))
        except Exception as e:
            print(f"Error parsing beneficiaries: {e}")

    report_data["beneficiaries"] = beneficiaries_list

    # 7. Parse Sections 3, 4, 5 (Qualitative)
    report_data["lessons_learned_summary"] = find_answer(r"3\.\s*LESSONS LEARNT")
    report_data["challenges_summary"] = find_answer(r"4\.\s*CHALLENGES")
    report_data["success_stories_summary"] = find_answer(r"5\.\s*SUCCESS STORIES")

    # 8. Validate and Return
    try:
        # Validate the data we just parsed
        report_model = CanonicalProgressReport(**report_data)
        validated_data = report_model.model_dump()

        print(
            f"[parse_progress_report] HTML Q&A parsing and validation successful. Org: {validated_data.get('organisation_name')}")
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