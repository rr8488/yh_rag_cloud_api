# yh_rag_cloud_api/parsers/proposal_parser_production.py

import json
import re
from typing import Dict, Any, List, Optional
from ..rag_utils import ask_gemini


def parse_proposal_cloud(proposal_text: str) -> Dict[str, Any]:
    """
    Parse proposal document using Gemini to extract structured fields.
    """
    prompt = f"""
    EXTRACT STRUCTURED DATA from this grant proposal document. Return ONLY valid JSON.

    PROPOSAL TEXT:
    {proposal_text[:15000]}

    INSTRUCTIONS:
    1. Extract the following fields from the proposal text
    2. Return ONLY valid JSON format
    3. Use null for missing fields
    4. Keep text concise but preserve key information

    REQUIRED FIELDS:
    {{
        "organisation_name": "string - name of applying organization",
        "company_id": "string - company registration ID if mentioned", 
        "project_title": "string - title of the project",
        "grant_application_id": "string - application reference if provided",
        "past_partner_status": "string - whether past/current partner of Hasanah",
        "collaboration_status": "string - whether implementing with other organizations",
        "funding_requested": "number - requested funding amount in MYR",
        "project_duration": "number - duration in months", 
        "project_location_list": ["string", "string"] - Return a LIST of Malaysian States involved (e.g. ["Sabah", "Kuala Lumpur"]). If nationwide, return ["Malaysia"],
        "location_details": "string - specific village/town/district details",
        "team_size": "number - number of team members/staff",
        "volunteer_count": "number - number of volunteers",
        "project_goal": "string - overall project goal",
        "project_summary": "string - brief project summary", 
        "project_description": "string - detailed project description",
        "impact_area": "string - selected impact area",
        "impact_alignment": "string - how project aligns with impact area",
        "sdg_alignment": "string - relevant Sustainable Development Goals",
        "beneficiaries": "string - description and count of beneficiaries",
        "beneficiary_involvement": "string - how beneficiaries are involved",
        "beneficiary_benefits": "string - how beneficiaries benefit",
        "problem_analysis": "string - situation and problem analysis", 
        "solution_approach": "string - how project addresses the problem",
        "organizational_expertise": "string - organization's sector expertise",
        "monitoring_evaluation": "string - M&E plan description",
        "financial_sustainability": "string - financial sustainability plan",
        "institutional_sustainability": "string - institutional sustainability plan",
        "policy_impact": "string - potential policy level impact",
        "project_objectives": ["string - list of specific objectives"]
    }}

    Return ONLY the JSON object, no additional text.
    """

    try:
        # Call Gemini
        response = ask_gemini(prompt)
        print(f"  Gemini raw response: {response[:500]}...")

        # Clean the response to extract just the JSON
        response_text = response.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

        # Parse JSON
        extracted_data = json.loads(response_text)
        print(f"  ✅ Successfully parsed JSON response")

        # Ensure all expected fields are present
        expected_fields = [
            "organisation_name", "company_id", "project_title", "grant_application_id",
            "past_partner_status", "collaboration_status", "funding_requested",
            "project_duration", "project_location", "location_details", "team_size",
            "volunteer_count", "project_goal", "project_summary", "project_description",
            "impact_area", "impact_alignment", "sdg_alignment", "beneficiaries",
            "beneficiary_involvement", "beneficiary_benefits", "problem_analysis",
            "solution_approach", "organizational_expertise", "monitoring_evaluation",
            "financial_sustainability", "institutional_sustainability", "policy_impact",
            "project_objectives"
        ]

        # Initialize missing fields with null
        for field in expected_fields:
            if field not in extracted_data:
                if field == "project_objectives":
                    extracted_data[field] = []
                else:
                    extracted_data[field] = None

        return extracted_data

    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parsing error: {e}")
        print(f"  Raw response that failed: {response}")
        # Return empty structure on error
        return create_empty_proposal_structure()

    except Exception as e:
        print(f"  ❌ Error in parse_proposal_cloud: {e}")
        # Return empty structure on any error
        return create_empty_proposal_structure()


def parse_proposal_cloud_enhanced(proposal_text: str) -> Dict[str, Any]:
    """
    Enhanced parser that returns both structured data AND Q&A pairs in one call.
    """
    # Pre-extract funding as fallback
    funding_fallback = extract_funding_fallback(proposal_text)

    prompt = f"""
    EXTRACT STRUCTURED DATA AND GENERATE Q&A PAIRS from this grant proposal.

    PROPOSAL TEXT:
    {proposal_text[:12000]}

    **SPECIAL EXTRACTION NOTES:**
    - For 'funding_requested': Look for phrases like "Requested funding", "Requested Funding (MYR)", "MYR" followed by numbers
    - Extract the numeric amount only (remove commas, keep just digits)
    - This is typically found in sections labeled "Requested Funding" or "Funding Requested"

    TASK 1: Extract structured data into the structured_data field
    TASK 2: Generate 10-15 diverse Q&A pairs for the qa_pairs field

    Return ONLY this JSON format - no other text:
    {{
        "structured_data": {{
            "organisation_name": "string",
            "project_title": "string", 
            "funding_requested": "number - **LOOK FOR 'Requested funding' PHRASE**",
            "project_duration": "number",
            "project_location": "string",
            "impact_area": "string",
            "beneficiaries": "string",
            "project_goal": "string",
            "project_summary": "string"
        }},
        "qa_pairs": [
            {{
                "question": "natural language question",
                "answer": "concise answer from text", 
                "tag": "category_tag"
            }}
        ]
    }}

    Generate Q&A pairs for these categories: organization, funding, project_details, impact, beneficiaries, sustainability.

    IMPORTANT: Include at least one Q&A pair about the funding amount.

    Return ONLY valid JSON.
    """

    try:
        response = ask_gemini(prompt)
        print(f"  Enhanced parser raw response: {response[:500]}...")

        # Clean response
        response_text = response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        enhanced_data = json.loads(response_text.strip())
        print(f"  ✅ Enhanced parser successful - {len(enhanced_data.get('qa_pairs', []))} Q&A pairs")

        # ✅ ADD FALLBACK EXTRACTION HERE
        # If Gemini missed funding, try regex fallback
        if enhanced_data.get("structured_data", {}).get("funding_requested") is None:
            funding_fallback = extract_funding_fallback(proposal_text)
            if funding_fallback:
                enhanced_data["structured_data"]["funding_requested"] = funding_fallback
                print(f"  🔧 Applied funding fallback: MYR {funding_fallback:,}")

        # ✅ THEN DO POST-PROCESSING
        if enhanced_data.get("structured_data", {}).get("funding_requested"):
            funding_amount = enhanced_data["structured_data"]["funding_requested"]

            # Remove any existing funding Q&A
            enhanced_data["qa_pairs"] = [
                qa for qa in enhanced_data.get("qa_pairs", [])
                if "funding" not in qa.get("tag", "").lower()
            ]

            # Add correct funding Q&A
            enhanced_data["qa_pairs"].append({
                "question": "What is the total funding requested for this project?",
                "answer": f"MYR {funding_amount:,}",
                "tag": "funding"
            })
            print(f"  🔧 Replaced funding Q&A: MYR {funding_amount:,}")

        return enhanced_data

    except json.JSONDecodeError as e:
        print(f"  ❌ Enhanced parser JSON error: {e}")
        print(f"  Raw response: {response}")
        # Fallback to basic structured data only
        return {
            "structured_data": parse_proposal_cloud(proposal_text),
            "qa_pairs": []
        }
    except Exception as e:
        print(f"  ❌ Enhanced parsing failed: {e}")
        return {
            "structured_data": parse_proposal_cloud(proposal_text),
            "qa_pairs": []
        }


def extract_funding_fallback(text: str) -> Optional[int]:
    """Fallback funding extraction using regex."""
    patterns = [
        r"Requested funding.*?MYR\s*([\d,]+)",
        r"Requested Funding.*?MYR\s*([\d,]+)",
        r"MYR\s*([\d,]+)\s*\(?Programme Management Fund\)?"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount = match.group(1).replace(',', '')
            return int(amount)
    return None

def create_empty_proposal_structure() -> Dict[str, Any]:
    """Return empty proposal structure for error handling."""
    return {
        "organisation_name": None,
        "company_id": None,
        "project_title": None,
        "grant_application_id": None,
        "past_partner_status": None,
        "collaboration_status": None,
        "funding_requested": None,
        "project_duration": None,
        "project_location": None,
        "location_details": None,
        "team_size": None,
        "volunteer_count": None,
        "project_goal": None,
        "project_summary": None,
        "project_description": None,
        "impact_area": None,
        "impact_alignment": None,
        "sdg_alignment": None,
        "beneficiaries": None,
        "beneficiary_involvement": None,
        "beneficiary_benefits": None,
        "problem_analysis": None,
        "solution_approach": None,
        "organizational_expertise": None,
        "monitoring_evaluation": None,
        "financial_sustainability": None,
        "institutional_sustainability": None,
        "policy_impact": None,
        "project_objectives": []
    }