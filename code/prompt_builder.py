from typing import Any


def build_extraction_prompt(
    chat_transcript: str,
    claim_object: str,
) -> tuple[str, str]:
    """Stage 1: Extract structured claims from user text. No vision, no analysis."""
    system_prompt = (
        "You are a precise claim extractor. "
        "Extract the customer's damage claim from their conversation transcript. "
        "Output ONLY valid JSON with these exact keys:\n"
        "- claim_object_part: the specific part of the object claimed to be damaged\n"
        "- claim_issue_type: the type of damage claimed\n"
        "- claim_description: a brief, neutral summary of the claim\n\n"
        "Do NOT analyze, verify, or comment on the claim. Just extract."
    )
    user_content = (
        f"## Claim Object\n{claim_object}\n\n"
        f"## Claim Conversation\n{chat_transcript}\n\n"
        "Extract the claim and output JSON."
    )
    return system_prompt, user_content


def build_blind_vision_prompt(
    claim_object: str,
    allowed_parts: set[str],
    allowed_issues: list[str],
) -> str:
    """Stage 2: Analyze images WITHOUT any claim context.
    
    The user's claim is deliberately NOT included in this prompt.
    The model acts as a strict classifier, not a generative assistant.
    """
    parts_str = ", ".join(sorted(allowed_parts))
    issues_str = ", ".join(allowed_issues)

    system_prompt = (
        "You are a strict damage classifier. Examine the image(s) carefully. "
        "Output ONLY valid JSON with these exact keys:\n"
        f"- issue_type: MUST be exactly one of [{issues_str}]. "
        "Pick the single best match. Use 'unknown' if unsure.\n"
        f"- object_part: MUST be exactly one of [{parts_str}]. "
        "Pick a single part name from this list. Do NOT combine multiple parts. "
        "Use 'unknown' if unsure.\n"
        "- severity: none | low | medium | high | unknown\n"
        "- valid_image: true if the image is clear, well-lit, and shows "
        "the claimed object; false if blurry, obstructed, or irrelevant\n"
        "- damage_description: brief objective description of visible damage. "
        "If none visible, say 'No visible damage.'\n"
        "- supporting_image_ids: semicolon-separated list of image filenames "
        "that show damage; 'none' if no damage visible\n\n"
        "Rules:\n"
        "1. Be conservative. If you are not sure, use 'unknown' or 'none'.\n"
        "2. Do NOT invent damage that is not clearly visible.\n"
        "3. Describe only what you see — not what you think might have happened.\n"
        "4. object_part must be a SINGLE value from the list, no extra words."
    )
    return system_prompt


def build_synthesis_prompt(
    extracted_claim: dict[str, Any],
    vision_report: dict[str, Any],
    history_json: str,
    reqs_json: str,
) -> tuple[str, str]:
    """Stage 3: Compare extracted claim against objective vision evidence."""
    system_prompt = (
        "You are a precise claim verification analyst. "
        "Compare the extracted claim against the objective vision report "
        "and produce a structured evaluation.\n\n"
        "Output ONLY valid JSON with these keys:\n"
        "evidence_standard_met, evidence_standard_met_reason, risk_flags, "
        "issue_type, object_part, claim_status, claim_status_justification, "
        "supporting_image_ids, valid_image, severity\n\n"
        "Allowed values:\n"
        "- claim_status: supported | contradicted | not_enough_information\n"
        "- issue_type: dent | scratch | crack | glass_shatter | broken_part | "
        "missing_part | torn_packaging | crushed_packaging | water_damage | "
        "stain | none | unknown\n"
        "- risk_flags: semicolon-separated from: none | blurry_image | "
        "cropped_or_obstructed | low_light_or_glare | wrong_angle | wrong_object | "
        "wrong_object_part | damage_not_visible | claim_mismatch | "
        "possible_manipulation | non_original_image | text_instruction_present | "
        "user_history_risk | manual_review_required\n"
        "- severity: none | low | medium | high | unknown\n"
        "- evidence_standard_met: true | false (boolean)\n"
        "- valid_image: true | false (boolean)\n"
        "- supporting_image_ids: semicolon-separated basenames or 'none'\n\n"
        "CRITICAL - claim_status MUST be exactly one of: supported | contradicted | "
        "not_enough_information. Do NOT use any other values.\n"
        "Decision logic:\n"
        "- supported: vision report confirms the claimed damage type and part\n"
        "- contradicted: vision report shows no damage or different damage type "
        "on the claimed part\n"
        "- not_enough_information: images are unclear, unrelated, or insufficient. "
        "Also use this when the vision report's issue_type is 'unknown' or 'none' "
        "but the images are valid (evidence simply doesn't show the claim)"
    )
    user_content = (
        f"## Extracted Claim\n{extracted_claim}\n\n"
        f"## Objective Vision Report\n{vision_report}\n\n"
        f"## User History\n{history_json}\n\n"
        f"## Evidence Requirements\n{reqs_json}\n\n"
        "Compare and produce the evaluation JSON."
    )
    return system_prompt, user_content
