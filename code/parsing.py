import json
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, field_validator


# ── DamageReport: lightweight schema for JSON mode / response_format ────
class DamageReport(BaseModel):
    claim_status: str
    issue_type: str


# ── Full evaluation model with literal enforcement ──────────────────────
RISK_FLAG_LITERALS = Literal[
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
]

ISSUE_TYPE_LITERALS = Literal[
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain",
    "none", "unknown",
]

CLAIM_STATUS_LITERALS = Literal["supported", "contradicted", "not_enough_information"]

SEVERITY_LITERALS = Literal["none", "low", "medium", "high", "unknown"]

CAR_PARTS = {
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
    "body", "unknown",
}
LAPTOP_PARTS = {
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
    "port", "base", "body", "unknown",
}
PACKAGE_PARTS = {
    "box", "package_corner", "package_side", "seal", "label",
    "contents", "item", "unknown",
}

ALL_OBJECT_PARTS: dict[str, set[str]] = {
    "car": CAR_PARTS,
    "laptop": LAPTOP_PARTS,
    "package": PACKAGE_PARTS,
}


# ── Extraction schema (Stage 1) ──────────────────────────────────────────
class ClaimExtraction(BaseModel):
    claim_object_part: str
    claim_issue_type: str
    claim_description: str


# ── Vision report schema (Stage 2) — with Literal enums for Gemini ───────
class VisionReport(BaseModel):
    issue_type: ISSUE_TYPE_LITERALS = "unknown"
    object_part: str = "unknown"
    severity: SEVERITY_LITERALS = "unknown"
    valid_image: bool = False
    damage_description: str = "No visible damage."
    supporting_image_ids: str = "none"


class ClaimEvaluation(BaseModel):
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: bool
    severity: str

    @field_validator("risk_flags")
    @classmethod
    def validate_risk_flags(cls, v: str) -> str:
        valid = RISK_FLAG_LITERALS.__args__
        parts = [p.strip() for p in v.split(";")]
        for part in parts:
            if part not in valid:
                raise ValueError(
                    f"Invalid risk_flag '{part}'. "
                    f"Must be one of: {', '.join(valid)}"
                )
        return v

    @field_validator("issue_type")
    @classmethod
    def validate_issue_type(cls, v: str) -> str:
        valid = ISSUE_TYPE_LITERALS.__args__
        if v not in valid:
            raise ValueError(
                f"Invalid issue_type '{v}'. "
                f"Must be one of: {', '.join(valid)}"
            )
        return v

    @field_validator("claim_status")
    @classmethod
    def validate_claim_status(cls, v: str) -> str:
        valid = CLAIM_STATUS_LITERALS.__args__
        if v not in valid:
            raise ValueError(
                f"Invalid claim_status '{v}'. "
                f"Must be one of: {', '.join(valid)}"
            )
        return v

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        valid = SEVERITY_LITERALS.__args__
        if v not in valid:
            raise ValueError(
                f"Invalid severity '{v}'. "
                f"Must be one of: {', '.join(valid)}"
            )
        return v

    @field_validator("supporting_image_ids")
    @classmethod
    def validate_supporting_image_ids(cls, v: str) -> str:
        parts = [p.strip() for p in v.split(";")]
        for part in parts:
            if not re.fullmatch(r"[a-zA-Z0-9_]+", part) and part != "none":
                raise ValueError(
                    f"Invalid supporting_image_id '{part}'. "
                    f"Must be alphanumeric/underscore or 'none'."
                )
        if "none" in parts and len(parts) > 1:
            raise ValueError(
                "supporting_image_ids cannot mix 'none' with other values."
            )
        return v


OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity",
]


# ── JSON extraction with edge-case interceptors ─────────────────────────
def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """Extract JSON from model output, handling known edge cases.
    
    Edge cases handled:
    1. Code-fenced JSON: ```json {...} ``` or ```{...}```
    2. List-wrapped JSON: [{...}]  (LLM wraps in array)
    3. Plain braces: {...} with trailing commas or whitespace
    4. Nested or truncated objects
    """
    if not text or not text.strip():
        return None

    # Strip code fences
    match = re.search(r"```(?:json)?\s*(\[?\s*\{.*?\}\s*\]?)\s*```", text, re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0]
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Find outermost braces
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start:brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try with cleaned whitespace/trailing commas
            candidate = _clean_json_string(candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # Try list-wrapped: find first '[' and last ']'
    list_start = text.find("[{")
    list_end = text.rfind("}]")
    if list_start != -1 and list_end > list_start:
        candidate = text[list_start:list_end + 2]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0]
        except json.JSONDecodeError:
            pass

    return None


def _clean_json_string(s: str) -> str:
    """Remove trailing commas before closing braces/brackets."""
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s


# ── Field-level validation helpers ──────────────────────────────────────
def _safe_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return False


def _validate_field(value: Any, allowed: set[str], default: str) -> str:
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in allowed:
            return cleaned
    return default


def _validate_risk_flags(value: Any, allowed_flags: tuple[str, ...], default: str) -> str:
    if not isinstance(value, str):
        return default
    parts = [p.strip() for p in value.split(";")]
    validated = [p for p in parts if p in allowed_flags]
    if not validated:
        return default
    return ";".join(validated)


def _validate_supporting_ids(value: Any) -> str:
    if not isinstance(value, str):
        return "none"
    parts = [p.strip() for p in value.split(";")]
    valid_parts = [p for p in parts if re.fullmatch(r"[a-zA-Z0-9_]+", p)]
    if not valid_parts:
        return "none"
    return ";".join(valid_parts)


# ── Full response parser ────────────────────────────────────────────────
def fallback_evaluation() -> dict[str, Any]:
    return {
        "evidence_standard_met": False,
        "evidence_standard_met_reason": "Evaluation failed; default fallback applied.",
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": (
            "The system could not parse the model output. "
            "Manual review is required."
        ),
        "supporting_image_ids": "none",
        "valid_image": False,
        "severity": "unknown",
    }


def parse_response_to_evaluation(
    response_text: str,
    claim_object: str,
) -> dict[str, Any]:
    parsed = _extract_json(response_text)
    if parsed is None:
        return fallback_evaluation()

    risk_allowed = RISK_FLAG_LITERALS.__args__
    issue_allowed = set(ISSUE_TYPE_LITERALS.__args__)
    status_allowed = set(CLAIM_STATUS_LITERALS.__args__)
    severity_allowed = set(SEVERITY_LITERALS.__args__)
    object_parts_allowed = ALL_OBJECT_PARTS.get(claim_object, {"unknown"})

    return {
        "evidence_standard_met": _safe_bool(parsed.get("evidence_standard_met")),
        "evidence_standard_met_reason": (
            str(parsed.get("evidence_standard_met_reason", ""))
            or "No reason provided."
        ),
        "risk_flags": _validate_risk_flags(
            parsed.get("risk_flags"), risk_allowed, "manual_review_required"
        ),
        "issue_type": _validate_field(
            parsed.get("issue_type"), issue_allowed, "unknown"
        ),
        "object_part": _validate_field(
            parsed.get("object_part"), object_parts_allowed, "unknown"
        ),
        "claim_status": _validate_field(
            parsed.get("claim_status"), status_allowed, "not_enough_information"
        ),
        "claim_status_justification": (
            str(parsed.get("claim_status_justification", ""))
            or "No justification provided."
        ),
        "supporting_image_ids": _validate_supporting_ids(
            parsed.get("supporting_image_ids")
        ),
        "valid_image": _safe_bool(parsed.get("valid_image")),
        "severity": _validate_field(
            parsed.get("severity"), severity_allowed, "unknown"
        ),
    }
