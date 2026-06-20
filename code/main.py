import json
import random
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config import (
    DATASET_DIR,
    INTER_REQUEST_DELAY,
    PROVIDER_MODELS,
    PROVIDER_ORDER,
    REPO_ROOT,
    get_google_api_key,
    get_openrouter_keys,
)
from llm_client import (
    ProviderRouter,
    mock_fallback_response,
)
from parsing import (
    ALL_OBJECT_PARTS,
    OUTPUT_COLUMNS,
    ClaimExtraction,
    ClaimEvaluation,
    VisionReport,
    fallback_evaluation,
    parse_response_to_evaluation,
)
from prompt_builder import (
    build_extraction_prompt,
    build_blind_vision_prompt,
    build_synthesis_prompt,
)


def load_datasets(
    use_sample: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    claims_file = "sample_claims.csv" if use_sample else "claims.csv"
    claims_path = DATASET_DIR / claims_file
    history_path = DATASET_DIR / "user_history.csv"
    reqs_path = DATASET_DIR / "evidence_requirements.csv"
    claims = pd.read_csv(claims_path, dtype=str)
    history = pd.read_csv(history_path, dtype=str)
    reqs = pd.read_csv(reqs_path, dtype=str)
    return claims, history, reqs


def resolve_image_paths(image_paths_str: str) -> list[Path]:
    parts = [p.strip() for p in image_paths_str.split(";")]
    resolved: list[Path] = []
    for rel_path in parts:
        if not rel_path:
            continue
        candidate = DATASET_DIR / rel_path
        if candidate.exists() and candidate.is_file():
            resolved.append(candidate.resolve())
    return resolved


def get_row_context(
    row: pd.Series,
    history_df: pd.DataFrame,
    reqs_df: pd.DataFrame,
) -> dict[str, str]:
    user_id = row.get("user_id", "")
    claim_object = row.get("claim_object", "")
    user_matches = history_df[history_df["user_id"] == user_id]
    user_history_json = (
        json.dumps(user_matches.iloc[0].to_dict(), indent=2)
        if not user_matches.empty
        else json.dumps({})
    )
    obj_mask = (reqs_df["claim_object"] == claim_object) | (
        reqs_df["claim_object"] == "all"
    )
    filtered_reqs = reqs_df[obj_mask]
    reqs_json = json.dumps(filtered_reqs.to_dict(orient="records"), indent=2)
    return {"user_history": user_history_json, "evidence_requirements": reqs_json}


def run_pipeline(
    strategy: str = "a",
    mode: str = "test",
    model_override: Optional[str] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    use_sample = mode == "sample"
    claims_df, history_df, reqs_df = load_datasets(use_sample=use_sample)

    # Initialize provider router
    gemini_key = get_google_api_key()
    or_keys = get_openrouter_keys()

    if not gemini_key and not or_keys:
        print("WARNING: No API keys found. Using mock fallback.")
        router = ProviderRouter("", [])
    else:
        router = ProviderRouter(gemini_key, or_keys, PROVIDER_ORDER)
        print(f"ProviderRouter: {router.active_providers} "
              f"(Gemini key: {'yes' if gemini_key else 'no'}, "
              f"OpenRouter keys: {len(or_keys)})")

    models = dict(PROVIDER_MODELS)
    if model_override:
        for prov in models:
            for stage in models[prov]:
                models[prov][stage] = model_override

    use_mock = not gemini_key and not or_keys

    output_rows: list[dict[str, Any]] = []
    metrics = {
        "total_rows": 0,
        "total_images": 0,
        "total_model_calls": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_visual_input_tokens": 0,
        "text_input_tokens": 0,
        "stage_breakdown": {"extraction": 0, "blind_vision": 0, "synthesis": 0},
        "provider_usage": {},  # provider -> call count
    }

    for idx, row in claims_df.iterrows():
        context = get_row_context(row, history_df, reqs_df)
        image_paths = resolve_image_paths(row.get("image_paths", ""))
        claim_object = row.get("claim_object", "")
        chat_transcript = row.get("user_claim", "")

        print(
            f"\n[{idx}] user={row.get('user_id')} "
            f"object={claim_object} "
            f"images={len(image_paths)}"
        )

        try:
            if use_mock:
                resp = mock_fallback_response()
                evaluation = parse_response_to_evaluation(resp.text, claim_object)
                metrics["total_model_calls"] += 3
            else:
                # ── Stage 1: Extraction ─────────────────────────────────
                sys_prompt, user_text = build_extraction_prompt(
                    chat_transcript, claim_object
                )
                print(f"  Stage 1 (extraction)")
                ext_resp = router.call_text(
                    models, "extraction", sys_prompt, user_text,
                    schema=ClaimExtraction,
                )
                metrics["total_model_calls"] += 1
                metrics["stage_breakdown"]["extraction"] += 1
                _accumulate_usage(metrics, ext_resp, len(image_paths))
                _count_provider(metrics, ext_resp)

                ext_json = _safe_parse_json(ext_resp.text)
                print(
                    f"  Extracted: {ext_json.get('claim_issue_type', '?')} "
                    f"on {ext_json.get('claim_object_part', '?')}"
                )

                # ── Stage 2: Blind Vision ──────────────────────────────
                allowed_parts = ALL_OBJECT_PARTS.get(claim_object, {"unknown"})
                vision_sys = build_blind_vision_prompt(
                    claim_object,
                    allowed_parts,
                    [
                        "dent", "scratch", "crack", "glass_shatter",
                        "broken_part", "missing_part", "torn_packaging",
                        "crushed_packaging", "water_damage", "stain",
                        "none", "unknown",
                    ],
                )
                vision_text = "Examine the image(s) and output JSON."

                print(f"  Stage 2 (blind vision)")
                vision_resp = router.call_vision(
                    models, "blind_vision", vision_sys, vision_text, image_paths,
                    schema=VisionReport,
                )
                metrics["total_model_calls"] += 1
                metrics["stage_breakdown"]["blind_vision"] += 1
                _accumulate_usage(metrics, vision_resp, len(image_paths))
                _count_provider(metrics, vision_resp)

                vision_json = _safe_parse_json(vision_resp.text)
                print(
                    f"  Vision: issue={vision_json.get('issue_type', '?')} "
                    f"part={vision_json.get('object_part', '?')}"
                )

                # ── Stage 3: Synthesis ─────────────────────────────────
                sys_prompt3, user_text3 = build_synthesis_prompt(
                    ext_json,
                    vision_json,
                    context["user_history"],
                    context["evidence_requirements"],
                )
                print(f"  Stage 3 (synthesis)")
                syn_resp = router.call_text(
                    models, "synthesis", sys_prompt3, user_text3,
                    schema=ClaimEvaluation,
                )
                metrics["total_model_calls"] += 1
                metrics["stage_breakdown"]["synthesis"] += 1
                _accumulate_usage(metrics, syn_resp, len(image_paths))
                _count_provider(metrics, syn_resp)

                evaluation = parse_response_to_evaluation(syn_resp.text, claim_object)

            print(
                f"  -> claim_status={evaluation['claim_status']} "
                f"issue_type={evaluation['issue_type']}"
            )

        except Exception as exc:
            print(f"  ERROR on row {idx}: {exc}")
            evaluation = fallback_evaluation()

        output_row = {
            "user_id": row.get("user_id", ""),
            "image_paths": row.get("image_paths", ""),
            "user_claim": chat_transcript,
            "claim_object": claim_object,
            **evaluation,
        }
        output_rows.append(output_row)
        metrics["total_rows"] += 1
        metrics["total_images"] += len(image_paths)

        delay = random.uniform(*INTER_REQUEST_DELAY)
        time.sleep(delay)

    metrics["text_input_tokens"] = max(
        0, metrics["total_input_tokens"] - metrics["total_visual_input_tokens"]
    )
    return output_rows, metrics


def _accumulate_usage(
    metrics: dict, response: Any, num_images: int
) -> None:
    metrics["total_input_tokens"] += response.prompt_tokens
    metrics["total_output_tokens"] += response.completion_tokens
    metrics["total_visual_input_tokens"] += 0  # Gemini reports actual vision tokens


def _count_provider(metrics: dict, response: Any) -> None:
    prov = response.provider
    metrics.setdefault("provider_usage", {})
    metrics["provider_usage"][prov] = metrics["provider_usage"].get(prov, 0) + 1


def _safe_parse_json(text: str) -> dict[str, Any]:
    from parsing import _extract_json
    return _extract_json(text) or {}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="3-Stage Blind Vision Pipeline")
    parser.add_argument(
        "--mode", type=str, default="test", choices=["test", "sample"]
    )
    parser.add_argument("--strategy", type=str, default="a", choices=["a", "b"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--model", type=str, default=None, help="Override model for all stages"
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        help="Force specific provider (gemini, openrouter)"
    )
    args = parser.parse_args()

    # Provider override
    if args.provider:
        import config as cfg
        cfg.PROVIDER_ORDER = [args.provider]

    rows, metrics = run_pipeline(
        strategy=args.strategy,
        mode=args.mode,
        model_override=args.model,
    )

    output_path = Path(args.output) if args.output else REPO_ROOT / "output.csv"
    output_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_df.to_csv(output_path, index=False, quoting=1)

    print(f"\n=== Pipeline Complete ===")
    print(f"Rows:        {metrics['total_rows']}")
    print(f"Images:      {metrics['total_images']}")
    print(f"Model calls: {metrics['total_model_calls']}")
    print(f"  extraction: {metrics['stage_breakdown']['extraction']}")
    print(f"  blind_vision: {metrics['stage_breakdown']['blind_vision']}")
    print(f"  synthesis: {metrics['stage_breakdown']['synthesis']}")
    print(f"Provider usage: {metrics.get('provider_usage', {})}")
    print(f"Input tok:   {metrics['total_input_tokens']}")
    print(f"Output tok:  {metrics['total_output_tokens']}")
    print(f"Output CSV:  {output_path}")


if __name__ == "__main__":
    main()
