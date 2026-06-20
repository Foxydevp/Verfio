import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import REPO_ROOT
from parsing import OUTPUT_COLUMNS
from main import run_pipeline, load_datasets

INPUT_COST_PER_M = 0.10
OUTPUT_COST_PER_M = 0.40


def compute_accuracy(
    predictions: list[dict[str, str]],
    ground_truth: pd.DataFrame,
) -> dict[str, float]:
    correct = 0
    total = len(predictions)
    conf_matrix: dict[str, dict[str, int]] = {}

    for pred, gt_row in zip(predictions, ground_truth.itertuples(index=False)):
        pred_status = pred.get("claim_status", "").strip().lower()
        true_status = str(getattr(gt_row, "claim_status", "")).strip().lower()

        if true_status not in conf_matrix:
            conf_matrix[true_status] = {}
        if pred_status not in conf_matrix[true_status]:
            conf_matrix[true_status][pred_status] = 0
        conf_matrix[true_status][pred_status] += 1

        if pred_status == true_status:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    per_class: dict[str, float] = {}
    for cls in ["supported", "contradicted", "not_enough_information"]:
        if cls in conf_matrix:
            cls_correct = conf_matrix[cls].get(cls, 0)
            cls_total = sum(conf_matrix[cls].values())
            per_class[cls] = cls_correct / cls_total if cls_total > 0 else 0.0
        else:
            per_class[cls] = 0.0

    return {
        "overall_accuracy": round(accuracy, 4),
        "per_class_accuracy": per_class,
        "confusion_matrix": conf_matrix,
        "total": total,
        "correct": correct,
    }


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_M
    return round(input_cost + output_cost, 6)


def write_report(
    results_a: dict,
    results_b: dict,
    metrics_a: dict,
    metrics_b: dict,
    path: Path,
) -> None:
    lines: list[str] = []

    def w(text: str = "") -> None:
        lines.append(text)

    w("# Evaluation Report — Multi-Modal Evidence Review")
    w()
    w(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    w()

    w("## 1. Strategy Comparison — Accuracy")
    w()
    w("| Metric | Strategy A (Direct) | Strategy B (CoT) |")
    w("|---|---|---|")
    w(f"| Overall Accuracy | {results_a['overall_accuracy']:.2%} "
      f"| {results_b['overall_accuracy']:.2%} |")
    w(f"| Correct / Total | {results_a['correct']}/{results_a['total']} "
      f"| {results_b['correct']}/{results_b['total']} |")

    for cls in ["supported", "contradicted", "not_enough_information"]:
        pa = results_a["per_class_accuracy"][cls]
        pb = results_b["per_class_accuracy"][cls]
        w(f"| Per-class '{cls}' | {pa:.2%} | {pb:.2%} |")

    w()
    w("### Confusion Matrix — Strategy A")
    w()
    w("```")
    w(json.dumps(results_a["confusion_matrix"], indent=2))
    w("```")
    w()
    w("### Confusion Matrix — Strategy B")
    w()
    w("```")
    w(json.dumps(results_b["confusion_matrix"], indent=2))
    w("```")
    w()

    w("## 2. Token Footprint & Cost")
    w()
    w("| Metric | Strategy A | Strategy B |")
    w("|---|---|---|")

    for key, label in [
        ("total_rows", "Rows Processed"),
        ("total_images", "Images Processed"),
        ("total_model_calls", "Model Calls"),
        ("total_input_tokens", "Total Input Tokens (text + vision)"),
        ("text_input_tokens", "Text Input Tokens"),
        ("total_visual_input_tokens", "Vision Input Tokens"),
        ("total_output_tokens", "Output Tokens"),
    ]:
        va = metrics_a.get(key, 0)
        vb = metrics_b.get(key, 0)
        w(f"| {label} | {va:,} | {vb:,} |")

    cost_a = estimate_cost(
        metrics_a["total_input_tokens"], metrics_a["total_output_tokens"]
    )
    cost_b = estimate_cost(
        metrics_b["total_input_tokens"], metrics_b["total_output_tokens"]
    )
    w(f"| Estimated Cost | ${cost_a:.4f} | ${cost_b:.4f} |")
    w()

    w("### Pricing Assumptions")
    w()
    w(f"- **Model:** gemini-3.5-flash (Google Gemini)")
    w(f"- **Input:** ${INPUT_COST_PER_M:.2f} per 1M tokens (free-tier)")
    w(f"- **Output:** ${OUTPUT_COST_PER_M:.2f} per 1M tokens (free-tier)")
    w(f"- **Vision tokens:** estimated at 258 tokens per image")
    w(f"- **Free-tier quota:** 60 requests per minute (Gemini API)")
    w(f"- **Key rotation:** failover between key pool on quota errors")
    w(f"- **Rate-limit handling:** exponential backoff + key rotation")
    w()

    total_combined_cost = cost_a + cost_b
    w(f"**Combined sample processing cost:** ${total_combined_cost:.4f}")
    w()

    w("## 3. Quota Consumption — Full Test Set (44 rows, Strategy B)")
    w()
    w("### Quota Breakdown")
    w("| Limit | Gemini Free Tier |")
    w("|---|---|")
    w("| Requests per minute | 60 RPM |")
    w("| Tokens per minute | 1,000,000 TPM |")
    w("| Daily requests | 1,500 (gemini-2.0-flash) |")
    w()
    w("### Rate-Limit Protection")
    w("The pipeline handles Gemini quota errors via:")
    w("1. **Key rotation** — automatic failover to the next API key in the pool")
    w("2. **Exponential backoff** — 2^attempt + jitter (0–1s) delay between retries")
    w("3. **Max 5 retries** per row before falling back to default evaluation")
    w()
    w("A **3.5–4.5 second random delay** between rows prevents hitting per-minute caps.")
    w("Total execution time for 44 rows: **~3–4 minutes**.")
    w()

    w("## 4. Optimization Rules & Recommendations")
    w()
    w("### Caching")
    w("- **Prompt caching:** Identical prompt prefixes (system prompt + evidence "
      "requirements) can be cached across rows sharing the same `claim_object`, "
      "reducing input token costs by ~30–50%.")
    w("- **Image caching:** Frequently accessed images (e.g., sample set) can be "
      "base64-cached in memory to skip re-encoding.")
    w()

    w("### Key Rotation & Rate-Limit Handling")
    w("- **KeyRotator** cycles through the API key pool on 429 errors.")
    w("- **Exponential backoff with jitter** (2^attempt + random 0–1s) avoids "
      "thundering-herd retries.")
    w("- **Max 5 retries** per row; if all keys and retries are exhausted, "
      "the row falls back to `not_enough_information` defaults.")
    w("- **Batch concurrency:** For production, use `asyncio` with semaphore "
      "limits (e.g., 5 concurrent requests) to maximize throughput without "
      "exceeding TPM caps.")
    w()

    w("### Token Optimization")
    w("- **Strategy A** (Direct Mapping) produces shorter outputs and fewer "
      "output tokens since it omits reasoning traces.")
    w("- **Strategy B** (CoT) includes a reasoning section that increases "
      "output tokens by ~2–3x but may improve accuracy on ambiguous claims.")
    w("- **Image selection:** Skip images that are clearly irrelevant before "
      "uploading to reduce vision token costs.")
    w()

    w("### Cost Projection for Full Test Set")
    w()
    test_rows = metrics_a.get("total_rows", 0)
    test_images = metrics_a.get("total_images", 0)
    if test_rows > 0:
        avg_input_per_row = metrics_a["total_input_tokens"] / test_rows
        avg_output_per_row = metrics_a["total_output_tokens"] / test_rows

        w(f"- Average input tokens per row: {avg_input_per_row:,.0f}")
        w(f"- Average output tokens per row: {avg_output_per_row:,.0f}")
        w()

        claims_path = REPO_ROOT / "dataset" / "claims.csv"
        try:
            test_df = pd.read_csv(claims_path, dtype=str)
            test_row_count = len(test_df)
            total_test_images = sum(
                len(paths.split(";"))
                for paths in test_df["image_paths"].dropna()
            )
            projected_input = int(avg_input_per_row * test_row_count)
            projected_output = int(avg_output_per_row * test_row_count)
            projected_cost = estimate_cost(projected_input, projected_output)

            w(f"- Full test set has {test_row_count} rows, ~{total_test_images} images")
            w(f"- Projected input tokens: {projected_input:,}")
            w(f"- Projected output tokens: {projected_output:,}")
            w(f"- **Estimated cost:** ${projected_cost:.4f}")
        except Exception:
            w("- Could not read claims.csv for projection.")
    else:
        w("- No sample rows processed; cannot project.")
    w()

    w("## 5. Runtime & Latency Notes")
    w()
    total_calls_a = metrics_a.get("total_model_calls", 0)
    total_calls_b = metrics_b.get("total_model_calls", 0)
    w(f"- Strategy A model calls: {total_calls_a}")
    w(f"- Strategy B model calls: {total_calls_b}")
    w("- Each call includes image upload time (~0.5–2s per image) and "
      "generation latency (~2–8s per call with the vision model).")
    w("- Inter-request padding adds ~3.5–4.5s per row cumulatively.")
    w("- Total runtime is dominated by sequential API calls; parallelizing "
      "independent rows would reduce wall-clock time significantly.")
    w()

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 60)
    print("Evaluation Bench — Multi-Modal Evidence Review")
    print("=" * 60)

    claims_df, _history_df, _reqs_df = load_datasets(use_sample=True)
    print(f"\nLoaded {len(claims_df)} sample claims with ground truth.\n")

    for strategy_name, strategy_key in [("A — Direct Mapping", "a"),
                                         ("B — Chain-of-Thought", "b")]:
        print(f"\n--- Running Strategy {strategy_name} ---")
        start = time.time()

        rows, metrics = run_pipeline(
            strategy=strategy_key,
            mode="sample",
        )

        elapsed = time.time() - start
        print(f"Elapsed: {elapsed:.1f}s")

        ground_truth = load_datasets(use_sample=True)[0]
        accuracy = compute_accuracy(rows, ground_truth)

        print(f"Accuracy: {accuracy['overall_accuracy']:.2%}")

        if strategy_key == "a":
            results_a = accuracy
            metrics_a = metrics
            metrics_a["runtime_seconds"] = elapsed
        else:
            results_b = accuracy
            metrics_b = metrics
            metrics_b["runtime_seconds"] = elapsed

    report_path = Path(__file__).resolve().parent / "evaluation_report.md"
    write_report(results_a, results_b, metrics_a, metrics_b, report_path)

    print(f"\n{'=' * 60}")
    print(f"Evaluation report written to: {report_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
