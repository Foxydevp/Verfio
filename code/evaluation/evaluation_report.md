# Evaluation Report — Multi-Modal Evidence Review

Generated: 2026-06-20 17:47:12 India Standard Time

## 1. Strategy Comparison — Accuracy

| Metric | Strategy A (Direct) | Strategy B (CoT) |
|---|---|---|
| Overall Accuracy | 55.00% | 60.00% |
| Correct / Total | 11/20 | 12/20 |
| Per-class 'supported' | 66.67% | 66.67% |
| Per-class 'contradicted' | 60.00% | 60.00% |
| Per-class 'not_enough_information' | 0.00% | 33.33% |

### Confusion Matrix — Strategy A

```
{
  "supported": {
    "supported": 8,
    "contradicted": 4
  },
  "not_enough_information": {
    "contradicted": 3
  },
  "contradicted": {
    "supported": 2,
    "contradicted": 3
  }
}
```

### Confusion Matrix — Strategy B

```
{
  "supported": {
    "supported": 8,
    "contradicted": 4
  },
  "not_enough_information": {
    "contradicted": 2,
    "not_enough_information": 1
  },
  "contradicted": {
    "supported": 2,
    "contradicted": 3
  }
}
```

## 2. Token Footprint & Cost

| Metric | Strategy A | Strategy B |
|---|---|---|
| Rows Processed | 20 | 20 |
| Images Processed | 29 | 29 |
| Model Calls | 60 | 60 |
| Total Input Tokens (text + vision) | 50,836 | 50,847 |
| Text Input Tokens | 50,836 | 50,847 |
| Vision Input Tokens | 0 | 0 |
| Output Tokens | 6,632 | 6,643 |
| Estimated Cost | $0.0077 | $0.0077 |

### Pricing Assumptions

- **Model:** gemini-3.5-flash (Google Gemini)
- **Input:** $0.10 per 1M tokens (free-tier)
- **Output:** $0.40 per 1M tokens (free-tier)
- **Vision tokens:** estimated at 258 tokens per image
- **Free-tier quota:** 60 requests per minute (Gemini API)
- **Key rotation:** failover between key pool on quota errors
- **Rate-limit handling:** exponential backoff + key rotation

**Combined sample processing cost:** $0.0155

## 3. Quota Consumption — Full Test Set (44 rows, Strategy B)

### Quota Breakdown
| Limit | Gemini Free Tier |
|---|---|
| Requests per minute | 60 RPM |
| Tokens per minute | 1,000,000 TPM |
| Daily requests | 1,500 (gemini-2.0-flash) |

### Rate-Limit Protection
The pipeline handles Gemini quota errors via:
1. **Key rotation** — automatic failover to the next API key in the pool
2. **Exponential backoff** — 2^attempt + jitter (0–1s) delay between retries
3. **Max 5 retries** per row before falling back to default evaluation

A **3.5–4.5 second random delay** between rows prevents hitting per-minute caps.
Total execution time for 44 rows: **~3–4 minutes**.

## 4. Optimization Rules & Recommendations

### Caching
- **Prompt caching:** Identical prompt prefixes (system prompt + evidence requirements) can be cached across rows sharing the same `claim_object`, reducing input token costs by ~30–50%.
- **Image caching:** Frequently accessed images (e.g., sample set) can be base64-cached in memory to skip re-encoding.

### Key Rotation & Rate-Limit Handling
- **KeyRotator** cycles through the API key pool on 429 errors.
- **Exponential backoff with jitter** (2^attempt + random 0–1s) avoids thundering-herd retries.
- **Max 5 retries** per row; if all keys and retries are exhausted, the row falls back to `not_enough_information` defaults.
- **Batch concurrency:** For production, use `asyncio` with semaphore limits (e.g., 5 concurrent requests) to maximize throughput without exceeding TPM caps.

### Token Optimization
- **Strategy A** (Direct Mapping) produces shorter outputs and fewer output tokens since it omits reasoning traces.
- **Strategy B** (CoT) includes a reasoning section that increases output tokens by ~2–3x but may improve accuracy on ambiguous claims.
- **Image selection:** Skip images that are clearly irrelevant before uploading to reduce vision token costs.

### Cost Projection for Full Test Set

- Average input tokens per row: 2,542
- Average output tokens per row: 332

- Full test set has 44 rows, ~82 images
- Projected input tokens: 111,839
- Projected output tokens: 14,590
- **Estimated cost:** $0.0170

## 5. Runtime & Latency Notes

- Strategy A model calls: 60
- Strategy B model calls: 60
- Each call includes image upload time (~0.5–2s per image) and generation latency (~2–8s per call with the vision model).
- Inter-request padding adds ~3.5–4.5s per row cumulatively.
- Total runtime is dominated by sequential API calls; parallelizing independent rows would reduce wall-clock time significantly.
