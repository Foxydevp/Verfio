# Verfio — Multi-Modal Evidence Review Pipeline

A structured claim verification system that analyzes images, chat transcripts, user history, and evidence requirements to determine whether submitted images **support**, **contradict**, or **do not provide enough information** for a damage claim.

Evaluates claims across three object types: **cars**, **laptops**, and **packages** (with multi-language transcript support including English, Hindi, Spanish, and Chinese code-mixed text).

---

## Architecture

```
User Claim (chat)  ──►  Stage 1: Extraction  ──►  Structured Claim
Submitted Images   ──►  Stage 2: Blind Vision ──►  Objective Vision Report
User History       ──►                                     │
Evidence Reqs      ──►                                     ▼
                    ┌──── Stage 3: Synthesis ◄──────────────┘
                    ▼
           Structured Evaluation (CSV output)
```

### Stage 1 — Claim Extraction
Parses the chat transcript to extract the claimed damage type, object part, and a neutral summary. Pure text extraction — no image analysis.

### Stage 2 — Blind Vision
Evaluates submitted images **without any claim context**. This bias-reduced design prevents the model from rationalizing what the user claims. Identifies visible issue type, object part, severity, image validity, and supporting image IDs.

### Stage 3 — Synthesis
Compares the extracted claim against the objective vision report, incorporating user history risk context and evidence requirements. Produces the final structured evaluation.

---

## Key Features

- **Bias-reduced design:** Vision analysis runs without knowledge of the claim, preventing confirmation bias
- **Multi-provider cascade:** Google Gemini native SDK → OpenRouter failover with automatic key rotation
- **Rate-limit resilience:** Exponential backoff with jitter, rotating API key pool, tenacity retry
- **Structured output validation:** Pydantic schemas with field-level enforcement of allowed values
- **Multi-language claims:** Handles English, Hindi, Spanish, Chinese code-mixed transcripts
- **Evaluation framework:** Dual-strategy comparison with accuracy, confusion matrix, cost, and token analysis

---

## Setup

```bash
git clone https://github.com/Foxydevp/Verfio.git
cd Verfio

# (Recommended) Virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate       # Windows

# Install dependencies
pip install pillow pandas pydantic openai tenacity google-genai
```

### Configuration

Create `.env` in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

At least one key is required. If both are provided, the pipeline tries Gemini first and falls back to OpenRouter on quota exhaustion.

---

## Usage

```bash
# Run on sample claims (with ground truth — evaluates accuracy)
python code/main.py --mode sample

# Run on full test set (produces output.csv)
python code/main.py --mode test

# Override model for all stages
python code/main.py --mode sample --model gemini-2.5-flash

# Force a specific provider
python code/main.py --mode sample --provider gemini

# Evaluation benchmark
python code/evaluation/main.py
```

---

## Project Structure

```
verfio/
├── code/
│   ├── main.py                 # Pipeline entry point
│   ├── config.py               # API keys, model config, paths
│   ├── llm_client.py           # Provider clients + router
│   ├── prompt_builder.py       # Stage-specific prompts
│   ├── parsing.py              # Pydantic schemas + JSON extraction
│   └── evaluation/
│       ├── main.py             # Evaluation benchmark
│       └── evaluation_report.md
├── dataset/
│   ├── claims.csv              # Test set (44 unlabeled claims)
│   ├── sample_claims.csv       # Sample set (20 labeled claims)
│   ├── user_history.csv        # User risk context
│   ├── evidence_requirements.csv
│   └── images/
│       ├── sample/             # Sample claim images
│       └── test/               # Test claim images
├── AGENTS.md
├── problem_statement.md
└── .env                        # API keys (gitignored)
```

---

## Results

| Metric | Strategy A (Direct) | Strategy B (CoT) |
|---|---|---|
| Overall Accuracy | 55.00% | 60.00% |
| Supported (per-class) | 66.67% | 66.67% |
| Contradicted (per-class) | 60.00% | 60.00% |
| Not Enough Info (per-class) | 0.00% | 33.33% |

*Evaluation on 20 sample claims. Full details in `code/evaluation/evaluation_report.md`.*

---

## Known Limitations

- **~55–60% accuracy** — model/prompt improvements needed for production use
- **6/44 test rows** fall back to default evaluation due to JSON parse failures
- **`supporting_image_ids`** always outputs `none` — vision-to-synthesis data flow needs completion
- **Vision token tracking** currently reports 0 (tracking bug)
- Strategy A and B prompts are not yet differentiated
- No unit tests

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ |
| LLM Providers | Google Gemini (native SDK), OpenRouter |
| Image Processing | Pillow (resize, compress) |
| Data | pandas |
| Validation | pydantic |
| Resilience | tenacity (retry), openai SDK |
