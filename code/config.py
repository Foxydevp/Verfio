import os
from pathlib import Path

# ── .env loader ──────────────────────────────────────────────────────────
def _load_dotenv() -> None:
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    if dotenv_path.exists():
        with open(dotenv_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        k = parts[0].strip()
                        v = parts[1].strip()
                        if not os.environ.get(k):
                            os.environ[k] = v

_load_dotenv()

# ── Provider cascade ─────────────────────────────────────────────────────
# Try Gemini native SDK first; on quota exhaustion, fall back to OpenRouter.
PROVIDER_ORDER = ["gemini", "openrouter"]

# ── Model names per provider per stage ────────────────────────────────────
PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "gemini": {
        "extraction":   "gemini-2.5-flash",
        "blind_vision": "gemini-2.5-flash-lite",  # cheaper for image-heavy step
        "synthesis":    "gemini-2.5-flash",
    },
    "openrouter": {
        "extraction":   "google/gemini-2.5-flash",
        "blind_vision": "google/gemini-2.5-flash-lite",
        "synthesis":    "google/gemini-2.5-flash",
    },
}

# ── OpenRouter ────────────────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_REFERER = "https://github.com/anomalyco/opencode"
OPENROUTER_TITLE = "Multi-Modal Evidence Review"

# ── Image optimization ────────────────────────────────────────────────────
MAX_IMAGE_DIM = 512  # Pillow resize target (maintains aspect ratio)

# ── Timing ────────────────────────────────────────────────────────────────
INTER_REQUEST_DELAY = (3.5, 4.5)  # random uniform range in seconds
API_TIMEOUT = 30.0
TENACITY_MAX_ATTEMPTS = 3

# ── Paths ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"

# ── API key getters ───────────────────────────────────────────────────────
def get_google_api_key() -> str:
    return os.environ.get("GOOGLE_API_KEY", "").strip()

def get_openrouter_keys() -> list[str]:
    keys_str = os.environ.get("OPENROUTER_KEYS", "")
    if keys_str.strip():
        return [k.strip() for k in keys_str.split(",") if k.strip()]
    single = os.environ.get("OPENROUTER_API_KEY", "")
    if single.strip():
        return [single.strip()]
    return []
