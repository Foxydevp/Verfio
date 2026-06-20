import base64
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import openai
import tenacity
from openai import OpenAI
from PIL import Image

from config import (
    API_TIMEOUT,
    MAX_IMAGE_DIM,
    OPENROUTER_BASE_URL,
    OPENROUTER_REFERER,
    OPENROUTER_TITLE,
    TENACITY_MAX_ATTEMPTS,
)


# ── Normalized response ──────────────────────────────────────────────────
@dataclass
class ProviderResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    provider: str = "unknown"


# ── Pillow image optimizer ───────────────────────────────────────────────
def optimize_image(img_path: Path) -> tuple[bytes, str]:
    """Resize image to MAX_IMAGE_DIM, return (bytes, mime_type)."""
    mime = "image/jpeg"
    if img_path.suffix.lower() in (".png",):
        mime = "image/png"
    elif img_path.suffix.lower() in (".webp",):
        mime = "image/webp"

    with Image.open(img_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"


def encode_image_to_data_uri(img_path: Path) -> str:
    """Optimize image and return base64 data URI (for OpenRouter)."""
    data, mime = optimize_image(img_path)
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# ── Abstract provider ────────────────────────────────────────────────────
class BaseProvider(ABC):
    @abstractmethod
    def call_text(
        self, model: str, system_prompt: str, user_text: str,
        schema: Optional[type] = None,
    ) -> ProviderResponse: ...

    @abstractmethod
    def call_vision(
        self,
        model: str,
        system_prompt: str,
        text_prompt: str,
        image_paths: list[Path],
        schema: Optional[type] = None,
    ) -> ProviderResponse: ...


# ── Gemini (native SDK) ──────────────────────────────────────────────────
class GeminiProvider(BaseProvider):
    def __init__(self, api_key: str) -> None:
        from google import genai
        from google.genai.types import GenerateContentConfig, ThinkingConfig

        self._client = genai.Client(api_key=api_key)
        self._config_cls = GenerateContentConfig
        self._thinking = ThinkingConfig(include_thoughts=False)

    def _base_config(self, schema: Optional[type] = None) -> Any:
        kwargs: dict[str, Any] = {
            "temperature": 0.1,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
            "thinking_config": self._thinking,
        }
        if schema is not None:
            kwargs["response_schema"] = schema
        return self._config_cls(**kwargs)

    def call_text(
        self, model: str, system_prompt: str, user_text: str,
        schema: Optional[type] = None,
    ) -> ProviderResponse:
        resp = self._client.models.generate_content(
            model=model,
            contents=user_text,
            config=self._base_config(schema=schema),
        )
        usage = resp.usage_metadata
        return ProviderResponse(
            text=resp.text,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            provider="gemini",
        )

    def call_vision(
        self,
        model: str,
        system_prompt: str,
        text_prompt: str,
        image_paths: list[Path],
        schema: Optional[type] = None,
    ) -> ProviderResponse:
        from google.genai.types import Part

        parts = [Part.from_text(text=text_prompt)]
        for p in image_paths:
            data, mime = optimize_image(p)
            parts.append(Part.from_bytes(data=data, mime_type=mime))

        resp = self._client.models.generate_content(
            model=model,
            contents=parts,
            config=self._base_config(schema=schema),
        )
        usage = resp.usage_metadata
        return ProviderResponse(
            text=resp.text,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            provider="gemini",
        )


# ── OpenRouter (OpenAI-compatible SDK) ───────────────────────────────────
class KeyRotator:
    def __init__(self, keys: list[str]) -> None:
        if not keys:
            raise ValueError("At least one API key is required.")
        self._keys = keys
        self._idx = 0
        self._clients = [
            OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=k,
                default_headers={
                    "HTTP-Referer": OPENROUTER_REFERER,
                    "X-Title": OPENROUTER_TITLE,
                },
            )
            for k in keys
        ]

    @property
    def client(self) -> OpenAI:
        return self._clients[self._idx]

    def rotate(self) -> None:
        self._idx = (self._idx + 1) % len(self._keys)


class OpenRouterProvider(BaseProvider):
    def __init__(self, keys: list[str]) -> None:
        self._rotator = KeyRotator(keys)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(TENACITY_MAX_ATTEMPTS),
        wait=tenacity.wait_exponential(multiplier=2, min=1, max=10),
        retry=tenacity.retry_if_exception_type((
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        )),
        before_sleep=lambda s: print(
            f"  [openrouter] tenacity retry {s.attempt_number}/"
            f"{TENACITY_MAX_ATTEMPTS - 1} "
            f"after {type(s.outcome.exception()).__name__}"
        ),
        reraise=True,
    )
    def _do_call(self, **kwargs: Any) -> Any:
        return self._rotator.client.chat.completions.create(**kwargs)

    def _api_call(
        self, model: str, messages: list[dict], max_attempts: int = 3
    ) -> Any:
        for attempt in range(max_attempts):
            try:
                return self._do_call(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2048,
                    timeout=API_TIMEOUT,
                )
            except openai.RateLimitError:
                if attempt < max_attempts - 1 and len(self._rotator._keys) > 1:
                    self._rotator.rotate()
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("OpenRouter: unexpected exit")

    def call_text(
        self, model: str, system_prompt: str, user_text: str,
        schema: Optional[type] = None,
    ) -> ProviderResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        resp = self._api_call(model, messages)
        return ProviderResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
            provider="openrouter",
        )

    def call_vision(
        self,
        model: str,
        system_prompt: str,
        text_prompt: str,
        image_paths: list[Path],
        schema: Optional[type] = None,
    ) -> ProviderResponse:
        content: list[dict] = [{"type": "text", "text": text_prompt}]
        for p in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": encode_image_to_data_uri(p)},
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        resp = self._api_call(model, messages)
        return ProviderResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
            provider="openrouter",
        )


# ── Provider router (Gemini -> OpenRouter cascade) ───────────────────────
def _is_quota_error(exc: Exception) -> bool:
    """Check if an exception is a quota-exhausted error from any provider."""
    # OpenAI RateLimitError (OpenRouter)
    if isinstance(exc, openai.RateLimitError):
        return True
    # Google genai ClientError with 429 (Gemini SDK)
    try:
        from google.genai.errors import ClientError as GenaiClientError
        if isinstance(exc, GenaiClientError):
            return getattr(exc, "code", None) == 429
    except ImportError:
        pass
    # Legacy google.api_core ResourceExhausted
    try:
        from google.api_core.exceptions import ResourceExhausted
        if isinstance(exc, ResourceExhausted):
            return True
    except ImportError:
        pass
    return False


class ProviderRouter:
    """Tries providers in order. On quota exhaustion, cascades to next."""

    def __init__(
        self,
        gemini_key: str,
        openrouter_keys: list[str],
        provider_order: Optional[list[str]] = None,
    ) -> None:
        self._providers: dict[str, BaseProvider] = {}
        if gemini_key:
            self._providers["gemini"] = GeminiProvider(gemini_key)
        if openrouter_keys:
            self._providers["openrouter"] = OpenRouterProvider(openrouter_keys)
        self._order = provider_order or ["gemini", "openrouter"]
        self._order = [p for p in self._order if p in self._providers]
        self._last_provider: dict[str, str] = {}  # stage -> provider name

    @property
    def active_providers(self) -> list[str]:
        return self._order

    def call_text(
        self,
        models: dict[str, dict[str, str]],
        stage: str,
        system_prompt: str,
        user_text: str,
        schema: Optional[type] = None,
    ) -> ProviderResponse:
        for name in self._order:
            model = models.get(name, {}).get(stage, "")
            if not model:
                continue
            try:
                resp = self._providers[name].call_text(
                    model, system_prompt, user_text, schema=schema
                )
                self._last_provider[stage] = name
                return resp
            except Exception as exc:
                if _is_quota_error(exc):
                    print(f"  [{stage}] {name} quota exhausted -> trying next provider")
                    continue
                raise
        raise RuntimeError(f"[{stage}] All providers exhausted")

    def call_vision(
        self,
        models: dict[str, dict[str, str]],
        stage: str,
        system_prompt: str,
        text_prompt: str,
        image_paths: list[Path],
        schema: Optional[type] = None,
    ) -> ProviderResponse:
        for name in self._order:
            model = models.get(name, {}).get(stage, "")
            if not model:
                continue
            try:
                resp = self._providers[name].call_vision(
                    model, system_prompt, text_prompt, image_paths, schema=schema
                )
                self._last_provider[stage] = name
                return resp
            except Exception as exc:
                if _is_quota_error(exc):
                    print(f"  [{stage}] {name} quota exhausted -> trying next provider")
                    continue
                raise
        raise RuntimeError(f"[{stage}] All providers exhausted")


# ── Mock fallback ────────────────────────────────────────────────────────
def mock_fallback_response() -> ProviderResponse:
    import json
    text = json.dumps({
        "evidence_standard_met": False,
        "evidence_standard_met_reason": "API keys exhausted; fallback applied.",
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "No API keys available. Manual review required.",
        "supporting_image_ids": "none",
        "valid_image": False,
        "severity": "unknown",
    })
    return ProviderResponse(text=text, provider="mock")
