"""
LLM Provider Abstraction for Finorbit LLM Backend.

Provides a unified interface for OpenAI and Gemini providers, mirroring
the pattern already used in Finorbit_RAG/core/llm_setup.py.

Usage:
    from backend.core.llm_provider import get_llm_provider

    provider = get_llm_provider()
    text = await provider.async_complete(user_prompt="...", system_prompt="...")

Environment variables:
    LLM_PROVIDER       "openai" (default) or "gemini"
    LLM_API_KEY        OpenAI secret key  (required when provider=openai)
    GOOGLE_API_KEY     Google/Gemini key  (required when provider=gemini)
    CUSTOM_MODEL_NAME  Model override (e.g. "gpt-4o", "gemini-1.5-flash")
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Provider implementations
# ─────────────────────────────────────────────────────────────────────────────

class _OpenAIProvider:
    """Thin wrapper around the OpenAI chat-completions API."""

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model
        logger.info(f"[LLMProvider] OpenAI initialised — model={model}")

    def complete(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    async def async_complete(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        return await asyncio.to_thread(
            self.complete,
            user_prompt,
            system_prompt,
            max_tokens,
            temperature,
        )

    # Expose the raw client so callers that need it directly can access it.
    @property
    def raw_client(self):
        return self._client

    @property
    def model_name(self) -> str:
        return self._model


class _GeminiProvider:
    """Thin wrapper around the Google Generative AI SDK (google-generativeai)."""

    def __init__(self, api_key: str, model: str) -> None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._model_name = model
            self._genai = genai
            logger.info(f"[LLMProvider] Gemini initialised — model={model}")
        except ImportError:
            raise ImportError(
                "google-generativeai package is required for Gemini provider. "
                "Install with: pip install google-generativeai"
            )

    def complete(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        model = self._genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=system_prompt or None,
            generation_config=self._genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        full_prompt = user_prompt
        response = model.generate_content(full_prompt)
        return (response.text or "").strip()

    async def async_complete(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        return await asyncio.to_thread(
            self.complete,
            user_prompt,
            system_prompt,
            max_tokens,
            temperature,
        )

    @property
    def model_name(self) -> str:
        return self._model_name


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

# Type alias for callers that want type hints
LLMProvider = _OpenAIProvider | _GeminiProvider

_singleton: Optional[LLMProvider] = None
_singleton_lock = asyncio.Lock() if False else __import__("threading").Lock()


def get_llm_provider() -> LLMProvider:
    """
    Return the module-level LLM provider singleton.

    The singleton is created on first call and reused on subsequent calls.
    Provider selection order:
      1. LLM_PROVIDER env var ("openai" | "gemini")
      2. Defaults to "openai"

    Raises:
        RuntimeError: if the selected provider cannot be initialised
                      (e.g. missing API key or missing package).
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    with _singleton_lock:
        if _singleton is not None:
            return _singleton

        provider_name = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        model_override = os.getenv("CUSTOM_MODEL_NAME", "").strip()

        if provider_name == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError(
                    "LLM_PROVIDER=gemini but GOOGLE_API_KEY is not set."
                )
            model = model_override or "gemini-1.5-flash"
            _singleton = _GeminiProvider(api_key=api_key, model=model)

        else:
            # Default: openai
            if provider_name not in ("openai",):
                logger.warning(
                    f"[LLMProvider] Unknown provider '{provider_name}', defaulting to 'openai'."
                )
            api_key = os.getenv("LLM_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError(
                    "LLM_PROVIDER=openai but LLM_API_KEY is not set."
                )
            model = model_override or "gpt-4o-mini"
            _singleton = _OpenAIProvider(api_key=api_key, model=model)

        return _singleton


def reset_provider() -> None:
    """Reset the singleton (useful for testing with different env vars)."""
    global _singleton
    with _singleton_lock:
        _singleton = None
