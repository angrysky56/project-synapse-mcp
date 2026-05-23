"""LLM provider implementations for chat completions with strict JSON output.

Three providers ship today:

  - :class:`OllamaProvider` — local Ollama server. Uses ``/api/chat`` with
    ``format: "json"`` (or a full JSON schema dict).
  - :class:`MinimaxProvider` — MiniMax via its OpenAI-compatible endpoint at
    ``https://api.minimax.io/v1``. Auth: ``Authorization: Bearer sk-cp-...``.
  - :class:`OpenRouterProvider` — OpenRouter chat completions at
    ``https://openrouter.ai/api/v1``. Proxies many model vendors behind a
    single OpenAI-style API; useful when you want to A/B different models
    without coding a new provider.

All providers expose the same async ``chat_json(system, user, *, ...)`` method
which returns a parsed ``dict``. They also handle the well-known failure mode
where models leak chain-of-thought or markdown around the JSON: see
:func:`_salvage_json` for the cleanup pipeline.
"""

from __future__ import annotations

import abc
import asyncio
import json
import os
import re
from typing import Any

import aiohttp

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class LlmResponseError(RuntimeError):
    """Raised when an LLM call produces an unrecoverable response."""


# Reasoning models (DeepSeek-R1, QwQ, Gemma-think, etc.) emit chain-of-thought
# inside <think>/<thought>/<reasoning> blocks even when asked for JSON. Strip
# them before json.loads().
_THINK_BLOCK_RE = re.compile(
    r"<(?:think|thought|reasoning)>[\s\S]*?</(?:think|thought|reasoning)>",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```")


def _salvage_json(content: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response that may contain prose.

    Strategy, in order:

    1. Strip ``<think>`` / ``<thought>`` / ``<reasoning>`` blocks.
    2. Unwrap markdown ``` ``` ``` code fences (with or without ``json`` lang tag).
    3. Slice between the first ``{`` and last ``}`` to skip preamble/postamble.
    4. ``json.loads`` the result.

    Raises:
        LlmResponseError: if no valid JSON object can be extracted.
    """
    if not content or not content.strip():
        raise LlmResponseError("LLM returned empty content")

    cleaned = _THINK_BLOCK_RE.sub("", content).strip()

    fence_match = _FENCE_RE.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1)
    elif "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LlmResponseError(
            f"No JSON object found in LLM response (preview: {content[:200]!r})"
        )

    cleaned = cleaned[start : end + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LlmResponseError(
            f"LLM response is not valid JSON: {e}. Preview: {cleaned[:200]!r}"
        ) from e


class LlmProvider(abc.ABC):
    """Abstract base for chat-completion providers with JSON output."""

    #: Short slug used in logs and env-var prefixes (``ollama``, ``minimax``…).
    name: str = "abstract"
    #: The model identifier this instance will call.
    model: str = ""
    #: The base URL this instance will hit.
    base_url: str = ""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = base_url or ""
        self.model = model or ""

    @abc.abstractmethod
    async def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        schema: dict[str, Any] | None = None,
        timeout: int = 120,
    ) -> dict[str, Any]:
        """Send a system+user prompt and return parsed JSON.

        Args:
            system: System prompt — hard rules, output schema description.
            user: User message — the actual content to process.
            temperature: Sampling temperature. Default 0 for determinism.
            max_tokens: Max output tokens.
            schema: Optional JSON schema. Providers that support structured
                outputs (OpenAI ``response_format=json_schema``, Ollama
                ``format=<schema>``) will use it; others fall back to a plain
                ``json_object`` mode.
            timeout: Per-request timeout in seconds.

        Returns:
            Parsed JSON object as a dict.

        Raises:
            LlmResponseError: on network error, non-2xx, empty body, or
                unparseable JSON.
        """

    @abc.abstractmethod
    async def check_available(self) -> tuple[bool, str]:
        """Probe the provider; return ``(ok, message)``.

        Should be cheap and fast — used at startup to fail fast with an
        actionable error instead of crashing on the first real call.
        """


class OllamaProvider(LlmProvider):
    """Local Ollama server. Uses ``/api/chat`` with ``format=json``."""

    name = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_EXTRACTION_MODEL") or "gemma4:latest"

    async def check_available(self) -> tuple[bool, str]:
        url = f"{self.base_url}/api/tags"
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status >= 400:
                        return False, f"Ollama returned HTTP {resp.status}"
                    data = await resp.json()
        except Exception as e:  # pylint: disable=broad-exception-caught
            return False, f"Cannot reach Ollama at {self.base_url}: {e}"

        # Ollama tag names may or may not include ':latest' — accept either.
        names = {m.get("name", "") for m in data.get("models", []) or []}
        wanted = {self.model, f"{self.model}:latest", self.model.split(":")[0]}
        if not names & wanted:
            return (
                False,
                f"Model '{self.model}' not loaded in Ollama. "
                f"Run: ollama pull {self.model}",
            )
        return True, "ok"

    async def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        schema: dict[str, Any] | None = None,
        timeout: int = 120,
    ) -> dict[str, Any]:
        # Ollama accepts either the string "json" or a full JSON schema dict
        # for `format`. Schema gives stronger guarantees on newer models.
        fmt: str | dict[str, Any] = schema if schema else "json"
        payload: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "stream": False,
            "format": fmt,
            "options": {
                "temperature": temperature,
                "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "131072")),
                "num_predict": max_tokens,
                "num_keep": -1,
                "top_k": 1,
                "top_p": 0.1,
            },
            # Allow reasoning models to think — segregated from `content` by
            # Ollama itself, so it doesn't pollute the JSON we parse.
            "think": os.getenv("OLLAMA_THINK", "true").lower() == "true",
        }

        api_url = f"{self.base_url}/api/chat"
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.post(api_url, json=payload) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        raise LlmResponseError(
                            f"Ollama HTTP {resp.status}: {body[:300]}"
                        )
                    result = await resp.json()
        except aiohttp.ClientError as e:
            raise LlmResponseError(f"Ollama request failed: {e}") from e
        except asyncio.TimeoutError as e:
            raise LlmResponseError(f"Ollama request timed out after {timeout}s") from e

        content = (result.get("message") or {}).get("content", "").strip()
        return _salvage_json(content)


class _OpenAICompatibleProvider(LlmProvider):
    """Shared implementation for OpenAI-compatible chat APIs.

    Both MiniMax and OpenRouter implement the OpenAI ``/chat/completions``
    contract, so all the wire-format logic lives here. Subclasses just set
    the defaults (``default_base_url``, ``default_model``, ``api_key_env``).
    """

    name = "openai-compat"
    api_key_env: str = ""
    default_base_url: str = ""
    default_model: str = ""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv(self.api_key_env, "")
        prefix = self.name.upper()
        self.base_url = (
            base_url or os.getenv(f"{prefix}_BASE_URL") or self.default_base_url
        ).rstrip("/")
        self.model = model or os.getenv(f"{prefix}_MODEL") or self.default_model
        # Subclasses can append vendor-specific headers (e.g. OpenRouter
        # wants HTTP-Referer / X-Title for analytics).
        self.extra_headers: dict[str, str] = {}

    async def check_available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, f"Missing API key (env: {self.api_key_env})"
        url = f"{self.base_url}/models"
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                **self.extra_headers,
            }
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 401:
                        return False, f"{self.name}: invalid API key (HTTP 401)"
                    if resp.status >= 500:
                        return False, f"{self.name}: server error HTTP {resp.status}"
        except Exception as e:  # pylint: disable=broad-exception-caught
            return False, f"{self.name}: cannot reach {self.base_url}: {e}"
        return True, "ok"

    async def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        schema: dict[str, Any] | None = None,
        timeout: int = 120,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise LlmResponseError(
                f"{self.name}: API key missing (set {self.api_key_env})"
            )

        # Try strict json_schema first if a schema was given; some servers
        # reject the json_schema variant — fall back to plain json_object on
        # 4xx. Most newer servers (OpenAI, OpenRouter for many models,
        # MiniMax-M2.7) accept json_schema; older/lighter ones don't.
        if schema:
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "synapse_response",
                    "schema": schema,
                    "strict": True,
                },
            }
        else:
            response_format = {"type": "json_object"}

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        url = f"{self.base_url}/chat/completions"
        client_timeout = aiohttp.ClientTimeout(total=timeout)

        last_err: Exception | None = None
        result_json: dict[str, Any] | None = None
        for attempt in range(2):
            try:
                async with aiohttp.ClientSession(timeout=client_timeout) as session:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        body_text = await resp.text()
                        if resp.status >= 400:
                            # Downgrade json_schema -> json_object on first 4xx
                            # before giving up.
                            if (
                                attempt == 0
                                and 400 <= resp.status < 500
                                and payload["response_format"].get("type")
                                == "json_schema"
                            ):
                                logger.warning(
                                    "%s rejected json_schema (HTTP %d); "
                                    "retrying with json_object. Body: %s",
                                    self.name,
                                    resp.status,
                                    body_text[:200],
                                )
                                payload["response_format"] = {"type": "json_object"}
                                continue
                            raise LlmResponseError(
                                f"{self.name} HTTP {resp.status}: {body_text[:300]}"
                            )
                        try:
                            result_json = json.loads(body_text)
                        except json.JSONDecodeError as e:
                            raise LlmResponseError(
                                f"{self.name}: response body is not JSON: {e}"
                            ) from e
                        break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                if attempt == 1:
                    raise LlmResponseError(f"{self.name} request failed: {e}") from e
                logger.warning("%s attempt %d failed: %s", self.name, attempt + 1, e)
                await asyncio.sleep(0.5)

        if result_json is None:
            raise LlmResponseError(f"{self.name}: exhausted retries ({last_err})")

        choices = result_json.get("choices") or []
        if not choices:
            raise LlmResponseError(f"{self.name}: no choices in response")
        content = (choices[0].get("message") or {}).get("content", "") or ""
        return _salvage_json(content.strip())


class MinimaxProvider(_OpenAICompatibleProvider):
    """MiniMax via its OpenAI-compatible endpoint.

    Docs: https://platform.minimax.io/docs/token-plan/other-tools

    Set ``MINIMAX_API_KEY`` (the ``sk-cp-...`` Token Plan key). Optionally
    override ``MINIMAX_BASE_URL`` and ``MINIMAX_MODEL``.
    """

    name = "minimax"
    api_key_env = "MINIMAX_API_KEY"
    default_base_url = "https://api.minimax.io/v1"
    # MiniMax-M2.7-highspeed is the latency-optimized variant — preferred for
    # insight synthesis where we make many small calls.
    default_model = "MiniMax-M2.7-highspeed"


class OpenRouterProvider(_OpenAICompatibleProvider):
    """OpenRouter chat completions (proxies many model vendors).

    Docs: https://openrouter.ai/docs/api-reference/chat-completion

    Set ``OPENROUTER_API_KEY``. Optionally override ``OPENROUTER_MODEL``
    (defaults to a fast JSON-strong model) and ``OPENROUTER_BASE_URL``.

    OpenRouter recommends sending ``HTTP-Referer`` and ``X-Title`` headers
    for analytics and rate-limit allocation — set ``OPENROUTER_HTTP_REFERER``
    and ``OPENROUTER_X_TITLE`` if you have a public app URL.
    """

    name = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"
    default_base_url = "https://openrouter.ai/api/v1"
    # Sensible default: small, fast, strong on structured output.
    default_model = "openai/gpt-4o-mini"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(base_url=base_url, model=model, api_key=api_key)
        referer = os.getenv("OPENROUTER_HTTP_REFERER", "")
        title = os.getenv("OPENROUTER_X_TITLE", "project-synapse-mcp")
        if referer:
            self.extra_headers["HTTP-Referer"] = referer
        if title:
            self.extra_headers["X-Title"] = title


# Registry — lowercase keys, used by get_provider().
_PROVIDERS: dict[str, type[LlmProvider]] = {
    "ollama": OllamaProvider,
    "minimax": MinimaxProvider,
    "openrouter": OpenRouterProvider,
}


def get_provider(
    name: str | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> LlmProvider:
    """Resolve a provider by name.

    Args:
        name: Provider slug (``ollama``, ``minimax``, ``openrouter``).
            If None, reads ``LLM_PROVIDER`` env var, then defaults to ``ollama``.
        model: Override the provider's default model.
        base_url: Override the provider's default base URL.

    Returns:
        An instantiated :class:`LlmProvider` ready for ``chat_json`` calls.

    Raises:
        ValueError: if ``name`` is not a registered provider.
    """
    resolved = (name or os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
    cls = _PROVIDERS.get(resolved)
    if cls is None:
        raise ValueError(
            f"Unknown LLM provider: {resolved!r}. "
            f"Available: {sorted(_PROVIDERS.keys())}"
        )
    return cls(base_url=base_url, model=model)
