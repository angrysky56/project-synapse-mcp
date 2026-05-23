"""LLM provider abstraction for Project Synapse.

Pluggable providers for chat completion with strict JSON output. Used by:

  - ``semantic.llm_extractor`` for entity/relation extraction during ingest
  - ``zettelkasten.insight_engine`` for insight synthesis

Provider selected via env var ``LLM_PROVIDER`` (``ollama`` | ``minimax`` |
``openrouter``). Separate env vars per concern allow extraction and insight
synthesis to use different models — e.g. local Ollama for extraction (cheap,
high-volume) and a cloud model for insight synthesis (rare, needs to be
sharp).
"""

from .providers import (
    LlmProvider,
    LlmResponseError,
    MinimaxProvider,
    OllamaProvider,
    OpenRouterProvider,
    get_provider,
)

__all__ = [
    "LlmProvider",
    "LlmResponseError",
    "MinimaxProvider",
    "OllamaProvider",
    "OpenRouterProvider",
    "get_provider",
]
