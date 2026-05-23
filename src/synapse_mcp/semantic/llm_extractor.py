"""LLM-based semantic extraction module.

Uses the pluggable :mod:`synapse_mcp.llm` provider abstraction so the same
extractor can run against local Ollama (default) or cloud APIs (MiniMax,
OpenRouter, …) without code changes — see ``EXTRACTION_LLM_PROVIDER`` and
``EXTRACTION_LLM_MODEL`` env vars, or fall back to ``LLM_PROVIDER`` /
``LLM_MODEL``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import BaseModel, Field

from ..llm import LlmProvider, LlmResponseError, get_provider

logger = logging.getLogger(__name__)


class ExtractedEntity(BaseModel):
    """Semantic entity extracted from text."""

    text: str
    type: str
    confidence: float
    properties: dict[str, Any] = Field(default_factory=dict)


class ExtractedRelation(BaseModel):
    """Semantic relation extracted from text."""

    subject: str
    object: str = Field(default="unknown")
    predicate: str = Field(default="related_to")
    confidence: float = Field(default=0.5)
    source_span: str | None = None


class ExtractionResult(BaseModel):
    """Container for semantic extraction results."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


# Kept as a module-level constant so the prompt is easy to diff against past
# versions when tuning extraction quality. Changes here have direct, large
# impact on what ends up in the knowledge graph.
EXTRACTION_INSTRUCTIONS = (
    "You are an expert knowledge extraction processor. Your task is to extract "
    "high-signal entities and their semantic relationships into JSON format from "
    "the provided source.\n\n"
    "CRITICAL CONSTRAINTS:\n"
    "1. Focus ONLY on core technical, conceptual, or factual knowledge.\n"
    "2. DO NOT extract: navigation labels, UI button text, citation counts, "
    "download counts, 'HuggingFace' page furniture, or page metadata.\n"
    "3. DO NOT extract: generic fragments like 'the article' or 'this model'.\n"
    "4. DO NOT extract code symbols (function names, class names, variable "
    "names) as Organizations or Products. Patterns like 'BuildGraph', "
    "'OnInit', 'find_swing_low', 'XAUUSD' are CODE — either skip them "
    "entirely or label them with type 'CodeSymbol'.\n"
    "5. DO NOT extract author surnames that appear ONLY in a bibliography "
    "or reference list (e.g., 'Schramm', 'Channon' appearing only in "
    "'(Schramm et al., 2011; Channon, 2006)'). Extract a Person only if "
    "they are discussed substantively in the narrative.\n"
    "6. Predicates must be technical, directed graph relations (e.g., "
    "'implements', 'depends_on', 'utilizes', 'defines', 'is_instance_of').\n"
    "7. Use consistent, snake_case or SCREAMING_SNAKE_CASE for entity types.\n"
    "8. Choose the MOST SPECIFIC entity type that applies. Use the controlled "
    "vocabulary below FIRST; only invent a new type when nothing fits.\n"
    "   - Person, Organization, Demographic (nationality/religion/group)\n"
    "   - Location (place/facility), TemporalEntity (date/time)\n"
    "   - Product (named software/hardware/system), Method (algorithm/technique),\n"
    "     Architecture (neural net/system design), Dataset, Benchmark, Library,\n"
    "     Language (programming or natural), Framework, Model (specific named model)\n"
    "   - CodeSymbol (function/class/variable identifiers from source code)\n"
    "   - Concept (USE SPARINGLY — only for genuine abstract ideas that don't fit\n"
    "     anything else: e.g. 'emergence', 'consciousness', 'lumpability'. \n"
    "     NEVER label a specific method, system, or named thing as 'Concept'.)\n"
    "   - Theorem, Law, Principle (for formal mathematical or scientific results)\n"
    "   - Quantity (numbers, percentages, measurements)\n"
    "   - CreativeWork (papers, books, articles, songs)\n"
    "   - Event (named events, conferences, releases)\n"
    "9. If the source contains URLs, DO NOT extract them as entities unless they "
    "are primary subjects.\n\n"
    "CRITICAL: Output ONLY valid JSON format. No chat, no summary, no markdown "
    "formatting outside the JSON, simply analyse and fill in the JSON object.\n\n"
    "OUTPUT FORMAT: Return a JSON object with:\n"
    "- 'entities': list of { 'text': str, 'type': str, 'confidence': float }\n"
    "- 'relations': list of { 'subject': str, 'object': str, 'predicate': str, "
    "'confidence': float }"
)


class LlmExtractor:
    """Extractor that uses an LLM to identify entities and relations.

    The extractor is provider-agnostic. By default it reads
    ``EXTRACTION_LLM_PROVIDER`` (falling back to ``LLM_PROVIDER``, then
    ``ollama``) and ``EXTRACTION_LLM_MODEL`` (falling back to provider
    defaults). This lets the heavier insight-synthesis path use a cloud
    model while extraction stays on local Ollama for cost reasons — or
    vice versa.
    """

    provider: LlmProvider

    def __init__(
        self,
        provider: LlmProvider | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the extractor.

        Args:
            provider: Optional pre-built provider. If None, one is resolved
                from ``EXTRACTION_LLM_PROVIDER`` / ``LLM_PROVIDER`` env vars.
            base_url: Override base URL (only used if ``provider`` is None).
            model: Override model (only used if ``provider`` is None).
        """
        if provider is not None:
            self.provider = provider
        else:
            provider_name = (
                os.getenv("EXTRACTION_LLM_PROVIDER") or os.getenv("LLM_PROVIDER")
            )
            resolved_model = (
                model
                or os.getenv("EXTRACTION_LLM_MODEL")
                or os.getenv("OLLAMA_EXTRACTION_MODEL")
            )
            self.provider = get_provider(
                provider_name, model=resolved_model, base_url=base_url
            )

    # Backwards-compatible attribute shims — older call sites read
    # ``.base_url`` and ``.model`` directly (e.g. semantic_integrator's
    # startup log line).
    @property
    def base_url(self) -> str:
        return self.provider.base_url

    @property
    def model(self) -> str:
        return self.provider.model

    async def check_available(self) -> tuple[bool, str]:
        """Fast probe via the underlying provider.

        Lets the server fail fast at startup with an actionable error
        message instead of crashing on the first ingest.
        """
        return await self.provider.check_available()

    async def extract_semantics(
        self, text: str, source_name: str | None = None
    ) -> ExtractionResult:
        """Extract entities and relations from text using the configured LLM.

        Args:
            text: The raw text to analyze.
            source_name: Optional name or path of the source.

        Returns:
            An :class:`ExtractionResult` with discovered entities and relations.
            Returns an empty result on extractor failure (never raises) so the
            ingest pipeline can keep going and the Montague parser can serve
            as a fallback.
        """
        source_info = f" [Source: {source_name}]" if source_name else ""
        prompt = (
            f"Extract knowledge from the following source{source_info}:\n\n{text}"
        )

        # 16384 was the value tuned for Gemma — leaves plenty of room for
        # entity-dense documents while still capping runaway generations.
        max_tokens = int(os.getenv("EXTRACTION_MAX_TOKENS", "16384"))
        timeout = int(os.getenv("OLLAMA_TIMEOUT", "120"))

        for attempt in range(2):
            try:
                data = await self.provider.chat_json(
                    system=EXTRACTION_INSTRUCTIONS,
                    user=prompt,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                raw_result = ExtractionResult.model_validate(data)
                if not raw_result.relations:
                    # Visibility: knowing whether the model produced zero
                    # relations vs produced some that failed validation is
                    # the difference between "model is being weird about
                    # this text" and "our schema is too strict".
                    logger.info(
                        "LLM extraction returned 0 relations "
                        "(entities=%d). Source: %s",
                        len(raw_result.entities),
                        source_name or "<inline>",
                    )
                return self._sanitize_results(raw_result)
            except LlmResponseError as e:
                logger.warning(
                    "Extraction attempt %d via %s failed: %s",
                    attempt + 1,
                    self.provider.name,
                    e,
                )
                if attempt == 1:
                    logger.error(
                        "All extraction attempts failed for text: %s...",
                        text[:50],
                    )
                    # Don't crash the pipeline — return empty and let the
                    # integrator fall back to Montague.
                    return ExtractionResult()
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Unexpected error in extraction attempt %d: %s", attempt + 1, e
                )
                if attempt == 1:
                    return ExtractionResult()

        return ExtractionResult()

    def _sanitize_results(self, result: ExtractionResult) -> ExtractionResult:
        """Filter out low-signal entities and UI furniture leakage."""
        junk_types = {"LINK", "NAV_METADATA", "UI_LABEL", "METADATA"}
        junk_names = {
            "citation downloads",
            "dataset",
            "files",
            "community",
            "settings",
            "inference providers",
            "downloads",
            "last month",
        }

        filtered_entities = []
        valid_entity_names: set[str] = set()

        for entity in result.entities:
            name_lower = entity.text.lower().strip()

            if entity.type.upper() in junk_types:
                continue
            if name_lower.isdigit() or len(name_lower) < 2:
                continue
            if name_lower in junk_names:
                continue
            # Obvious UI fragments (e.g., "about 24", "8192 4096")
            if (
                any(char.isdigit() for char in name_lower)
                and len(name_lower.split()) < 3
                and not any(c.isalpha() for c in name_lower)
            ):
                continue

            filtered_entities.append(entity)
            valid_entity_names.add(entity.text)

        # Keep all extracted relations; KnowledgeGraph handles node creation.
        return ExtractionResult(
            entities=filtered_entities, relations=result.relations
        )
