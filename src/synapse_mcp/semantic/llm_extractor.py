"""LLM-based semantic extraction module."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import aiohttp
from pydantic import BaseModel, Field

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
    object: str
    predicate: str
    confidence: float
    source_span: str | None = None


class ExtractionResult(BaseModel):
    """Container for semantic extraction results."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


class LlmExtractor:
    """Extractor that uses an LLM to identify entities and relations."""

    base_url: str
    model: str
    api_url: str

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        """Initialize the extractor with base URL and model name."""
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        )
        self.model = model or os.getenv("OLLAMA_EXTRACTION_MODEL") or "gemma4:latest"
        self.api_url = f"{self.base_url.rstrip('/')}/api/chat"

    async def extract_semantics(self, text: str) -> ExtractionResult:
        """
        Extract entities and relations from text using an LLM.

        Args:
            text: The raw text to analyze.

        Returns:
            An ExtractionResult containing discovered entities and relations.
        """
        extraction_instructions = (
            "You are an expert knowledge extraction agent. Your task is to extract "
            "high-signal entities and their semantic relationships from the provided text.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. Focus ONLY on core technical, conceptual, or factual knowledge.\n"
            "2. DO NOT extract: navigation labels, UI button text, citation counts, "
            "download counts, 'HuggingFace' page furniture, or page metadata.\n"
            "3. DO NOT extract: generic fragments like 'the article' or 'this model'.\n"
            "4. Relationship types must be specific and descriptive (e.g., 'trained_on', 'achieves_performance').\n"
            "5. Use consistent, snake_case or SCREAMING_SNAKE_CASE for entity types.\n"
            "6. If the text contains URLs, DO NOT extract them as entities unless they are primary subjects.\n\n"
            "OUTPUT FORMAT: Return a JSON object with:\n"
            "- 'entities': list of { 'text': str, 'type': str, 'confidence': float }\n"
            "- 'relations': list of { 'subject': str, 'object': str, 'predicate': str, 'confidence': float }"
        )

        prompt = f"Extract knowledge from the following text:\n\n{text}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": extraction_instructions,
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
        }

        # Use a generous default for local models, configurable via env
        env_timeout = int(os.getenv("OLLAMA_TIMEOUT", "120"))
        timeout = aiohttp.ClientTimeout(total=env_timeout)
        for attempt in range(2):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(self.api_url, json=payload) as response:
                        if response.status >= 400:
                            error_text = await response.text()
                            logger.warning(
                                "Ollama API returned error %d: %s",
                                response.status,
                                error_text,
                            )
                            response.raise_for_status()

                        result_json = await response.json()
                        content = result_json.get("message", {}).get("content", "{}")
                        data = json.loads(content)
                        raw_result = ExtractionResult.model_validate(data)
                        return self._sanitize_results(raw_result)
            except (
                aiohttp.ClientError,
                json.JSONDecodeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                logger.warning("Extraction attempt %d failed: %s", attempt + 1, e)
                if attempt == 1:
                    logger.error(
                        "All extraction attempts failed for text: %s...", text[:50]
                    )
                    # We don't want to crash the whole pipeline if LLM fails,
                    # so we return an empty result and let the integrator fallback.
                    return ExtractionResult()

        return ExtractionResult()

    def _sanitize_results(self, result: ExtractionResult) -> ExtractionResult:
        """Filter out low-signal entities and furniture leakage."""
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
        valid_entity_names = set()

        for entity in result.entities:
            name_lower = entity.text.lower().strip()

            # Skip junk types
            if entity.type.upper() in junk_types:
                continue

            # Skip pure numbers or single characters
            if name_lower.isdigit() or len(name_lower) < 2:
                continue

            # Skip common UI headers
            if name_lower in junk_names:
                continue

            # Skip obvious UI fragments (e.g., "about 24", "8192 4096")
            if (
                any(char.isdigit() for char in name_lower)
                and len(name_lower.split()) < 3
                and not any(c.isalpha() for c in name_lower)
            ):
                continue

            filtered_entities.append(entity)
            valid_entity_names.add(entity.text)

        # Filter relations to only include those between valid entities
        filtered_relations = [
            rel
            for rel in result.relations
            if rel.subject in valid_entity_names and rel.object in valid_entity_names
        ]

        return ExtractionResult(
            entities=filtered_entities, relations=filtered_relations
        )
