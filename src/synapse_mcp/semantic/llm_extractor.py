"""LLM-based semantic extraction module."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
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
    object: str = Field(default="unknown")
    predicate: str = Field(default="related_to")
    confidence: float = Field(default=0.5)
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

    async def extract_semantics(
        self, text: str, source_name: str | None = None
    ) -> ExtractionResult:
        """
        Extract entities and relations from text using an LLM.

        Args:
            text: The raw text to analyze.
            source_name: Optional name or path of the source.

        Returns:
            An ExtractionResult containing discovered entities and relations.
        """
        extraction_instructions = (
            "You are an expert knowledge extraction processor. Your task is to extract "
            "high-signal entities and their semantic relationships into JSON format from the provided source.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. Focus ONLY on core technical, conceptual, or factual knowledge.\n"
            "2. DO NOT extract: navigation labels, UI button text, citation counts, "
            "download counts, 'HuggingFace' page furniture, or page metadata.\n"
            "3. DO NOT extract: generic fragments like 'the article' or 'this model'.\n"
            "4. Predicates must be precise, directed ontological morphisms (e.g., 'implements_paradigm', 'instantiates_pattern', 'formalizes').\n"
            "5. Use consistent, snake_case or SCREAMING_SNAKE_CASE for entity types.\n"
            "7. If the source contains URLs, DO NOT extract them as entities unless they are primary subjects.\n\n"
            "CRITICAL: Output ONLY valid JSON format. No chat, no summary, no markdown formatting outside the JSON, simply analyse and fill in the JSON object.\n\n"
            "OUTPUT FORMAT: Return a JSON object with:\n"
            "- 'entities': list of { 'text': str, 'type': str, 'confidence': float }\n"
            "- 'relations': list of { 'subject': str, 'object': str, 'predicate': str, 'confidence': float }"
        )

        source_info = f" [Source: {source_name}]" if source_name else ""
        prompt = f"Extract knowledge from the following source{source_info}:\n\n{text}"

        payload = {
            "model": self.model,
            "system": extraction_instructions,  # Top-level hard control
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_ctx": 131072,  # Full 128K context for Gemma 4
                "num_predict": 4096,
                "num_keep": -1,  # Keep the system instructions in KV cache
                "top_k": 1,
                "top_p": 0.1,
            },
            "think": True,  # Allow frontier reasoning (segregated in response)
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
                        content = (
                            result_json.get("message", {}).get("content", "").strip()
                        )
                        thinking = result_json.get("message", {}).get("thinking", "")
                        if thinking:
                            logger.debug(
                                "Gemma Reasoning trace: %s", thinking[:200] + "..."
                            )
                        if not content:
                            logger.warning("Ollama returned empty content for message.")
                            continue

                        try:
                            # 1. Strip <think>, <thought>, or <reasoning> blocks
                            content = re.sub(
                                r"<(?:think|thought|reasoning)>[\s\S]*?</(?:think|thought|reasoning)>",
                                "",
                                content,
                            )

                            # 2. Clean markdown code fences if present
                            if "```" in content:
                                # Try to find json block specifically
                                json_match = re.search(
                                    r"```(?:json)?\s*([\s\S]*?)\s*```", content
                                )
                                if json_match:
                                    content = json_match.group(1)
                                else:
                                    # Fallback: just strip all fences
                                    content = (
                                        content.replace("```json", "")
                                        .replace("```", "")
                                        .strip()
                                    )

                            # 3. Try to find the first '{' and last '}' to skip remaining preamble/postamble
                            start_idx = content.find("{")
                            end_idx = content.rfind("}")
                            if start_idx != -1 and end_idx != -1:
                                content = content[start_idx : end_idx + 1]

                            data = json.loads(content)
                        except (json.JSONDecodeError, ValueError) as je:
                            logger.warning(
                                "Failed to parse LLM content as JSON: %s...",
                                content[:100],
                            )
                            raise je

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
