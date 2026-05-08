"""
Semantic Integration Pipeline for Project Synapse.

This module bridges the text processor and Montague parser to create
a complete semantic analysis pipeline.
"""

import hashlib
import re
import string
from datetime import datetime
from typing import Any

from ..knowledge.knowledge_types import KnowledgeUtils
from ..semantic.montague_parser import MontagueParser
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class SemanticIntegrator:
    """
    Integrates text processing with semantic analysis to extract knowledge structures.

    This class coordinates between the text processor and Montague parser to create
    a unified semantic analysis pipeline that extracts entities, relationships, and facts.
    """

    def __init__(self, montague_parser: MontagueParser | None = None) -> None:
        """Initialize the semantic integrator."""
        self.montague_parser = montague_parser
        self.entity_cache: dict[str, dict[str, Any]] = {}  # Cache for deduplication
        self.logger = logger

    @logger.timer()
    async def initialize(self) -> None:
        """Initialize the semantic integrator with required components."""
        if self.montague_parser is None:
            self.montague_parser = MontagueParser()

        try:
            await self.montague_parser.initialize()
            logger.info("Semantic integrator initialized successfully")
        except Exception as e:
            logger.warning(
                "Montague parser initialization failed: %s", e, exc_info=True
            )
            logger.info("Continuing with basic text processing only")

    @logger.timer()
    async def process_text_with_semantics(
        self,
        text: str,
        source: str = "user_input",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process text through complete semantic analysis pipeline.

        Args:
            text: Raw text to process
            source: Source identifier for provenance
            metadata: Additional metadata about the text

        Returns:
            Dictionary containing entities, relationships, facts, and semantic analysis
        """
        logger.debug("Processing text with full semantic analysis: %s...", text[:100])

        # Start with basic text processing
        processed_data = await self._basic_text_processing(text, source, metadata)

        # Add semantic analysis if Montague parser is available
        if self.montague_parser and self.montague_parser.nlp:
            try:
                semantic_analysis = await self.montague_parser.parse_text(
                    processed_data["cleaned_text"]
                )
                await self._integrate_semantic_analysis(
                    processed_data, semantic_analysis, source
                )
            except Exception as e:
                logger.warning(
                    "Semantic analysis failed for source '%s', continuing with basic processing: %s",
                    source,
                    e,
                    exc_info=True,
                )

        # Post-process and validate
        await self._post_process_data(processed_data)

        logger.info(
            "Extracted %d entities, %d relationships, %d facts",
            len(processed_data["entities"]),
            len(processed_data["relationships"]),
            len(processed_data["facts"]),
        )

        return processed_data

    async def _basic_text_processing(
        self, text: str, source: str, metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Perform basic text processing and structure preparation."""
        # Clean and normalize text
        cleaned_text = self._clean_text(text)

        # Split into sentences
        sentences = await self._split_sentences(cleaned_text)

        # Generate unique ID
        text_id = self._generate_text_id(text, source)

        return {
            "text_id": text_id,
            "source": source,
            "metadata": metadata or {},
            "processed_at": datetime.now().isoformat(),
            "original_text": text,
            "cleaned_text": cleaned_text,
            "sentences": sentences,
            "entities": [],
            "relationships": [],
            "facts": [],
        }

    @logger.timer()
    async def _integrate_semantic_analysis(
        self,
        processed_data: dict[str, Any],
        semantic_analysis: dict[str, Any],
        source: str,
    ) -> None:
        """Integrate results from Montague parser into processed data."""

        # Extract and convert entities
        for entity_data in semantic_analysis.get("entities", []):
            entity = await self._create_entity_from_analysis(entity_data, source)
            if entity and entity["id"] not in self.entity_cache:
                processed_data["entities"].append(entity)
                self.entity_cache[entity["id"]] = entity

        # Extract and convert relationships, auto-creating entities for endpoints
        for relation_data in semantic_analysis.get("relations", []):
            # Auto-create entity nodes for relation endpoints not in entity cache
            for endpoint_name in [
                relation_data.get("subject"),
                relation_data.get("object"),
            ]:
                if not endpoint_name:
                    continue
                endpoint_id = KnowledgeUtils.generate_entity_id(
                    endpoint_name, "Entity"
                )
                if endpoint_id not in self.entity_cache:
                    entity = {
                        "id": endpoint_id,
                        "name": endpoint_name,
                        "type": "Concept",
                        "confidence": 0.6,
                        "source": source,
                        "properties": {
                            "original_label": "",
                            "start_char": -1,
                            "end_char": -1,
                        },
                    }
                    processed_data["entities"].append(entity)
                    self.entity_cache[endpoint_id] = entity

            relationship = await self._create_relationship_from_analysis(
                relation_data, source
            )
            if relationship:
                processed_data["relationships"].append(relationship)

        # Create facts from propositions
        for proposition in semantic_analysis.get("propositions", []):
            fact = await self._create_fact_from_proposition(proposition, source)
            if fact:
                processed_data["facts"].append(fact)

        # Add semantic features to metadata
        if "semantic_features" in semantic_analysis:
            processed_data["metadata"]["semantic_features"] = semantic_analysis[
                "semantic_features"
            ]

    async def _create_entity_from_analysis(
        self, entity_data: dict[str, Any], source: str
    ) -> dict[str, Any] | None:
        """Convert semantic analysis entity to knowledge graph format."""
        try:
            entity_id = KnowledgeUtils.generate_entity_id(
                entity_data["text"], entity_data["type"]
            )

            # Ensure all property values are primitive types for Neo4j compatibility
            properties: dict[str, Any] = {}

            # Handle original_label safely
            original_label = entity_data.get("label", "")
            if hasattr(original_label, "text"):
                properties["original_label"] = str(original_label.text)
            else:
                properties["original_label"] = (
                    str(original_label) if original_label else ""
                )

            # Handle character positions safely, ensuring they're integers
            start_char = entity_data.get("start", -1)
            end_char = entity_data.get("end", -1)

            if hasattr(start_char, "__int__"):
                properties["start_char"] = int(start_char)
            else:
                properties["start_char"] = (
                    int(start_char) if isinstance(start_char, int | float) else -1
                )

            if hasattr(end_char, "__int__"):
                properties["end_char"] = int(end_char)
            else:
                properties["end_char"] = (
                    int(end_char) if isinstance(end_char, int | float) else -1
                )

            return {
                "id": entity_id,
                "name": entity_data["text"],
                "type": entity_data.get("type", "Entity"),
                "confidence": float(entity_data.get("confidence", 0.8)),
                "source": source,
                "properties": properties,
            }
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            logger.warning(
                "Failed to create entity from analysis data '%s': %s",
                entity_data.get("text", "UNKNOWN"),
                e,
            )
            return None

    async def _create_relationship_from_analysis(
        self, relation_data: dict[str, Any], source: str
    ) -> dict[str, Any] | None:
        """Convert semantic analysis relation to knowledge graph format."""
        try:
            # Create entity IDs for subject and object
            subject_id = KnowledgeUtils.generate_entity_id(
                relation_data["subject"], "Entity"
            )
            object_id = KnowledgeUtils.generate_entity_id(
                relation_data["object"], "Entity"
            )

            # Ensure all property values are primitive types
            properties = {}
            properties["predicate"] = str(relation_data.get("predicate", ""))
            properties["source_span"] = str(relation_data.get("source_span", ""))

            return {
                "source_id": subject_id,
                "target_id": object_id,
                "type": relation_data.get("predicate", "RELATES"),
                "confidence": float(relation_data.get("confidence", 0.7)),
                "source": source,
                "properties": properties,
            }
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            logger.warning(
                "Failed to create relationship between '%s' and '%s': %s",
                relation_data.get("subject", "UNKNOWN"),
                relation_data.get("object", "UNKNOWN"),
                e,
            )
            return None

    async def _create_fact_from_proposition(
        self, proposition: dict[str, Any], source: str
    ) -> dict[str, Any] | None:
        """Convert semantic proposition to knowledge graph fact."""
        try:
            # Ensure entities is a list of strings (entity IDs)
            entities = proposition.get("entities", [])
            if not isinstance(entities, list):
                entities = []
            entities = [str(e) for e in entities]  # Convert to strings

            # Ensure metadata contains only primitive types
            metadata = {"extraction_method": "semantic_analysis", "entities": entities}

            return {
                "id": proposition.get("id", f"fact_{datetime.now().timestamp()}"),
                "content": str(proposition["content"]),
                "logical_form": str(proposition.get("logical_form", "")),
                "confidence": float(proposition.get("confidence", 0.9)),
                "source": source,
                "metadata": metadata,
                "entities": entities,
            }
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            logger.warning(
                "Failed to create fact from proposition '%s': %s",
                proposition.get("id", "UNKNOWN"),
                e,
            )
            return None

    @logger.timer()
    async def _post_process_data(self, processed_data: dict[str, Any]) -> None:
        """Post-process extracted data for validation and cleanup."""
        # Deduplicate entities
        unique_entities: dict[str, dict[str, Any]] = {}
        for entity in processed_data["entities"]:
            if entity["id"] not in unique_entities:
                unique_entities[entity["id"]] = entity
        processed_data["entities"] = list(unique_entities.values())

        # Validate relationships have valid entity references
        valid_entity_ids = {entity["id"] for entity in processed_data["entities"]}
        valid_relationships = []

        for rel in processed_data["relationships"]:
            if (
                rel["source_id"] in valid_entity_ids
                and rel["target_id"] in valid_entity_ids
            ):
                valid_relationships.append(rel)
            else:
                logger.debug("Removing invalid relationship: %s", rel)

        processed_data["relationships"] = valid_relationships

        # Create basic facts from sentences if no semantic facts exist
        if not processed_data["facts"] and processed_data["sentences"]:
            for i, sentence in enumerate(processed_data["sentences"]):
                if len(sentence.strip()) > 10:
                    # Simplified fact structure with completely flattened metadata
                    fact = {
                        "id": f"{processed_data['text_id']}_fact_{i}",
                        "content": sentence,
                        "logical_form": "",
                        "confidence": 0.8,
                        "source": processed_data["source"],
                        "metadata": {
                            "extraction_method": "basic_sentence_split",
                            "entities": "",  # Empty string for entity list
                        },
                        "entities": [],  # Empty list for entity connections
                    }
                    processed_data["facts"].append(fact)

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.

        Removes academic noise like LaTeX commands, HTML artifacts, and CID strings
        that pollute the entity space during ingestion.
        """
        # 1. Remove HTML tags but keep content (basic stripping)
        text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)

        # 2. Remove CID strings (PDF artifact)
        text = re.sub(r"\(CID:\d+\)", "", text)

        # 3. Remove LaTeX math environments
        text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
        text = re.sub(r"\$[^$]+\$", " ", text)

        # 4. Remove metadata-only LaTeX commands
        text = re.sub(
            r"\\(cite|ref|label|url|href|bibliographystyle|bibliography)\{.*?\}",
            " ",
            text,
        )
        text = re.sub(
            r"\\(cite|ref|label|url|href|bibliographystyle|bibliography)", " ", text
        )

        # 5. Keep content of formatting LaTeX commands
        text = re.sub(
            r"\\(textbf|textit|emph|section|subsection|subsubsection)\{(.*?)\}",
            r"\2",
            text,
        )

        # 6. Remove remaining generic LaTeX commands
        text = re.sub(r"\\[a-zA-Z]+\{.*?\}", " ", text)
        text = re.sub(r"\\[a-zA-Z]+", " ", text)

        # 7. Remove numeric citations like [1], [1, 2], [1-3]
        text = re.sub(r"\[\d+(?:,\s*\d+|-\d+)*\]", " ", text)

        # 8. Remove parenthetical citations like (Author, 2020)
        text = re.sub(r"\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)", " ", text)

        # 9. Normalize whitespace and problematic characters
        text = re.sub(r"\s+", " ", text)

        # Keep alphanumeric, spaces, and basic punctuation
        allowed_chars = (
            string.ascii_letters
            + string.digits
            + string.whitespace
            + ".,!?;:-()[]\"'/&"
        )
        text = "".join(char for char in text if char in allowed_chars)

        # Normalize quotes (using Unicode escape sequences)
        text = re.sub(r"[\u201C\u201D]", '"', text)  # Smart double quotes
        text = re.sub(r"[\u2018\u2019]", "'", text)  # Smart single quotes

        # 10. Strip and ensure proper ending
        text = text.strip()
        # Remove leading/trailing punctuation that might result from stripping noise
        text = re.sub(r"^[.,!?;:-]+", "", text).strip()

        if text and text[-1] not in ".!?":
            text += "."

        return text

    async def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using improved rules."""
        # Better sentence splitting that handles common abbreviations
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5:  # Skip very short fragments
                if sentence and sentence[-1] not in ".!?":
                    sentence += "."
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _generate_text_id(self, text: str, source: str) -> str:
        """Generate unique identifier for text."""
        # Create hash of text + source for uniqueness
        combined = f"{text}{source}".encode()
        content_hash = hashlib.md5(combined, usedforsecurity=False).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"text_{timestamp}_{content_hash}"

    async def get_processing_statistics(self) -> dict[str, Any]:
        """Get statistics about semantic processing performance."""
        return {
            "montague_parser_available": self.montague_parser is not None,
            "entities_cached": len(self.entity_cache),
            "status": "active",
        }
