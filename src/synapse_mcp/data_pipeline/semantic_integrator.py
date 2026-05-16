"""
Semantic Integration Pipeline for Project Synapse.

This module bridges the text processor and Montague parser to create
a complete semantic analysis pipeline.
"""

import hashlib
import os
import re
import string
from datetime import datetime
from typing import Any

from ..knowledge.knowledge_types import KnowledgeUtils
from ..semantic.llm_extractor import LlmExtractor
from ..semantic.montague_parser import MontagueParser
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class SemanticIntegrator:
    """
    Integrates text processing with semantic analysis to extract knowledge structures.

    This class coordinates between the text processor and Montague parser to create
    a unified semantic analysis pipeline that extracts entities, relationships, and facts.
    """

    def __init__(
        self,
        montague_parser: MontagueParser | None = None,
        llm_extractor: LlmExtractor | None = None,
    ) -> None:
        """Initialize the semantic integrator."""
        self.montague_parser = montague_parser
        self.llm_extractor = llm_extractor
        self.extraction_provider = "montague"
        self.entity_cache: dict[str, dict[str, Any]] = {}  # Cache for deduplication
        self.logger = logger

    @logger.timer()
    async def initialize(self) -> None:
        """Initialize the semantic integrator with required components."""
        self.extraction_provider = os.getenv("EXTRACTION_PROVIDER", "montague").lower()

        if self.montague_parser is None:
            self.montague_parser = MontagueParser()

        if self.llm_extractor is None and self.extraction_provider == "llm":
            self.llm_extractor = LlmExtractor()

        try:
            await self.montague_parser.initialize()
            logger.info(
                "Semantic integrator initialized with provider: %s",
                self.extraction_provider,
            )
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
                semantic_analysis = None

                # 1. Try LLM Provider if selected
                if self.extraction_provider == "llm" and self.llm_extractor:
                    try:
                        logger.info("Using LLM extractor for semantic analysis")
                        llm_result = await self.llm_extractor.extract_semantics(
                            processed_data["cleaned_text"], source_name=source
                        )
                        semantic_analysis = {
                            "entities": [e.model_dump() for e in llm_result.entities],
                            "relations": [r.model_dump() for r in llm_result.relations],
                            "propositions": [],
                        }

                        # Hybrid Merge: Still run Montague for logical forms and spaCy NER
                        try:
                            montague_analysis = await self.montague_parser.parse_text(
                                processed_data["cleaned_text"]
                            )
                            semantic_analysis["propositions"] = montague_analysis.get(
                                "propositions", []
                            )
                            semantic_analysis["semantic_features"] = (
                                montague_analysis.get("semantic_features", {})
                            )
                            # Merge spaCy entities with LLM results
                            semantic_analysis["entities"] = (
                                await self._hybrid_entity_merge(
                                    semantic_analysis["entities"],
                                    montague_analysis.get("entities", []),
                                )
                            )
                            # Relation fallback: if the LLM produced no relations
                            # (output truncated by num_predict, model behavior,
                            # silent parse failure...), supplement with Montague's
                            # relations rather than losing them entirely. This
                            # keeps the graph connected even when Gemma punts on
                            # the relations field — quality drops to "decent"
                            # Montague output instead of "empty".
                            if not semantic_analysis["relations"]:
                                montague_relations = montague_analysis.get(
                                    "relations", []
                                )
                                if montague_relations:
                                    logger.info(
                                        "LLM returned 0 relations for '%s' "
                                        "(entities=%d); falling back to %d "
                                        "Montague relations.",
                                        source,
                                        len(semantic_analysis["entities"]),
                                        len(montague_relations),
                                    )
                                    semantic_analysis["relations"] = montague_relations
                        except Exception as me:
                            logger.warning("Hybrid Montague pass failed: %s", me)

                    except Exception as le:
                        logger.warning(
                            "LLM extraction failed, falling back to pure Montague: %s",
                            le,
                        )
                        semantic_analysis = None  # Trigger fallback below

                # 2. Fallback to (or primary use of) Montague
                if semantic_analysis is None:
                    semantic_analysis = await self.montague_parser.parse_text(
                        processed_data["cleaned_text"]
                    )

                if semantic_analysis:
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
            subj_text = relation_data.get("subject", "").strip()
            obj_text = relation_data.get("object", "").strip()

            if not subj_text or not obj_text:
                continue

            # Auto-create entity nodes for relation endpoints not in entity cache
            for endpoint_name in [subj_text, obj_text]:
                # Attempt to find the entity in processed_data["entities"]
                endpoint_id = None
                for ent in processed_data["entities"]:
                    if ent["name"].lower() == endpoint_name.lower():
                        endpoint_id = ent["id"]
                        break

                # Also check cache
                if not endpoint_id:
                    for ent in self.entity_cache.values():
                        if ent["name"].lower() == endpoint_name.lower():
                            endpoint_id = ent["id"]
                            break

                if not endpoint_id:
                    # This endpoint was named in a relation but never extracted
                    # as a typed entity. Use a distinct ``UnresolvedReference``
                    # type rather than the generic ``Concept`` so these stubs
                    # are queryable and obviously distinguishable from genuine
                    # concept nodes — and so we can later promote them to a
                    # real type when we see better evidence for what they are.
                    endpoint_id = KnowledgeUtils.generate_entity_id(
                        endpoint_name, "UnresolvedReference"
                    )
                    # If a previous document's merge step redirected this ID,
                    # the cache will hold the canonical entity dict at this
                    # key. Reuse the canonical ID so the relation points at a
                    # real stored Neo4j node instead of a consumed-and-dropped
                    # ghost — closes the cross-document edge-leak path.
                    if endpoint_id in self.entity_cache:
                        cached = self.entity_cache[endpoint_id]
                        if cached.get("id") and cached["id"] != endpoint_id:
                            endpoint_id = cached["id"]
                    elif endpoint_id not in self.entity_cache:
                        entity = {
                            "id": endpoint_id,
                            "name": endpoint_name,
                            "type": "UnresolvedReference",
                            "confidence": 0.4,
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
                relation_data, source, processed_data
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
        self, relation_data: dict[str, Any], source: str, processed_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Convert semantic analysis relation to knowledge graph format."""
        try:
            subj_text = relation_data.get("subject", "")
            obj_text = relation_data.get("object", "")

            # Create entity IDs for subject and object
            subject_id = None
            object_id = None
            for ent in processed_data["entities"]:
                if ent["name"].lower() == subj_text.lower():
                    subject_id = ent["id"]
                if ent["name"].lower() == obj_text.lower():
                    object_id = ent["id"]

            if not subject_id:
                for ent in self.entity_cache.values():
                    if ent["name"].lower() == subj_text.lower():
                        subject_id = ent["id"]
                        break
            if not object_id:
                for ent in self.entity_cache.values():
                    if ent["name"].lower() == obj_text.lower():
                        object_id = ent["id"]
                        break

            if not subject_id:
                subject_id = KnowledgeUtils.generate_entity_id(subj_text, "Concept")
            if not object_id:
                object_id = KnowledgeUtils.generate_entity_id(obj_text, "Concept")

            # Ensure all property values are primitive types
            properties = {}
            properties["predicate"] = str(relation_data.get("predicate", ""))
            properties["source_span"] = str(relation_data.get("source_span", ""))

            # Normalize predicate to a small set of meaningful edge types.
            # Most verbs should become RELATES with the original predicate stored as a property.
            predicate_raw = str(relation_data.get("predicate", "RELATES"))
            pred_lower = predicate_raw.lower()
            if pred_lower in ("is-be", "is-become", "is", "are"):
                edge_type = "IS_A"
            elif pred_lower in ("possesses", "have", "has", "had"):
                edge_type = "POSSESSES"
            elif pred_lower.startswith("relates-via-"):
                edge_type = "RELATES"
            elif pred_lower in ("modifies", "compound"):
                edge_type = "MODIFIES"
            else:
                edge_type = "RELATES"
            properties["predicate"] = predicate_raw  # Preserve original for queries

            return {
                "source_id": subject_id,
                "target_id": object_id,
                "type": edge_type,
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

    @staticmethod
    def _find_mentioned_entity_ids(
        sentence: str, name_to_id_sorted: list[tuple[str, str]]
    ) -> list[str]:
        """Return entity IDs whose names appear (as case-insensitive substrings)
        in ``sentence``, with no overlapping matches.

        ``name_to_id_sorted`` must be sorted by name length DESCENDING so
        longer names match before any of their substrings.
        """
        sentence_lower = sentence.lower()
        found: list[str] = []
        seen_spans: list[tuple[int, int]] = []
        for ent_name_lower, ent_id in name_to_id_sorted:
            # Use word-boundary semantics by checking surrounding chars.
            # Pure substring matching would link a fact about "Java" to an
            # entity named "ja", which would explode false positives.
            start = 0
            while True:
                idx = sentence_lower.find(ent_name_lower, start)
                if idx < 0:
                    break
                end = idx + len(ent_name_lower)
                # Boundary check: char before must not be alphanumeric,
                # char after must not be alphanumeric. (Allows hyphens
                # and punctuation as boundaries.)
                before_ok = idx == 0 or not sentence_lower[idx - 1].isalnum()
                after_ok = (
                    end >= len(sentence_lower)
                    or not sentence_lower[end].isalnum()
                )
                if before_ok and after_ok:
                    # Skip overlap with a longer match already recorded.
                    span = (idx, end)
                    if not any(
                        not (span[1] <= s or e <= span[0]) for s, e in seen_spans
                    ):
                        seen_spans.append(span)
                        found.append(ent_id)
                        break  # one ID per name per sentence is enough
                start = idx + 1
        return found

    @logger.timer()
    async def _merge_entities_by_name(
        self, entities: list[dict]
    ) -> tuple[list[dict], dict[str, str]]:
        """Merge entities that share the same normalized name or where one is a substring of another.

        Fixes name fragmentation: e.g., 'Amirhossein' and 'Amirhossein Kazemnejad'
        get merged into the longer form (higher-confidence / longer name wins).

        Returns:
            (merged_entities, consumed_to_canonical_id_map)
            The map lets callers rewrite relation endpoints that pointed to the
            consumed (shorter) entity so they point to the surviving canonical
            entity. Without this rewrite, ``_store_relationship`` silently drops
            edges because its Cypher uses MATCH (not MERGE) on endpoint IDs.
        """
        if not entities:
            return entities, {}

        # Sort by name length descending so longer names are kept as canonical
        sorted_entities = sorted(entities, key=lambda e: len(e["name"]), reverse=True)
        merged: list[dict] = []
        consumed_ids: set[str] = set()
        # Maps consumed entity ID -> surviving canonical ID. Used by the caller
        # to rewrite relation endpoints so the edges don't get orphaned.
        consumed_to_canonical: dict[str, str] = {}

        for i, entity in enumerate(sorted_entities):
            if entity["id"] in consumed_ids:
                continue
            canonical = dict(entity)
            consumed_ids.add(entity["id"])

            # Check if any remaining entity name is contained within this one
            for j, other in enumerate(sorted_entities):
                if j <= i or other["id"] in consumed_ids:
                    continue
                # Substring containment (case-insensitive)
                if other["name"].lower() in canonical["name"].lower():
                    # Merge: keep the confidence of the more confident one
                    if other.get("confidence", 0.5) > canonical.get("confidence", 0.5):
                        canonical["confidence"] = other["confidence"]
                    # Prefer the longer name's type, but allow type upgrades
                    consumed_ids.add(other["id"])
                    consumed_to_canonical[other["id"]] = canonical["id"]

            merged.append(canonical)

        return merged, consumed_to_canonical

    async def _hybrid_entity_merge(
        self, llm_entities: list[dict[str, Any]], spacy_entities: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Merge entities from LLM and spaCy, preferring LLM but keeping unique spaCy ones.
        """
        # Start with LLM entities
        merged = list(llm_entities)
        llm_names = {e["text"].lower() for e in llm_entities}

        for se in spacy_entities:
            se_text = se["text"].lower()
            # If spaCy found something the LLM missed, add it
            if se_text not in llm_names:
                # Check if it's a substring or superstring of any LLM entity
                is_redundant = False
                for ln in llm_names:
                    if se_text in ln or ln in se_text:
                        is_redundant = True
                        break

                if not is_redundant:
                    merged.append(se)

        return merged

    async def _post_process_data(self, processed_data: dict[str, Any]) -> None:
        """Post-process extracted data for validation and cleanup."""
        # FIX Bug 2: Merge fragmented entities before deduplication
        if processed_data["entities"]:
            processed_data["entities"], id_remap = await self._merge_entities_by_name(
                processed_data["entities"]
            )

            # Rewrite relation endpoints that pointed to entities just consumed
            # by the merge step. Without this, ``_store_relationship`` would
            # silently drop the edge: its Cypher uses MATCH (not MERGE) on the
            # endpoint IDs, so a relation pointing to a no-longer-stored entity
            # produces no rows and no error — the edge just never gets written.
            if id_remap and processed_data.get("relationships"):
                remapped = 0
                for rel in processed_data["relationships"]:
                    new_src = id_remap.get(rel["source_id"])
                    if new_src and new_src != rel["source_id"]:
                        rel["source_id"] = new_src
                        remapped += 1
                    new_tgt = id_remap.get(rel["target_id"])
                    if new_tgt and new_tgt != rel["target_id"]:
                        rel["target_id"] = new_tgt
                        remapped += 1
                if remapped:
                    logger.debug(
                        "Remapped %d relation endpoints after entity merge",
                        remapped,
                    )

            # Same MATCH-not-MERGE silent-drop applies to Fact MENTIONS edges:
            # facts carry an ``entities`` list of entity IDs, and the storage
            # query MATCHes both Fact and Entity. If the entity was consumed
            # during merge, the MENTIONS edge silently disappears. Rewrite
            # the fact entity lists the same way.
            if id_remap and processed_data.get("facts"):
                for fact in processed_data["facts"]:
                    fact_ents = fact.get("entities") or []
                    if fact_ents:
                        fact["entities"] = [id_remap.get(eid, eid) for eid in fact_ents]
                    # ``metadata.entities`` is the duplicated copy that gets
                    # stored as a flat string property. Rewrite that too so
                    # the stored value reflects canonical IDs.
                    md = fact.get("metadata") or {}
                    md_ents = md.get("entities")
                    if isinstance(md_ents, list):
                        md["entities"] = [id_remap.get(eid, eid) for eid in md_ents]

            # Cache hygiene: redirect consumed IDs in the cross-document cache
            # to point at canonical entities. Future docs that look up a name
            # matching a consumed entity will then find the canonical ID, and
            # relations referencing the consumed ID via cache will still resolve
            # to a node that actually got stored in Neo4j.
            if id_remap:
                canonical_by_id = {e["id"]: e for e in processed_data["entities"]}
                for consumed_id, canonical_id in id_remap.items():
                    canonical_entity = canonical_by_id.get(canonical_id)
                    if canonical_entity is not None:
                        self.entity_cache[consumed_id] = canonical_entity

        # Deduplicate entities by ID
        unique_entities: dict[str, dict[str, Any]] = {}
        for entity in processed_data["entities"]:
            if entity["id"] not in unique_entities:
                unique_entities[entity["id"]] = entity
        processed_data["entities"] = list(unique_entities.values())

        # Validate relationships have valid entity references (including cached entities)
        valid_entity_ids = {entity["id"] for entity in processed_data["entities"]}
        valid_relationships = []

        for rel in processed_data["relationships"]:
            source_valid = (
                rel["source_id"] in valid_entity_ids
                or rel["source_id"] in self.entity_cache
            )
            target_valid = (
                rel["target_id"] in valid_entity_ids
                or rel["target_id"] in self.entity_cache
            )

            if source_valid and target_valid:
                valid_relationships.append(rel)
            else:
                logger.debug("Removing invalid relationship: %s", rel)

        processed_data["relationships"] = valid_relationships

        # Create basic facts from sentences if no semantic facts exist.
        # (MENTIONS linking for these facts is handled by the universal pass
        # below — every fact lacking entity backlinks gets substring-matched.)
        if not processed_data["facts"] and processed_data["sentences"]:
            for i, sentence in enumerate(processed_data["sentences"]):
                if len(sentence.strip()) > 10:
                    fact = {
                        "id": f"{processed_data['text_id']}_fact_{i}",
                        "content": sentence,
                        "logical_form": "",
                        "confidence": 0.8,
                        "source": processed_data["source"],
                        "metadata": {
                            "extraction_method": "basic_sentence_split",
                            "entities": [],
                        },
                        "entities": [],
                    }
                    processed_data["facts"].append(fact)

        # Universal MENTIONS backfill: any Fact that wasn't given an entities
        # list by its extractor (either path) gets one built by substring-
        # matching entity names against the fact text. This fixes the orphan
        # problem for types like Organization/Concept/TemporalEntity that
        # appear in narrative text but are rarely the subject/object of an
        # action verb the LLM extractor would emit a relation for.
        if processed_data["facts"] and processed_data["entities"]:
            name_to_id: list[tuple[str, str]] = sorted(
                (
                    (e["name"].lower(), e["id"])
                    for e in processed_data["entities"]
                    if len(e["name"]) >= 3
                ),
                key=lambda pair: len(pair[0]),
                reverse=True,
            )

            backfilled = 0
            for fact in processed_data["facts"]:
                if fact.get("entities"):
                    continue  # already linked by the extractor
                matched_ids = self._find_mentioned_entity_ids(
                    fact["content"], name_to_id
                )
                if matched_ids:
                    fact["entities"] = matched_ids
                    md = fact.get("metadata") or {}
                    md["entities"] = matched_ids
                    backfilled += 1
            if backfilled:
                logger.debug(
                    "Backfilled MENTIONS on %d facts via name substring match",
                    backfilled,
                )

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.

        Removes academic noise like LaTeX commands, HTML artifacts, and CID strings
        that pollute the entity space during ingestion.
        """
        # 0. Remove Markdown headers, bold, italics, links, code blocks, and INLINE CODE
        text = re.sub(
            r"^(#+)\s+\[", r"", text, flags=re.MULTILINE
        )  # heading+link combos
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)  # fenced code blocks
        text = re.sub(r"``.*?``", " ", text, flags=re.DOTALL)  # inline double-backtick
        text = re.sub(r"`[^`]+`", " ", text)  # inline single-backtick
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # bold
        text = re.sub(r"\*(.*?)\*", r"\1", text)  # italic
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # markdown links

        # 0b. Strip YAML frontmatter
        text = re.sub(r"^---\s*\n.*?\n---", " ", text, flags=re.MULTILINE | re.DOTALL)

        # 0c. Remove table rows and wiki-table syntax, replacing with a sentence
        # break so spaCy's sentence splitter doesn't merge a stripped table cell
        # into adjacent prose — that was producing spurious cross-sentence
        # relations like "seven different retailers --in--> the heart".
        text = re.sub(r"^\|.*\|$", ". ", text, flags=re.MULTILINE)  # table rows
        text = re.sub(r"^[=|-]{3,}$", ". ", text, flags=re.MULTILINE)  # separators

        # 0d. Markdown list markers — convert leading bullet/number to a
        # sentence break for the same reason as table rows above.
        text = re.sub(r"^\s*[-*+]\s+", ". ", text, flags=re.MULTILINE)  # bullets
        text = re.sub(r"^\s*\d+\.\s+", ". ", text, flags=re.MULTILINE)  # numbered

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
