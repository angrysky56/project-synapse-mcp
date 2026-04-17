"""
Montague Grammar-based semantic parser for Project Synapse.

This module implements the Semantic Blueprint component, providing
formal semantic analysis and logical form generation for natural language.
"""

import re

# trunk-ignore(bandit/B404)
import subprocess
import sys
from typing import Any

import spacy  # type: ignore[import-untyped]
from spacy.language import Language  # type: ignore[import-untyped]
from spacy.tokens import Doc, Span, Token  # type: ignore[import-untyped]

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class MontagueParser:
    """
    Montague Grammar-based semantic parser.

    Implements formal semantic analysis using compositional semantics
    and lambda calculus for precise meaning representation.
    """

    def __init__(self) -> None:
        self.nlp: Language | None = None
        self.entity_types = {
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "PRODUCT",
            "EVENT",
            "WORK_OF_ART",
            "LAW",
            "LANGUAGE",
        }

    async def initialize(self) -> None:
        """Initialize the semantic parser with spaCy models."""
        try:
            # Load spaCy model for English
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Montague parser initialized successfully")

        except OSError:
            logger.warning("spaCy model not found, downloading...")
            # In production, this should be handled during setup
            # trunk-ignore(bandit/B603)
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True
            )
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Downloaded and loaded spaCy model")

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to initialize Montague parser: %s", e)
            raise

    async def parse_text(self, text: str) -> dict[str, Any]:
        """
        Parse text using Montague Grammar principles.

        Args:
            text: Input text to parse

        Returns:
            Dictionary containing semantic analysis results
        """
        if not self.nlp:
            raise ValueError("Parser not initialized")

        logger.debug("Parsing text: %s...", text[:100])

        # Process with spaCy
        doc = self.nlp(text)

        # Extract semantic components
        analysis = {
            "original_text": text,
            "entities": await self._extract_entities(doc),
            "relations": await self._extract_relations(doc),
            "logical_form": await self._generate_logical_form(doc),
            "semantic_features": await self._extract_semantic_features(doc),
            "propositions": await self._extract_propositions(doc),
        }

        logger.debug(
            "Extracted %d entities, %d relations",
            len(analysis["entities"]),
            len(analysis["relations"]),
        )

        return analysis

    async def _extract_entities(self, doc: Doc) -> list[dict[str, Any]]:
        """Extract named entities with type and confidence information."""
        entities = []

        for ent in doc.ents:
            # Calculate confidence based on entity characteristics
            confidence = self._calculate_entity_confidence(ent)

            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "type": self._normalize_entity_type(ent.label_),
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": confidence,
                    "id": self._generate_entity_id(ent.text, ent.label_),
                }
            )

        return entities

    async def _extract_relations(self, doc: Doc) -> list[dict[str, Any]]:
        """Extract semantic relations between entities and key noun phrases."""
        relations = []

        # Extract noun chunks for richer relationship detection
        noun_chunks = {chunk.root.i: chunk.text for chunk in doc.noun_chunks}

        # Strategy 1: Subject-Verb-Object patterns
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                head = token.head

                # Get subject (prefer entity, fall back to noun chunk)
                subject = self._find_entity_for_token(doc, token)
                if not subject and token.i in noun_chunks:
                    subject = noun_chunks[token.i]

                if not subject:
                    continue

                # Get predicate
                predicate = head.lemma_ if head.pos_ == "VERB" else head.text

                # Find object
                obj = None
                for child in head.children:
                    if child.dep_ in ["dobj", "attr", "pobj"]:
                        obj = self._find_entity_for_token(doc, child)
                        if not obj and child.i in noun_chunks:
                            obj = noun_chunks[child.i]
                        if obj:
                            break

                if subject and obj:
                    relations.append(
                        {
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                            "confidence": 0.8,
                            "source_span": f"{token.i}-{head.i}",
                        }
                    )

        # Strategy 2: Copula/Linking verb patterns (is, are, becomes, equals)
        for token in doc:
            if token.lemma_ in ["be", "become", "equal", "represent", "constitute"]:
                # Find subject
                cop_subject: str | None = None
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        cop_subject = self._find_entity_for_token(doc, child)
                        if not cop_subject and child.i in noun_chunks:
                            cop_subject = str(noun_chunks[child.i])
                        break

                # Find complement/attribute
                cop_obj: str | None = None
                for child in token.children:
                    if child.dep_ in ["attr", "acomp", "dobj"]:
                        cop_obj = self._find_entity_for_token(doc, child)
                        if not cop_obj and child.i in noun_chunks:
                            cop_obj = str(noun_chunks[child.i])
                        break

                if cop_subject and cop_obj:
                    relations.append(
                        {
                            "subject": cop_subject,
                            "predicate": f"is-{token.lemma_}",
                            "object": cop_obj,
                            "confidence": 0.75,
                            "source_span": f"{token.i}",
                        }
                    )

        # Strategy 3: Prepositional relationships
        for token in doc:
            if token.dep_ == "prep" and token.head.pos_ in ["NOUN", "PROPN", "VERB"]:
                # Get the noun being modified
                subject = self._find_entity_for_token(doc, token.head)
                if not subject and token.head.i in noun_chunks:
                    subject = noun_chunks[token.head.i]

                # Get the object of the preposition
                obj = None
                for child in token.children:
                    if child.dep_ == "pobj":
                        obj = self._find_entity_for_token(doc, child)
                        if not obj and child.i in noun_chunks:
                            obj = noun_chunks[child.i]
                        break

                if subject and obj:
                    relations.append(
                        {
                            "subject": subject,
                            "predicate": f"relates-via-{token.text}",
                            "object": obj,
                            "confidence": 0.7,
                            "source_span": f"{token.head.i}-{token.i}",
                        }
                    )

        # Strategy 4: Compound and possessive patterns
        for token in doc:
            if token.dep_ in ["compound", "poss"] and token.head.pos_ in [
                "NOUN",
                "PROPN",
            ]:
                subject = token.text
                obj = token.head.text

                if len(subject) > 2 and len(obj) > 2:  # Filter very short words
                    relation_type = (
                        "modifies" if token.dep_ == "compound" else "possesses"
                    )
                    relations.append(
                        {
                            "subject": subject,
                            "predicate": relation_type,
                            "object": obj,
                            "confidence": 0.65,
                            "source_span": f"{token.i}-{token.head.i}",
                        }
                    )

        return relations

    async def _generate_logical_form(self, doc: Doc) -> str:
        """
        Generate logical form representation using lambda calculus.

        This is a simplified implementation of Montague Grammar principles.
        """
        logical_forms = []

        for sent in doc.sents:
            # Extract the main components
            subject, verb, obj = self._extract_svo_pattern(sent)

            if subject and verb:
                if obj:
                    # Binary predicate: R(x,y)
                    form = f"{verb.lemma_}({subject.text}, {obj.text})"
                else:
                    # Unary predicate: P(x)
                    form = f"{verb.lemma_}({subject.text})"

                logical_forms.append(form)

        return " ∧ ".join(logical_forms) if logical_forms else ""

    async def _extract_semantic_features(self, doc: Doc) -> dict[str, Any]:
        """Extract semantic features from the parsed document."""
        features = {
            "sentence_count": len(list(doc.sents)),
            "entity_count": len(doc.ents),
            "token_count": len(doc),
            "has_negation": any(token.dep_ == "neg" for token in doc),
            "tense": self._extract_tense(doc),
            "modality": self._extract_modality(doc),
            "sentiment_polarity": "neutral",  # Placeholder
        }

        return features

    async def _extract_propositions(self, doc: Doc) -> list[dict[str, Any]]:
        """Extract atomic propositions that can be stored as facts."""
        propositions = []

        for sent in doc.sents:
            # Each sentence potentially represents a proposition
            entities_in_sent = [
                ent
                for ent in doc.ents
                if ent.start >= sent.start and ent.end <= sent.end
            ]

            if entities_in_sent:
                proposition = {
                    "id": f"prop_{sent.start}_{sent.end}",
                    "content": sent.text.strip(),
                    "entities": [
                        self._generate_entity_id(ent.text, ent.label_)
                        for ent in entities_in_sent
                    ],
                    "confidence": 0.9,  # High confidence for direct extraction
                    "logical_form": await self._generate_logical_form_for_sentence(
                        sent
                    ),
                }
                propositions.append(proposition)

        return propositions

    def _calculate_entity_confidence(self, ent: Span) -> float:
        """Calculate confidence score for an entity."""
        base_confidence = 0.8

        # Boost confidence for known entity types
        if ent.label_ in self.entity_types:
            base_confidence += 0.1

        # Consider entity length (longer entities often more reliable)
        if len(ent.text) > 10:
            base_confidence += 0.05

        # Check if entity is capitalized (proper nouns)
        if ent.text[0].isupper():
            base_confidence += 0.05

        return min(base_confidence, 1.0)

    def _normalize_entity_type(self, spacy_label: str) -> str:
        """Normalize spaCy entity labels to our schema."""
        mapping = {
            "PERSON": "Person",
            "ORG": "Organization",
            "GPE": "Location",
            "LOC": "Location",
            "PRODUCT": "Product",
            "EVENT": "Event",
            "WORK_OF_ART": "CreativeWork",
            "LAW": "Law",
            "LANGUAGE": "Language",
            "DATE": "TemporalEntity",
            "TIME": "TemporalEntity",
            "MONEY": "MonetaryValue",
            "QUANTITY": "Quantity",
        }

        return mapping.get(spacy_label, "Entity")

    def _generate_entity_id(self, text: str, label: str) -> str:
        """Generate unique identifier for an entity."""
        normalized = re.sub(r"[^a-zA-Z0-9]", "_", text.lower())
        return f"{label.lower()}_{normalized}"

    def _find_entity_for_token(self, doc: Doc, token: Token) -> str:
        """Find if a token is part of a named entity."""
        for ent in doc.ents:
            if token.i >= ent.start and token.i < ent.end:
                return str(ent.text)
        return str(token.text)

    def _extract_svo_pattern(
        self, sent: Span
    ) -> tuple[Token | None, Token | None, Token | None]:
        """Extract Subject-Verb-Object pattern from a sentence."""
        subject = None
        verb = None
        obj = None

        for token in sent:
            if token.dep_ == "nsubj":
                subject = token
            elif token.pos_ == "VERB" and token.dep_ == "ROOT":
                verb = token
            elif token.dep_ in ["dobj", "pobj"]:
                obj = token

        return subject, verb, obj

    def _extract_tense(self, doc: Doc) -> str:
        """Extract tense information from the document."""
        for token in doc:
            if token.pos_ == "VERB" and token.tag_:
                if "VBD" in token.tag_ or "VBN" in token.tag_:
                    return "past"
                if "VBG" in token.tag_:
                    return "present_continuous"
                if "VBZ" in token.tag_ or "VBP" in token.tag_:
                    return "present"
        return "unknown"

    def _extract_modality(self, doc: Doc) -> list[str]:
        """Extract modal verbs and expressions."""
        modals = []
        modal_verbs = {
            "can",
            "could",
            "may",
            "might",
            "must",
            "shall",
            "should",
            "will",
            "would",
        }

        for token in doc:
            if token.lemma_.lower() in modal_verbs:
                modals.append(token.lemma_.lower())

        return modals

    async def _generate_logical_form_for_sentence(self, sent: Span) -> str:
        """Generate logical form for a specific sentence."""
        # Simplified logical form generation
        subject, verb, obj = self._extract_svo_pattern(sent)

        if subject and verb:
            if obj:
                return (
                    f"∃x∃y({subject.lemma_}(x) ∧ {obj.lemma_}(y) ∧ "
                    f"{verb.lemma_}(x,y))"
                )
            return f"∃x({subject.lemma_}(x) ∧ {verb.lemma_}(x))"

        return f"proposition({sent.text[:50]}...)"
