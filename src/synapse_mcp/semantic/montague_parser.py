"""
Montague Grammar-based semantic parser for Project Synapse.

This module implements the Semantic Blueprint component, providing
formal semantic analysis and logical form generation for natural language.
"""

import asyncio
import re
from typing import Dict, List, Optional, Tuple, Any
import spacy
from spacy import Language
import logging

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class MontagueParser:
    """
    Montague Grammar-based semantic parser.

    Implements formal semantic analysis using compositional semantics
    and lambda calculus for precise meaning representation.
    """

    def __init__(self):
        self.nlp: Optional[Language] = None
        self.entity_types = {
            'PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT',
            'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE'
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
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Downloaded and loaded spaCy model")

        except Exception as e:
            logger.error(f"Failed to initialize Montague parser: {e}")
            raise

    async def parse_text(self, text: str) -> Dict[str, Any]:
        """
        Parse text using Montague Grammar principles.

        Args:
            text: Input text to parse

        Returns:
            Dictionary containing semantic analysis results
        """
        if not self.nlp:
            raise ValueError("Parser not initialized")

        logger.debug(f"Parsing text: {text[:100]}...")

        # Process with spaCy
        doc = self.nlp(text)

        # Extract semantic components
        analysis = {
            'original_text': text,
            'entities': await self._extract_entities(doc),
            'relations': await self._extract_relations(doc),
            'logical_form': await self._generate_logical_form(doc),
            'semantic_features': await self._extract_semantic_features(doc),
            'propositions': await self._extract_propositions(doc)
        }

        logger.debug(f"Extracted {len(analysis['entities'])} entities, "
                    f"{len(analysis['relations'])} relations")

        return analysis

    async def _extract_entities(self, doc) -> List[Dict]:
        """Extract named entities with type and confidence information."""
        entities = []

        for ent in doc.ents:
            # Calculate confidence based on entity characteristics
            confidence = self._calculate_entity_confidence(ent)

            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'type': self._normalize_entity_type(ent.label_),
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': confidence,
                'id': self._generate_entity_id(ent.text, ent.label_)
            })

        return entities

    async def _extract_relations(self, doc) -> List[Dict]:
        """Extract semantic relations between entities."""
        relations = []

        # Simple dependency-based relation extraction
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                # Find subject-verb-object patterns
                head = token.head

                # Look for related entities
                subject = self._find_entity_for_token(doc, token)
                predicate = head.lemma_ if head.pos_ == 'VERB' else head.text

                # Find object if it exists
                obj = None
                for child in head.children:
                    if child.dep_ in ['dobj', 'pobj'] and child != token:
                        obj = self._find_entity_for_token(doc, child)
                        break

                if subject and obj:
                    relations.append({
                        'subject': subject,
                        'predicate': predicate,
                        'object': obj,
                        'confidence': 0.8,  # Default confidence
                        'source_span': f"{token.i}-{head.i}"
                    })

        return relations

    async def _generate_logical_form(self, doc) -> str:
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

    async def _extract_semantic_features(self, doc) -> Dict[str, Any]:
        """Extract semantic features from the parsed document."""
        features = {
            'sentence_count': len(list(doc.sents)),
            'entity_count': len(doc.ents),
            'token_count': len(doc),
            'has_negation': any(token.dep_ == 'neg' for token in doc),
            'tense': self._extract_tense(doc),
            'modality': self._extract_modality(doc),
            'sentiment_polarity': 'neutral'  # Placeholder
        }

        return features

    async def _extract_propositions(self, doc) -> List[Dict]:
        """Extract atomic propositions that can be stored as facts."""
        propositions = []

        for sent in doc.sents:
            # Each sentence potentially represents a proposition
            entities_in_sent = [ent for ent in doc.ents if ent.start >= sent.start and ent.end <= sent.end]

            if entities_in_sent:
                proposition = {
                    'id': f"prop_{sent.start}_{sent.end}",
                    'content': sent.text.strip(),
                    'entities': [self._generate_entity_id(ent.text, ent.label_) for ent in entities_in_sent],
                    'confidence': 0.9,  # High confidence for direct extraction
                    'logical_form': await self._generate_logical_form_for_sentence(sent)
                }
                propositions.append(proposition)

        return propositions

    def _calculate_entity_confidence(self, ent) -> float:
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
            'PERSON': 'Person',
            'ORG': 'Organization',
            'GPE': 'Location',
            'LOC': 'Location',
            'PRODUCT': 'Product',
            'EVENT': 'Event',
            'WORK_OF_ART': 'CreativeWork',
            'LAW': 'Law',
            'LANGUAGE': 'Language',
            'DATE': 'TemporalEntity',
            'TIME': 'TemporalEntity',
            'MONEY': 'MonetaryValue',
            'QUANTITY': 'Quantity'
        }

        return mapping.get(spacy_label, 'Entity')

    def _generate_entity_id(self, text: str, label: str) -> str:
        """Generate unique identifier for an entity."""
        normalized = re.sub(r'[^a-zA-Z0-9]', '_', text.lower())
        return f"{label.lower()}_{normalized}"

    def _find_entity_for_token(self, doc, token) -> Optional[str]:
        """Find if a token is part of a named entity."""
        for ent in doc.ents:
            if token.i >= ent.start and token.i < ent.end:
                return ent.text
        return token.text

    def _extract_svo_pattern(self, sent) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Extract Subject-Verb-Object pattern from a sentence."""
        subject = None
        verb = None
        obj = None

        for token in sent:
            if token.dep_ == 'nsubj':
                subject = token
            elif token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                verb = token
            elif token.dep_ in ['dobj', 'pobj']:
                obj = token

        return subject, verb, obj

    def _extract_tense(self, doc) -> str:
        """Extract tense information from the document."""
        for token in doc:
            if token.pos_ == 'VERB' and token.tag_:
                if 'VBD' in token.tag_ or 'VBN' in token.tag_:
                    return 'past'
                elif 'VBG' in token.tag_:
                    return 'present_continuous'
                elif 'VBZ' in token.tag_ or 'VBP' in token.tag_:
                    return 'present'
        return 'unknown'

    def _extract_modality(self, doc) -> List[str]:
        """Extract modal verbs and expressions."""
        modals = []
        modal_verbs = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'}

        for token in doc:
            if token.lemma_.lower() in modal_verbs:
                modals.append(token.lemma_.lower())

        return modals

    async def _generate_logical_form_for_sentence(self, sent) -> str:
        """Generate logical form for a specific sentence."""
        # Simplified logical form generation
        subject, verb, obj = self._extract_svo_pattern(sent)

        if subject and verb:
            if obj:
                return f"∃x∃y({subject.lemma_}(x) ∧ {obj.lemma_}(y) ∧ {verb.lemma_}(x,y))"
            else:
                return f"∃x({subject.lemma_}(x) ∧ {verb.lemma_}(x))"

        return f"proposition({sent.text[:50]}...)"
