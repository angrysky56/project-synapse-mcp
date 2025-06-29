"""
Knowledge representation and management utilities for Project Synapse.

This module provides utilities for working with knowledge structures,
semantic relationships, and graph operations.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    type: str
    confidence: float = 1.0
    properties: dict[str, Any] | None = None
    source: str = "unknown"
    created_at: datetime | None = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source_id: str
    target_id: str
    type: str
    confidence: float = 1.0
    properties: dict[str, Any] | None = None
    source: str = "unknown"
    created_at: datetime | None = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Fact:
    """Represents a semantic fact extracted from text."""
    id: str
    content: str
    logical_form: str = ""
    confidence: float = 1.0
    source: str = "unknown"
    metadata: dict[str, Any] | None = None
    entities: list[str] | None = None
    created_at: datetime | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.entities is None:
            self.entities = []
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Insight:
    """Represents an AI-generated insight (Zettel)."""
    zettel_id: str
    title: str
    content: str
    topic: str = ""
    confidence: float = 0.0
    pattern_type: str = "unknown"
    evidence: list[dict] | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()


class KnowledgeUtils:
    """Utility functions for knowledge management."""

    @staticmethod
    def generate_entity_id(name: str, entity_type: str) -> str:
        """Generate a unique entity ID."""
        normalized_name = name.lower().replace(' ', '_')
        content = f"{entity_type.lower()}_{normalized_name}"
        return content[:50]  # Limit length

    @staticmethod
    def generate_fact_id(content: str, source: str) -> str:
        """Generate a unique fact ID."""
        content_hash = hashlib.md5(f"{content}{source}".encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"fact_{timestamp}_{content_hash}"

    @staticmethod
    def generate_zettel_id(pattern_type: str) -> str:
        """Generate a unique Zettel ID for insights."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(f"{pattern_type}{timestamp}".encode()).hexdigest()[:8]
        return f"insight_{timestamp}_{pattern_type}_{content_hash}"

    @staticmethod
    def calculate_confidence(evidence_count: int, base_confidence: float = 0.5) -> float:
        """Calculate confidence score based on evidence."""
        # Simple confidence calculation - can be enhanced
        confidence = base_confidence + (evidence_count * 0.1)
        return min(confidence, 1.0)

    @staticmethod
    def validate_entity(entity: Entity) -> bool:
        """Validate entity data structure."""
        if not entity.id or not entity.name:
            return False
        if entity.confidence < 0 or entity.confidence > 1:
            return False
        return True

    @staticmethod
    def validate_relationship(relationship: Relationship) -> bool:
        """Validate relationship data structure."""
        if not relationship.source_id or not relationship.target_id:
            return False
        if not relationship.type:
            return False
        if relationship.confidence < 0 or relationship.confidence > 1:
            return False
        return True

    @staticmethod
    def validate_fact(fact: Fact) -> bool:
        """Validate fact data structure."""
        if not fact.id or not fact.content:
            return False
        if fact.confidence < 0 or fact.confidence > 1:
            return False
        return True

    @staticmethod
    def normalize_relationship_type(rel_type: str) -> str:
        """Normalize relationship type to standard format."""
        return rel_type.upper().replace(' ', '_')

    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 5) -> list[str]:
        """Extract keywords from text for topic classification."""
        # Simple keyword extraction - can be enhanced with NLP
        import re

        # Remove punctuation and convert to lowercase
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = clean_text.split()

        # Filter out common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }

        keywords = [word for word in words if word not in stop_words and len(word) > 3]

        # Return unique keywords, limited to max_keywords
        return list(dict.fromkeys(keywords))[:max_keywords]


class KnowledgeValidator:
    """Validator for knowledge graph data integrity."""

    @staticmethod
    def validate_graph_consistency(entities: list[Entity], relationships: list[Relationship]) -> dict[str, list[str]]:
        """Validate consistency between entities and relationships."""
        issues = {
            'missing_entities': [],
            'orphaned_relationships': [],
            'duplicate_entities': [],
            'invalid_confidences': []
        }

        entity_ids = {entity.id for entity in entities}
        entity_names = {}

        # Check for duplicate entities
        for entity in entities:
            if entity.name in entity_names:
                issues['duplicate_entities'].append(f"Duplicate entity name: {entity.name}")
            entity_names[entity.name] = entity.id

            # Check confidence values
            if entity.confidence < 0 or entity.confidence > 1:
                issues['invalid_confidences'].append(f"Invalid confidence for entity {entity.id}: {entity.confidence}")

        # Check relationships reference valid entities
        for relationship in relationships:
            if relationship.source_id not in entity_ids:
                issues['missing_entities'].append(f"Relationship references missing source entity: {relationship.source_id}")
            if relationship.target_id not in entity_ids:
                issues['missing_entities'].append(f"Relationship references missing target entity: {relationship.target_id}")

            # Check confidence values
            if relationship.confidence < 0 or relationship.confidence > 1:
                issues['invalid_confidences'].append(f"Invalid confidence for relationship {relationship.source_id}->{relationship.target_id}: {relationship.confidence}")

        return issues

    @staticmethod
    def suggest_entity_merges(entities: list[Entity], similarity_threshold: float = 0.8) -> list[tuple[str, str]]:
        """Suggest entities that might be duplicates and should be merged."""
        suggestions = []

        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Simple similarity check based on name
                similarity = KnowledgeValidator._calculate_name_similarity(entity1.name, entity2.name)
                if similarity >= similarity_threshold:
                    suggestions.append((entity1.id, entity2.id))

        return suggestions

    @staticmethod
    def _calculate_name_similarity(name1: str, name2: str) -> float:
        """Calculate similarity between two entity names."""
        # Simple Jaccard similarity of words
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())

        if not words1 and not words2:
            return 1.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0


# Export main classes and functions
__all__ = [
    'Entity', 'Relationship', 'Fact', 'Insight',
    'KnowledgeUtils', 'KnowledgeValidator'
]
