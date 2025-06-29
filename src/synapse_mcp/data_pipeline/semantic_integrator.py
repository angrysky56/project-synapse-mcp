"""
Semantic Integration Pipeline for Project Synapse.

This module bridges the text processor and Montague parser to create
a complete semantic analysis pipeline.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..semantic.montague_parser import MontagueParser
from ..knowledge.knowledge_types import Entity, Relationship, Fact, KnowledgeUtils
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class SemanticIntegrator:
    """
    Integrates text processing with semantic analysis to extract complete knowledge structures.
    
    This class coordinates between the text processor and Montague parser to create
    a unified semantic analysis pipeline that extracts entities, relationships, and facts.
    """

    def __init__(self, montague_parser: Optional[MontagueParser] = None):
        self.montague_parser = montague_parser
        self.entity_cache = {}  # Cache for entity deduplication

    async def initialize(self) -> None:
        """Initialize the semantic integrator with required components."""
        if self.montague_parser is None:
            self.montague_parser = MontagueParser()
            
        try:
            await self.montague_parser.initialize()
            logger.info("Semantic integrator initialized successfully")
        except Exception as e:
            logger.warning(f"Montague parser initialization failed: {e}")
            logger.info("Continuing with basic text processing only")

    async def process_text_with_semantics(
        self,
        text: str,
        source: str = "user_input",
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process text through complete semantic analysis pipeline.

        Args:
            text: Raw text to process
            source: Source identifier for provenance
            metadata: Additional metadata about the text

        Returns:
            Dictionary containing entities, relationships, facts, and semantic analysis
        """
        logger.debug(f"Processing text with full semantic analysis: {text[:100]}...")

        # Start with basic text processing
        processed_data = await self._basic_text_processing(text, source, metadata)

        # Add semantic analysis if Montague parser is available
        if self.montague_parser and self.montague_parser.nlp:
            try:
                semantic_analysis = await self.montague_parser.parse_text(text)
                await self._integrate_semantic_analysis(processed_data, semantic_analysis, source)
            except Exception as e:
                logger.warning(f"Semantic analysis failed, continuing with basic processing: {e}")

        # Post-process and validate
        await self._post_process_data(processed_data)

        logger.info(f"Extracted {len(processed_data['entities'])} entities, "
                   f"{len(processed_data['relationships'])} relationships, "
                   f"{len(processed_data['facts'])} facts")

        return processed_data

    async def _basic_text_processing(
        self,
        text: str,
        source: str,
        metadata: Optional[Dict]
    ) -> Dict[str, Any]:
        """Perform basic text processing and structure preparation."""
        # Clean and normalize text
        cleaned_text = await self._clean_text(text)
        
        # Split into sentences
        sentences = await self._split_sentences(cleaned_text)
        
        # Generate unique ID
        text_id = self._generate_text_id(text, source)

        return {
            'text_id': text_id,
            'source': source,
            'metadata': metadata or {},
            'processed_at': datetime.now().isoformat(),
            'original_text': text,
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'entities': [],
            'relationships': [],
            'facts': []
        }

    async def _integrate_semantic_analysis(
        self,
        processed_data: Dict[str, Any],
        semantic_analysis: Dict[str, Any],
        source: str
    ) -> None:
        """Integrate results from Montague parser into processed data."""
        
        # Extract and convert entities
        for entity_data in semantic_analysis.get('entities', []):
            entity = await self._create_entity_from_analysis(entity_data, source)
            if entity and entity['id'] not in self.entity_cache:
                processed_data['entities'].append(entity)
                self.entity_cache[entity['id']] = entity

        # Extract and convert relationships
        for relation_data in semantic_analysis.get('relations', []):
            relationship = await self._create_relationship_from_analysis(relation_data, source)
            if relationship:
                processed_data['relationships'].append(relationship)

        # Create facts from propositions
        for proposition in semantic_analysis.get('propositions', []):
            fact = await self._create_fact_from_proposition(proposition, source)
            if fact:
                processed_data['facts'].append(fact)

        # Add semantic features to metadata
        if 'semantic_features' in semantic_analysis:
            processed_data['metadata']['semantic_features'] = semantic_analysis['semantic_features']

    async def _create_entity_from_analysis(self, entity_data: Dict, source: str) -> Optional[Dict]:
        """Convert semantic analysis entity to knowledge graph format."""
        try:
            entity_id = KnowledgeUtils.generate_entity_id(entity_data['text'], entity_data['type'])
            
            # Ensure all property values are primitive types for Neo4j compatibility
            properties = {}
            
            # Handle original_label safely
            original_label = entity_data.get('label', '')
            if hasattr(original_label, 'text'):
                properties['original_label'] = str(original_label.text)
            else:
                properties['original_label'] = str(original_label) if original_label else ''
            
            # Handle character positions safely, ensuring they're integers
            start_char = entity_data.get('start', -1)
            end_char = entity_data.get('end', -1)
            
            if hasattr(start_char, '__int__'):
                properties['start_char'] = int(start_char)
            else:
                properties['start_char'] = int(start_char) if isinstance(start_char, (int, float)) else -1
                
            if hasattr(end_char, '__int__'):
                properties['end_char'] = int(end_char)  
            else:
                properties['end_char'] = int(end_char) if isinstance(end_char, (int, float)) else -1
            
            return {
                'id': entity_id,
                'name': entity_data['text'],
                'type': entity_data.get('type', 'Entity'),
                'confidence': float(entity_data.get('confidence', 0.8)),
                'source': source,
                'properties': properties
            }
        except Exception as e:
            logger.warning(f"Failed to create entity from analysis: {e}")
            return None

    async def _create_relationship_from_analysis(self, relation_data: Dict, source: str) -> Optional[Dict]:
        """Convert semantic analysis relation to knowledge graph format."""
        try:
            # Create entity IDs for subject and object
            subject_id = KnowledgeUtils.generate_entity_id(
                relation_data['subject'], 'Entity'
            )
            object_id = KnowledgeUtils.generate_entity_id(
                relation_data['object'], 'Entity'
            )

            # Ensure all property values are primitive types
            properties = {}
            properties['predicate'] = str(relation_data.get('predicate', ''))
            properties['source_span'] = str(relation_data.get('source_span', ''))

            return {
                'source_id': subject_id,
                'target_id': object_id,
                'type': relation_data.get('predicate', 'RELATES'),
                'confidence': float(relation_data.get('confidence', 0.7)),
                'source': source,
                'properties': properties
            }
        except Exception as e:
            logger.warning(f"Failed to create relationship from analysis: {e}")
            return None

    async def _create_fact_from_proposition(self, proposition: Dict, source: str) -> Optional[Dict]:
        """Convert semantic proposition to knowledge graph fact."""
        try:
            # Ensure entities is a list of strings (entity IDs) 
            entities = proposition.get('entities', [])
            if not isinstance(entities, list):
                entities = []
            entities = [str(e) for e in entities]  # Convert to strings
            
            # Ensure metadata contains only primitive types
            metadata = {
                'extraction_method': 'semantic_analysis',
                'entities': entities
            }
            
            return {
                'id': proposition.get('id', f"fact_{datetime.now().timestamp()}"),
                'content': str(proposition['content']),
                'logical_form': str(proposition.get('logical_form', '')),
                'confidence': float(proposition.get('confidence', 0.9)),
                'source': source,
                'metadata': metadata,
                'entities': entities
            }
        except Exception as e:
            logger.warning(f"Failed to create fact from proposition: {e}")
            return None

    async def _post_process_data(self, processed_data: Dict[str, Any]) -> None:
        """Post-process extracted data for validation and cleanup."""
        # Deduplicate entities
        unique_entities = {}
        for entity in processed_data['entities']:
            if entity['id'] not in unique_entities:
                unique_entities[entity['id']] = entity
        processed_data['entities'] = list(unique_entities.values())

        # Validate relationships have valid entity references
        valid_entity_ids = {entity['id'] for entity in processed_data['entities']}
        valid_relationships = []
        
        for rel in processed_data['relationships']:
            if (rel['source_id'] in valid_entity_ids and 
                rel['target_id'] in valid_entity_ids):
                valid_relationships.append(rel)
            else:
                logger.debug(f"Removing invalid relationship: {rel}")
                
        processed_data['relationships'] = valid_relationships

        # Create basic facts from sentences if no semantic facts exist
        if not processed_data['facts'] and processed_data['sentences']:
            for i, sentence in enumerate(processed_data['sentences']):
                if len(sentence.strip()) > 10:
                    # Simplified fact structure with completely flattened metadata
                    fact = {
                        'id': f"{processed_data['text_id']}_fact_{i}",
                        'content': sentence,
                        'logical_form': '',
                        'confidence': 0.8,
                        'source': processed_data['source'],
                        'metadata': {
                            'extraction_method': 'basic_sentence_split',
                            'entities': ''  # Empty string for entity list
                        },
                        'entities': []  # Empty list for entity connections
                    }
                    processed_data['facts'].append(fact)

    async def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing."""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove problematic characters with a simpler approach
        # Keep alphanumeric, spaces, and basic punctuation
        import string
        allowed_chars = string.ascii_letters + string.digits + string.whitespace + '.,!?;:-()[]"\'/&'
        text = ''.join(char for char in text if char in allowed_chars)
        
        # Normalize quotes (using Unicode escape sequences)
        text = re.sub(r'[\u201C\u201D]', '"', text)  # Smart double quotes
        text = re.sub(r'[\u2018\u2019]', "'", text)  # Smart single quotes
        
        # Strip and ensure proper ending
        text = text.strip()
        if text and text[-1] not in '.!?':
            text += '.'
            
        return text

    async def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using improved rules."""
        import re
        
        # Better sentence splitting that handles common abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5:  # Skip very short fragments
                if sentence and sentence[-1] not in '.!?':
                    sentence += '.'
                cleaned_sentences.append(sentence)
                
        return cleaned_sentences

    def _generate_text_id(self, text: str, source: str) -> str:
        """Generate unique identifier for text."""
        import hashlib
        
        # Create hash of text + source for uniqueness
        content_hash = hashlib.md5(f"{text}{source}".encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"text_{timestamp}_{content_hash}"

    async def get_processing_statistics(self) -> Dict:
        """Get statistics about semantic processing performance."""
        return {
            'montague_parser_available': self.montague_parser is not None,
            'entities_cached': len(self.entity_cache),
            'status': 'active'
        }
