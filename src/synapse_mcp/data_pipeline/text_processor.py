"""
Text processing pipeline for Project Synapse.

This module handles the ingestion and preprocessing of text from various sources,
preparing it for semantic analysis and knowledge extraction.
"""

import asyncio
import hashlib
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import aiofiles
import re
import logging

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class TextProcessor:
    """
    Text processing pipeline for converting raw text into structured data.

    Handles text cleaning, normalization, and preparation for semantic analysis.
    """

    def __init__(self):
        self.supported_formats = {'.txt', '.md', '.json'}
        self.batch_size = int(os.getenv('SEMANTIC_BATCH_SIZE', '50'))

    async def initialize(self) -> None:
        """Initialize the text processor."""
        logger.info("Text processor initialized successfully")

    async def process_text(
        self,
        text: str,
        source: str = "user_input",
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process raw text into structured data ready for knowledge graph storage.

        Args:
            text: Raw text to process
            source: Source identifier for provenance
            metadata: Additional metadata about the text

        Returns:
            Dictionary containing processed entities, relationships, and facts
        """
        logger.debug(f"Processing text from source: {source}")

        # Clean and normalize text
        cleaned_text = await self._clean_text(text)

        # Split into sentences for atomic processing
        sentences = await self._split_sentences(cleaned_text)

        # Generate unique IDs
        text_id = self._generate_text_id(text, source)

        # Prepare structured data
        processed_data = {
            'text_id': text_id,
            'source': source,
            'metadata': metadata or {},
            'processed_at': datetime.now().isoformat(),
            'sentences': sentences,
            'entities': [],
            'relationships': [],
            'facts': []
        }

        # Process each sentence for facts
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 10:  # Skip very short sentences
                fact = await self._create_fact_from_sentence(
                    sentence, source, text_id, i
                )
                processed_data['facts'].append(fact)

        logger.debug(f"Processed {len(sentences)} sentences into {len(processed_data['facts'])} facts")

        return processed_data

    async def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a text file into structured data.

        Args:
            file_path: Path to the text file

        Returns:
            Processed data dictionary
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info(f"Processing file: {file_path}")

        # Read file content
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()

        # Extract metadata from file
        metadata = {
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'file_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'file_format': file_path.suffix
        }

        return await self.process_text(
            text=content,
            source=f"file:{file_path.name}",
            metadata=metadata
        )

    async def process_batch(self, texts: List[Dict]) -> List[Dict]:
        """
        Process multiple texts in batch for efficiency.

        Args:
            texts: List of dictionaries with 'text', 'source', and 'metadata' keys

        Returns:
            List of processed data dictionaries
        """
        logger.info(f"Processing batch of {len(texts)} texts")

        # Process in chunks to avoid memory issues
        results = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]

            # Process chunk concurrently
            tasks = [
                self.process_text(
                    text=item['text'],
                    source=item.get('source', f'batch_item_{i}'),
                    metadata=item.get('metadata', {})
                )
                for item in chunk
            ]

            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            for j, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process batch item {i+j}: {result}")
                else:
                    results.append(result)

        logger.info(f"Successfully processed {len(results)} out of {len(texts)} texts")
        return results

    async def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters that might interfere with processing
        # Keep alphanumeric, spaces, and basic punctuation
        import string
        allowed_chars = string.ascii_letters + string.digits + string.whitespace + '.,!?;:-()[]"\'/'
        text = ''.join(char for char in text if char in allowed_chars)

        # Normalize quotes (using Unicode escape sequences)
        text = re.sub(r'[\u201C\u201D]', '"', text)  # Smart double quotes
        text = re.sub(r'[\u2018\u2019]', "'", text)  # Smart single quotes

        # Strip and ensure text ends with punctuation
        text = text.strip()
        if text and text[-1] not in '.!?':
            text += '.'

        return text

    async def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple rules."""
        # Simple sentence splitting - can be enhanced with spaCy later
        sentences = re.split(r'[.!?]+\s+', text)

        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5:  # Skip very short fragments
                # Ensure sentence has proper ending
                if sentence and sentence[-1] not in '.!?':
                    sentence += '.'
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    async def _create_fact_from_sentence(
        self,
        sentence: str,
        source: str,
        text_id: str,
        sentence_index: int
    ) -> Dict:
        """Create a fact record from a sentence."""
        fact_id = f"{text_id}_fact_{sentence_index}"

        return {
            'id': fact_id,
            'content': sentence,
            'source': source,
            'confidence': 1.0,  # High confidence for direct extraction
            'metadata': {
                'extraction_method': 'direct',
                'entities': ''  # Empty string for entity list
            },
            'entities': [],  # Will be populated by semantic analysis
            'logical_form': ''  # Will be populated by Montague parser
        }

    def _generate_text_id(self, text: str, source: str) -> str:
        """Generate unique identifier for text."""
        # Create hash of text + source for uniqueness
        content_hash = hashlib.md5(f"{text}{source}".encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"text_{timestamp}_{content_hash}"

    async def extract_entities_preview(self, text: str) -> List[Dict]:
        """
        Quick entity extraction preview without full processing.

        Args:
            text: Text to analyze

        Returns:
            List of potential entities found
        """
        # Simple regex-based entity detection for preview
        entities = []

        # Detect potential person names (capitalized words)
        person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        for match in re.finditer(person_pattern, text):
            entities.append({
                'text': match.group(),
                'type': 'Person',
                'confidence': 0.7,
                'start': match.start(),
                'end': match.end()
            })

        # Detect potential organizations (words ending with common org suffixes)
        org_pattern = r'\b[A-Z][a-zA-Z\s]*(Inc|Corp|LLC|Ltd|Company|Organization|Institute)\b'
        for match in re.finditer(org_pattern, text):
            entities.append({
                'text': match.group(),
                'type': 'Organization',
                'confidence': 0.8,
                'start': match.start(),
                'end': match.end()
            })

        # Detect potential locations (capitalized words after location indicators)
        location_pattern = r'\b(?:in|at|from|to)\s+([A-Z][a-zA-Z\s]+(?:City|State|Country|County|Province)?)\b'
        for match in re.finditer(location_pattern, text):
            entities.append({
                'text': match.group(1),
                'type': 'Location',
                'confidence': 0.6,
                'start': match.start(1),
                'end': match.end(1)
            })

        return entities

    async def get_processing_statistics(self) -> Dict:
        """Get statistics about text processing performance."""
        return {
            'supported_formats': list(self.supported_formats),
            'batch_size': self.batch_size,
            'status': 'active'
        }


class DataPipelineManager:
    """
    Manager for coordinating different data input sources.

    Handles various input types and coordinates with text processor.
    """

    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor
        self.supported_sources = ['file', 'url', 'api', 'user_input']

    async def ingest_from_source(self, source_type: str, source_config: Dict) -> Dict:
        """
        Ingest data from various sources.

        Args:
            source_type: Type of source (file, url, api, user_input)
            source_config: Configuration for the source

        Returns:
            Processed data dictionary
        """
        if source_type not in self.supported_sources:
            raise ValueError(f"Unsupported source type: {source_type}")

        if source_type == 'file':
            return await self.text_processor.process_file(source_config['path'])
        elif source_type == 'user_input':
            return await self.text_processor.process_text(
                text=source_config['text'],
                source=source_config.get('source', 'user_input'),
                metadata=source_config.get('metadata', {})
            )
        elif source_type == 'url':
            # Placeholder for URL processing
            raise NotImplementedError("URL processing not yet implemented")
        elif source_type == 'api':
            # Placeholder for API processing
            raise NotImplementedError("API processing not yet implemented")

        return {}
