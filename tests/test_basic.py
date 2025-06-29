"""
Test suite for Project Synapse MCP Server.

Basic tests to verify installation and core functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

# Import the modules to test
from project_synapse_mcp.data_pipeline.text_processor import TextProcessor
from project_synapse_mcp.semantic.montague_parser import MontagueParser
from project_synapse_mcp.utils.logging_config import setup_logging


class TestTextProcessor:
    """Test the text processing pipeline."""

    @pytest.fixture
    async def text_processor(self):
        """Create a text processor instance."""
        processor = TextProcessor()
        await processor.initialize()
        return processor

    @pytest.mark.asyncio
    async def test_process_text_basic(self, text_processor):
        """Test basic text processing functionality."""
        text = "This is a test sentence. It contains some information."
        result = await text_processor.process_text(text, "test_source")

        assert 'text_id' in result
        assert result['source'] == 'test_source'
        assert 'sentences' in result
        assert 'facts' in result
        assert len(result['sentences']) >= 1
        assert len(result['facts']) >= 1

    @pytest.mark.asyncio
    async def test_entity_preview(self, text_processor):
        """Test entity preview functionality."""
        text = "John Smith works at Microsoft Corporation in Seattle."
        entities = await text_processor.extract_entities_preview(text)

        assert len(entities) > 0
        assert any(entity['type'] == 'Person' for entity in entities)


class TestLoggingConfig:
    """Test logging configuration."""

    def test_setup_logging(self):
        """Test that logging can be configured."""
        logger = setup_logging("test_logger", level="INFO")
        assert logger is not None
        assert logger.name == "test_logger"


class TestMockNeo4j:
    """Test basic functionality without requiring Neo4j."""

    def test_mock_knowledge_graph(self):
        """Test that knowledge graph can be mocked."""
        # This test verifies the structure without requiring a real database
        from project_synapse_mcp.core.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        assert kg.uri is not None
        assert kg.user is not None
        assert kg.password is not None


if __name__ == "__main__":
    pytest.main([__file__])
