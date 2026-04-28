from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapse_mcp.server import SynapseServer


@pytest.fixture
def mock_server():
    server = SynapseServer()
    server.knowledge_graph = AsyncMock()
    server.wiki_adapter = AsyncMock()
    return server


@pytest.mark.asyncio
async def test_server_health_success(mock_server):
    """Test health check when all components are healthy."""
    mock_server.knowledge_graph.check_health.return_value = True
    mock_server.wiki_adapter.check_health.return_value = True

    status = await mock_server.check_health()

    assert status["status"] == "healthy"
    assert status["components"]["knowledge_graph"] == "healthy"
    assert status["components"]["wiki_adapter"] == "healthy"


@pytest.mark.asyncio
async def test_server_health_kg_failure(mock_server):
    """Test health check when knowledge graph is unhealthy."""
    mock_server.knowledge_graph.check_health.side_effect = RuntimeError(
        "Neo4j connection lost"
    )
    mock_server.wiki_adapter.check_health.return_value = True

    status = await mock_server.check_health()

    assert status["status"] == "unhealthy"
    assert "unhealthy: Neo4j connection lost" in status["components"]["knowledge_graph"]
    assert status["components"]["wiki_adapter"] == "healthy"


@pytest.mark.asyncio
async def test_server_health_wiki_failure(mock_server):
    """Test health check when wiki vault is unhealthy."""
    mock_server.knowledge_graph.check_health.return_value = True
    mock_server.wiki_adapter.check_health.side_effect = RuntimeError(
        "Vault not writable"
    )

    status = await mock_server.check_health()

    assert status["status"] == "unhealthy"
    assert status["components"]["knowledge_graph"] == "healthy"
    assert "unhealthy: Vault not writable" in status["components"]["wiki_adapter"]


@pytest.mark.asyncio
async def test_server_initialize_failure_stops_startup():
    """Test that a health check failure during initialization raises an exception."""
    with patch("synapse_mcp.server.KnowledgeGraph", return_value=AsyncMock()):
        with patch(
            "synapse_mcp.server.WikiAdapter", return_value=AsyncMock()
        ) as mock_wiki:
            mock_wiki.return_value.initialize.side_effect = RuntimeError(
                "Wiki init failed"
            )

            server = SynapseServer()
            with pytest.raises(RuntimeError, match="Wiki init failed"):
                await server.initialize()
