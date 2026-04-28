# Project Structure

## Directory Map

### `src/synapse_mcp/`
- `server.py`: Entry point, tool definitions, and server orchestration.
- `__about__.py`: Metadata and versioning.

### `src/synapse_mcp/core/`
- `knowledge_graph.py`: Neo4j driver and graph query logic.
- `mock_knowledge_graph.py`: Testing utility for graph operations.

### `src/synapse_mcp/semantic/`
- `montague_parser.py`: Implementation of formal semantic analysis.

### `src/synapse_mcp/data_pipeline/`
- `text_processor.py`: spaCy/NLTK wrapper for basic NLP.
- `semantic_integrator.py`: Bridges text processing and graph storage.

### `src/synapse_mcp/zettelkasten/`
- `insight_engine.py`: Autonomous synthesis and pattern recognition logic.

### `src/synapse_mcp/wiki/`
- `wiki_adapter.py`: Bridge to local markdown knowledge vaults.

### `src/synapse_mcp/utils/`
- `logging_config.py`: Specialized logging for MCP (stderr redirection).

### `tests/`
- `test_setup.py`: Environment and dependency verification.
- `test_kg_storage.py`: Integration tests for Neo4j.
- `test_basic.py`: Unit tests for core components.
- `debug_test.py`: End-to-end server verification.

## Data Files
- `.env`: Secret management (Neo4j credentials, API keys).
- `pyproject.toml`: Dependency and tool configuration.
- `uv.lock`: Deterministic dependency lockfile.
