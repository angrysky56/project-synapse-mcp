# Integrations

## External Services

### Neo4j Graph Database
- **Role**: Primary persistence layer for the knowledge graph.
- **Integration**: `KnowledgeGraph` class in `src/synapse_mcp/core/knowledge_graph.py`.
- **Functionality**: Stores entities, relationships, and semantic facts; supports hybrid RRF (Reciprocal Rank Fusion) retrieval.

### spaCy / NLTK
- **Role**: Linguistic analysis and entity extraction.
- **Integration**: `TextProcessor` in `src/synapse_mcp/data_pipeline/text_processor.py`.
- **Functionality**: Tokenization, POS tagging, and named entity recognition (NER).

### LLM-WIKI (Local Vault)
- **Role**: Human-readable knowledge interface and documentation.
- **Integration**: `WikiAdapter` in `src/synapse_mcp/wiki/wiki_adapter.py`.
- **Functionality**: Manages markdown pages in a local vault, including indexing, search, and graph-based link analysis.

## Internal Component Mapping

### MCP Server (`server.py`)
- Acts as the orchestrator for all components.
- Exposes tools like `ingest_text`, `query_knowledge`, and `generate_insights`.

### Semantic Integrator
- Bridges raw text processing and graph storage.
- Uses `MontagueParser` for formal semantic representation.

### Insight Engine (Zettelkasten)
- Runs autonomous background tasks to identify patterns in the knowledge graph.
- Generates new "zettels" (insights) with evidence trails.

### DuckDB
- Used for analytical queries and local data processing that doesn't fit the graph model.
