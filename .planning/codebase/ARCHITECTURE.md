# Architecture

## System Overview
Project Synapse is designed as an autonomous knowledge synthesis engine. It doesn't just store facts; it attempts to understand the semantic structure of information and generate novel insights.

## Core Pipeline (Ingestion)
1. **Raw Input**: Text received via `ingest_text`.
2. **Linguistic Processing**: `TextProcessor` extracts entities and structural features.
3. **Semantic Parsing**: `MontagueParser` converts natural language into logical forms.
4. **Graph Integration**: `SemanticIntegrator` maps semantic forms to graph primitives.
5. **Storage**: `KnowledgeGraph` commits the data to Neo4j.

## Synthesis Loop (Zettelkasten)
- **Autonomous Processing**: `InsightEngine` monitors the graph for new data.
- **Pattern Recognition**: Identifies isomorphisms, contradictions, or emerging themes.
- **Insight Generation**: Creates new nodes in the graph representing synthesized knowledge.
- **Wiki Sync**: Optionally updates the LLM-WIKI vault with summaries and indexes.

## Retrieval Strategy (Hybrid Search)
- **Vector Search**: ANN (Approximate Nearest Neighbor) using sentence-transformers.
- **Keyword Search**: BM25 style ranking via Neo4j/DuckDB.
- **Graph Traversal**: Direct relationship exploration starting from query-extracted entities.
- **RRF (Reciprocal Rank Fusion)**: Merges results from multiple retrieval paths for optimal relevance.

## Component Lifecycle
- Handled by `SynapseServer` in `server.py` using `FastMCP`'s lifespan context.
- Ensures clean connection management for Neo4j and background task orchestration.
