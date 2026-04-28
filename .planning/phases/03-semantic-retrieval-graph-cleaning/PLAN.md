# Plan: Phase 3 - Semantic Retrieval & Graph Cleaning

## Objective
Enhance the quality of the knowledge graph and the precision of hybrid search results by cleaning ingestion noise and improving entity resolution.

## Tasks

### 3.1. Noise Filtering in Ingestion Pipeline
- [ ] Research and implement regex patterns for common academic paper artifacts (LaTeX, CID strings, HTML spans).
- [ ] Update `SemanticIntegrator._clean_text` in `src/synapse_mcp/data_pipeline/semantic_integrator.py`.
- [ ] Add unit tests for paper-specific noise removal.

### 3.2. Entity Mapping & Schema Expansion
- [ ] Update `MontagueParser` in `src/synapse_mcp/semantic/montague_parser.py` to support `Concept` and `Method` types.
- [ ] Implement a heuristic-based entity re-mapper to prevent technical concepts (e.g., "Eidetic Learning") from being labeled as `Organization`.
- [ ] Update Neo4j constraints to support new entity types if necessary.

### 3.3. Hybrid Search (RRF) Optimization
- [ ] Refactor `KnowledgeGraph.query_hybrid` to optimize the `k` parameter and result fusion.
- [ ] Ensure that results from `query_by_entities` (graph traversal) are prioritized or interleaved effectively with vector/fulltext results.
- [ ] Implement a basic "relevance threshold" to filter out low-score RRF results.

### 3.4. Verification & Cleanup
- [ ] Create a maintenance tool `wiki_resolve_entities` to retroactively fix misidentified node types in the graph.
- [ ] Re-ingest a sample paper to verify noise reduction and entity accuracy.
- [ ] Run benchmark queries to compare search quality.

## Verification Plan
1. **Unit Tests**: Test the new regex cleaner with raw LaTeX/HTML snippets.
2. **Integration Test**: Ingest a new academic paper and verify that no `id="A6.T8..."` entities are created.
3. **Manual UAT**: Run `query_knowledge` and verify that "Eidetic Learning" is returned with high relevance and the correct `Concept` type.
