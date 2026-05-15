# Requirements: Project Synapse Hardening

**Defined:** 2026-04-28
**Core Value:** Building a rock-solid foundation for autonomous synthesis where knowledge compounds reliably.

## v1 Requirements (Hardening)

### Tool Stability
- [x] **STAB-01**: Fix `Context.info` calls in `server.py` to match `FastMCP` signature (single message string).
- [x] **STAB-02**: Audit all `Context` logging methods (`info`, `error`, `warn`) for correct usage.
- [x] **STAB-03**: Refactor broad `try/except` blocks in `server.py` to return specific error types (e.g., ConnectionError, ParsingError).

### Wiki Resilience
- [x] **WIKI-01**: Detect and handle "File Not Found" errors when reading wiki pages that may have been renamed or deleted.
- [x] **WIKI-02**: Ensure `wiki_update_index` correctly removes entries for deleted files.

### Infrastructure Observability
- [x] **INF-01**: Implement a detailed connection health check for Neo4j at startup.
- [x] **INF-02**: Add detailed tracing for the semantic pipeline stages (Ingest -> Parse -> Integrate -> Store).

## v2 Requirements (Expansion)

### Infrastructure Optimization
- [x] **INF-03**: Remove `start_autonomous_processing` from default server startup lifespan.
- [x] **INF-04**: Implement `SYNAPSE_AUTONOMOUS_INSIGHTS` environment variable to gate autonomous processing.

### LLM-Enhanced Extraction
- [ ] **EXT-01**: Implement `LlmExtractor` using Gemma e4b (Ollama) with JSON structured output.
- [ ] **EXT-02**: Integrate Pydantic validation for extraction results with a retry/fallback mechanism.
- [ ] **EXT-03**: Create a hybrid extraction pipeline combining spaCy NER with LLM relation/entity extraction.

### On-Demand Synthesis
- [ ] **SYN-01**: Create `synthesize_insights` MCP tool using RAG (Vector + BM25) and Gemma.
- [ ] **SYN-02**: Support optional creation of Zettelkasten nodes from synthesis results.

## Out of Scope
| Feature | Reason |
|---------|--------|
| Multi-graph support | Still out of scope |
| PDF Ingestion | Focus on Markdown and Text for now |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| STAB-01 | Phase 1 | Completed |
| STAB-02 | Phase 1 | Completed |
| STAB-03 | Phase 1 | Completed |
| WIKI-01 | Phase 2 | Completed |
| WIKI-02 | Phase 2 | Completed |
| INF-01 | Phase 1 | Completed |
| INF-02 | Phase 1 | Completed |
| INF-03 | Phase 4 | Completed |
| INF-04 | Phase 4 | Completed |
| EXT-01 | Phase 5 | Pending |
| EXT-02 | Phase 5 | Pending |
| EXT-03 | Phase 5 | Pending |
| SYN-01 | Phase 6 | Pending |
| SYN-02 | Phase 6 | Pending |

**Coverage:**
- v1 + v2 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-28*
*Last updated: 2026-05-15 after adding v2 requirements*
