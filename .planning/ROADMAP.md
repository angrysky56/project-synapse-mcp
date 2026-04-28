# Roadmap: Project Synapse Hardening

## Overview
A two-phase hardening journey focused on fixing critical server-level bugs, improving error granularity, and ensuring the stability of the Obsidian-Wiki bridge.

## Phases
- [x] **Phase 1: Server Stability & Observability** - Fix `Context.info` bugs, refactor error handling, and enhance infrastructure health checks.
- [x] **Phase 2: Wiki Bridge Resilience** - Stabilize `WikiAdapter` against file system changes and improve indexing robustness.

## Phase Details

### Phase 1: Server Stability & Observability
**Goal**: Eliminate tool failures caused by incorrect API usage and improve debuggability.
**Depends on**: Nothing
**Requirements**: [STAB-01, STAB-02, STAB-03, INF-01, INF-02]
**Success Criteria**:
  1. `generate_insights` tool no longer fails with `Context.info` positional argument error.
  2. All MCP tools return specific error messages instead of broad exception summaries.
  3. Neo4j connection health is verified and logged at startup.
**Plans**: 3 plans

Plans:
- [x] **01-01**: Fix broken logging in `generate_insights` and fix test imports.
- [x] **01-02**: Implement `check_health()` for all components and granular error handling.
- [x] **01-03**: Standardize logging and metrics (Prometheus/Grafana ready).

### Phase 2: Wiki Bridge Resilience [In Progress]
**Goal**: Ensure the LLM-WIKI interface is resilient to manual user changes in the Obsidian vault.
**Depends on**: Phase 1
**Requirements**: [WIKI-01, WIKI-02]
**Success Criteria**:
  1. `wiki_read_page` gracefully handles missing files.
  2. `wiki_update_index` correctly purges stale entries for deleted markdown files.
**Plans**: 2 plans

Plans:
- [x] 02-01: Harden `WikiAdapter` file I/O operations.
- [x] 02-02: Improve index management and orphan cleanup logic.

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Server Stability | 3/3 | Completed | 2026-04-28 |
| 2. Wiki Bridge | 2/2 | Completed | 2026-04-28 |

### Phase 3: Semantic Retrieval & Graph Cleaning
**Goal**: Enhance the quality of the knowledge graph and improve the precision of semantic retrieval.
**Depends on**: Phase 2
**Requirements**: [SEM-01, SEM-02, SEM-03]
**Success Criteria**:
  1. `query_knowledge` uses a consistent, modern retrieval strategy across vector and fulltext.
  2. Ingestion pipeline filters out LaTeX/PDF noise (e.g., `id="A6.T8..."`).
  3. Entity resolution logic handles near-duplicate entities and type corrections.
**Plans**: 3 plans

Plans:
- [ ] 03-01: Standardize hybrid search query logic (RRF optimization).
- [ ] 03-02: Implement noise-filtering pre-processor for markdown ingestion.
- [ ] 03-03: Create `wiki_resolve_entities` tool for graph maintenance.

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Server Stability | 3/3 | Completed | 2026-04-28 |
| 2. Wiki Bridge | 2/2 | Completed | 2026-04-28 |
| 3. Semantic Retrieval | 0/3 | Planning | - |
