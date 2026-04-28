# Roadmap: Project Synapse Hardening

## Overview
A two-phase hardening journey focused on fixing critical server-level bugs, improving error granularity, and ensuring the stability of the Obsidian-Wiki bridge.

## Phases
- [ ] **Phase 1: Server Stability & Observability** - Fix `Context.info` bugs, refactor error handling, and enhance infrastructure health checks.
- [ ] **Phase 2: Wiki Bridge Resilience** - Stabilize `WikiAdapter` against file system changes and improve indexing robustness.

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
- [ ] 01-01: Fix `Context.info` and audit logging usage across `server.py`.
- [ ] 01-02: Refactor error handling in `server.py` and implement health checks.
- [ ] 01-03: Implement stage-level tracing for the semantic pipeline.

### Phase 2: Wiki Bridge Resilience
**Goal**: Ensure the LLM-WIKI interface is resilient to manual user changes in the Obsidian vault.
**Depends on**: Phase 1
**Requirements**: [WIKI-01, WIKI-02]
**Success Criteria**:
  1. `wiki_read_page` gracefully handles missing files.
  2. `wiki_update_index` correctly purges stale entries for deleted markdown files.
**Plans**: 2 plans

Plans:
- [ ] 02-01: Harden `WikiAdapter` file I/O operations.
- [ ] 02-02: Improve index management and orphan cleanup logic.

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Server Stability | 0/3 | Not started | - |
| 2. Wiki Bridge | 0/2 | Not started | - |
