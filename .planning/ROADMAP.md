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

| Phase               | Plans Complete | Status    | Completed  |
| ------------------- | -------------- | --------- | ---------- |
| 1. Server Stability | 3/3            | Completed | 2026-04-28 |
| 2. Wiki Bridge      | 2/2            | Completed | 2026-04-28 |

47: ### Phase 4: Infrastructure Optimization (Autonomous Engine Kill-Switch)
48: **Goal**: Stabilize server startup and eliminate background resource saturation by disabling the autonomous insight engine by default.
49: **Depends on**: Phase 2
50: **Requirements**: [INF-03, INF-04]
51: **Success Criteria**:
52: 1. Server starts significantly faster (no background loop initialization lag).
53: 2. `start_autonomous_processing` is removed from the default lifespan tasks.
54: 3. Autonomous processing is gated behind `SYNAPSE_AUTONOMOUS_INSIGHTS=on`.
55: **Plans**: 1 plan
56:
57: Plans:
58: - [x] 04-01: Implement `SYNAPSE_AUTONOMOUS_INSIGHTS` toggle and refactor lifespan.
59:
60: ### Phase 5: LLM-Enhanced Knowledge Extraction
61: **Goal**: Improve the quality of entity and relationship extraction by introducing a hybrid LLM-Montague pipeline.
62: **Depends on**: Phase 4
63: **Requirements**: [EXT-01, EXT-02, EXT-03]
64: **Success Criteria**:
65: 1. `LlmExtractor` successfully parses text using Gemma e4b via Ollama.
66: 2. Pydantic validation handles LLM output failures with Montague fallback.
67: 3. `EXTRACTION_PROVIDER` toggle allows switching between providers.
68: **Plans**: 2 plans
69:
70: Plans:
71: - [x] 05-01: Implement `LlmExtractor` with Pydantic validation and spaCy hybrid NER.
72: - [x] 05-02: Integrate `EXTRACTION_PROVIDER` into the ingestion pipeline.
73:
74: ### Phase 6: On-Demand Synthesis Tooling
75: **Goal**: Replace background synthesis with a manual, on-demand MCP tool for better control and accuracy.
76: **Depends on**: Phase 5
77: **Requirements**: [SYN-01, SYN-02]
78: **Success Criteria**:
79: 1. New MCP tool `synthesize_insights` returns high-quality synthesis for a given topic.
80: 2. Synthesis results can be optionally saved as new Zettelkasten nodes.
81: **Plans**: 1 plan
82:
83: Plans:
84: - [ ] 06-01: Implement `synthesize_insights` tool with RAG-based Gemma prompting.
85:
86: ## Progress
87:
88: | Phase | Plans Complete | Status | Completed |
89: |-------|----------------|--------|-----------|
90: | 1. Server Stability | 3/3 | Completed | 2026-04-28 |
91: | 2. Wiki Bridge | 2/2 | Completed | 2026-04-28 |
92: | 3. Semantic Retrieval | 0/3 | Backlogged | - |
93: | 4. Infrastructure Opt | 1/1 | Completed | 2026-05-15 |
94: | 5. LLM Extraction | 2/2 | Completed | 2026-05-15 |
95: | 6. On-Demand Synthesis | 0/1 | Upcoming | - |
