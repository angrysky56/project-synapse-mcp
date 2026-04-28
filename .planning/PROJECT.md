# Project Synapse (Maintenance & Hardening)

## What This Is
An autonomous knowledge synthesis engine that bridges a Neo4j graph database with an Obsidian-based LLM-WIKI. This phase focuses on stabilizing the core server, fixing critical tool bugs, and improving error resilience.

## Core Value
Building a rock-solid foundation for autonomous synthesis where knowledge compounds reliably without session-breaking errors.

## Requirements

### Validated
(None yet — ship to validate)

### Active
- [ ] Fix `Context.info` positional argument bug in `generate_insights` and other tools.
- [ ] Implement granular error handling across all MCP tools to prevent broad "Error during..." messages.
- [ ] Stabilize `WikiAdapter` to handle file deletions and renames in the Obsidian vault.
- [ ] Enhance Neo4j connection resilience and logging for better observability.

### Out of Scope
- New synthesis algorithms — Deferred until the foundation is hardened.
- Multi-graph support — Deferred to v2.
- Integration with paid LLM APIs — System remains local-first for now.

## Context
- **Infrastructure**: Running as an MCP server with Neo4j 2026.x.
- **Problem**: Recent issues with `Context.info` usage in the `FastMCP` framework have caused tool failures.
- **Tech Debt**: Broad try/except blocks in `server.py` hide specific failure modes.

## Constraints
- **Tech Stack**: Python 3.12+, `uv`, Neo4j 2026.x.
- **Local-First**: All embeddings and processing must remain local (sentence-transformers).

## Key Decisions
| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Focus on Hardening | Stability is required before expanding the synthesis engine. | — Pending |

---
*Last updated: 2026-04-28 after /gsd-new-project*
