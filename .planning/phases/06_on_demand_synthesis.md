# Plan: Phase 6 — On-Demand Synthesis Tooling

## Objective
Replace the background autonomous processing with a high-quality, on-demand MCP tool that performs RAG-based knowledge synthesis on specific topics or questions.

## Proposed Changes

### `src/synapse_mcp/server.py`
- Add a new MCP tool: `synthesize_insights(topic_or_question: str)`.
- Wire it to the `InsightEngine`.

### `src/synapse_mcp/zettelkasten/insight_engine.py`
- Implement `manual_synthesis(topic: str)`:
  1. Perform hybrid search via `KnowledgeGraph.query_hybrid`.
  2. Filter high-confidence facts.
  3. Construct a RAG prompt for Gemma e4b (via Ollama).
  4. Generate a structured insight (Zettel).
  5. Store the insight in the knowledge graph.
  6. Return the insight with its evidence trail.

### New Logic: Synthesis Prompt
- Prompt Gemma to:
  - Identify non-obvious patterns across the retrieved facts.
  - Formulate a novel hypothesis.
  - Reference specific facts as evidence.
  - Assign a confidence score.

## Tasks

### 06-01: Implement `manual_synthesis` in `InsightEngine`
- [ ] Add `manual_synthesis` method.
- [ ] Implement RAG logic with hybrid search.
- [ ] Implement Gemma-based synthesis prompt.

### 06-02: Expose `synthesize_insights` MCP Tool
- [ ] Add tool definition to `server.py`.
- [ ] Implement tool handler.

### 06-03: Verification
- [ ] Test synthesis on a known topic (e.g., "Project Synapse Architecture").
- [ ] Verify that evidence trails are correctly linked in Neo4j.

## Verification Plan

### Automated Tests
- [ ] Mock hybrid search results and Ollama generation to test the synthesis flow.

### Manual Verification
- [ ] Use the `synthesize_insights` tool through the MCP inspector or a client.
- [ ] Inspect the generated Zettel in Neo4j (via `MATCH (z:Zettel) RETURN z`).
