# Plan: Phase 5 — LLM-Enhanced Knowledge Extraction

## Objective
Introduce a hybrid extraction pipeline that leverages LLMs (Gemma e4b via Ollama) for high-quality entity and relation extraction while maintaining the precision of spaCy for proper nouns.

## Proposed Changes

### New Component: `src/synapse_mcp/semantic/llm_extractor.py`
- `LlmExtractor` class.
- Integration with Ollama (via `aiohttp` or `ollama` library).
- Pydantic schemas:
  - `ExtractedEntity(name: str, type: str, confidence: float)`
  - `ExtractedRelation(subject: str, predicate: str, object: str, confidence: float)`
  - `ExtractionResult(entities: list[ExtractedEntity], relations: list[ExtractedRelation])`
- Structured prompt for Gemma e4b to output JSON.
- One-retry logic on parse failure.

### `src/synapse_mcp/data_pipeline/semantic_integrator.py`
- Support for `EXTRACTION_PROVIDER` environment variable (`montague` vs `llm`).
- Coordination logic:
  1. Call `LlmExtractor` if provider is `llm`.
  2. Fallback to `MontagueParser` on failure.
  3. Hybrid Merge: Inject spaCy NER results (PERSON, ORG, GPE) into the entity list.

### `src/synapse_mcp/semantic/montague_parser.py`
- Expose spaCy `nlp` or a specialized NER method for use by `LlmExtractor`.

## Tasks

### 05-01: Implement `LlmExtractor` & Schemas
- [ ] Define Pydantic schemas.
- [ ] Implement Ollama client logic and prompt engineering.
- [ ] Add unit tests for extraction parsing.

### 05-02: Integrate Provider Toggle & Hybrid Logic
- [ ] Update `SemanticIntegrator` to support provider switching.
- [ ] Implement hybrid NER merge (spaCy + LLM).
- [ ] Verify fallback mechanism works when Ollama is unavailable.

## Verification Plan

### Automated Tests
- [ ] Mock Ollama responses to verify Pydantic validation and retry logic.
- [ ] Verify that spaCy entities are correctly prioritized/merged.

### Quality Assessment
- [ ] Run extraction on a set of gold-standard sentences.
- [ ] Compare "before" (Montague) and "after" (LLM) graph density and accuracy.
