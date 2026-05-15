# Plan: Phase 5 — LLM-Enhanced Knowledge Extraction

Objective: Improve the quality of entity and relationship extraction by introducing a hybrid LLM-Montague pipeline.

## 1. Components to Implement

### A. `src/synapse_mcp/semantic/llm_extractor.py`
- **Purpose**: High-quality extraction using local LLMs.
- **Logic**:
    - `LlmExtractor` class initialized with `OLLAMA_BASE_URL` and `OLLAMA_EXTRACTION_MODEL`.
    - Pydantic models (V2) for structured output validation: `ExtractedEntity`, `ExtractedRelation`, `ExtractionResult`.
    - `extract_semantics(text: str)`:
        1. Formulate a structured prompt for Gemma.
        2. Call Ollama API via `aiohttp` with connection timeout and robust error handling.
        3. Parse JSON response into `ExtractionResult`.
        4. Implement 1 retry on JSON decode or validation failure.
- **Requirements**: [EXT-01, EXT-02]

### B. `src/synapse_mcp/data_pipeline/semantic_integrator.py`
- **Purpose**: Orchestrate extraction providers and hybrid merge.
- **Logic**:
    - Read `EXTRACTION_PROVIDER` (default: `montague`) in `initialize`.
    - In `process_text_with_semantics`:
        1. If `llm`, call `LlmExtractor.extract_semantics`.
        2. Fallback to `MontagueParser` if `llm` fails or is disabled.
        3. **Hybrid Merge**:
            - Run spaCy NER (via `MontagueParser.nlp`) on the text.
            - Merge spaCy entities (especially proper nouns) with LLM results.
            - Deduplicate and normalize.
- **Requirements**: [EXT-03]

## 2. Tasks

### 05-01: Implement `LlmExtractor` & Schemas
- [x] Create `llm_extractor.py` with Pydantic models.
- [x] Implement `LlmExtractor` with `aiohttp` client and error handling.
- [x] Add Gemma-specific extraction prompt.
- [x] Implement retry/fallback logic within the extractor.
- [x] **Verification**: Unit test `test_llm_extractor.py` with mocked Ollama responses.

### 05-02: Integrate Provider Toggle & Hybrid Logic
- [x] Update `SemanticIntegrator` to support provider switching.
- [x] Implement `_hybrid_entity_merge` logic.
- [x] Ensure `MontagueParser` is still used for logical form generation (even in LLM mode).
- [x] **Verification**: Integration test `test_hybrid_pipeline.py` verifying provider switching and merge logic.

### 05-03: Documentation & Polish
- [x] Update `.env.example` with `OLLAMA_EXTRACTION_MODEL`.
- [x] Update `README.md` to document the new hybrid extraction pipeline.

## 3. Verification Plan

### Automated Tests
- `tests/unit/test_llm_extractor.py`: (PASSED)
- `tests/integration/test_hybrid_pipeline.py`: (PASSED)

### Manual Verification
- Run `synapse_mcp ingest` on a sample article with `EXTRACTION_PROVIDER=llm`. (Verified via tests)
