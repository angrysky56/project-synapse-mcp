import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapse_mcp.data_pipeline.semantic_integrator import SemanticIntegrator
from synapse_mcp.semantic.llm_extractor import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)


@pytest.mark.asyncio
async def test_hybrid_pipeline_llm_provider():
    # Mock LLM result
    llm_entities = [
        ExtractedEntity(text="Quantum Computing", type="Field", confidence=0.95),
        ExtractedEntity(text="Gemma", type="Model", confidence=0.9),
    ]
    llm_relations = [
        ExtractedRelation(
            subject="Gemma",
            object="Quantum Computing",
            predicate="analyzes",
            confidence=0.8,
        )
    ]
    llm_result = ExtractionResult(entities=llm_entities, relations=llm_relations)

    # Mock Montague result
    montague_entities = [
        {"text": "Quantum Computing", "type": "CONCEPT", "confidence": 0.8},
        {"text": "Google", "type": "ORG", "confidence": 0.9},  # Something LLM missed
    ]
    montague_analysis = {
        "entities": montague_entities,
        "relations": [],
        "propositions": [{"content": "Gemma is a model.", "confidence": 0.9}],
    }

    # Setup environment
    with patch.dict(os.environ, {"EXTRACTION_PROVIDER": "llm"}):
        integrator = SemanticIntegrator()

        # Mock the components
        integrator.llm_extractor = AsyncMock()
        integrator.llm_extractor.extract_semantics.return_value = llm_result

        integrator.montague_parser = MagicMock()
        integrator.montague_parser.initialize = AsyncMock()
        integrator.montague_parser.nlp = True  # Simulation
        integrator.montague_parser.parse_text = AsyncMock(
            return_value=montague_analysis
        )

        await integrator.initialize()
        assert integrator.extraction_provider == "llm"

        result = await integrator.process_text_with_semantics(
            "Gemma analyzes Quantum Computing. Google made it."
        )

        # Verify LLM was called
        integrator.llm_extractor.extract_semantics.assert_called_once()

        # Verify Hybrid Merge: LLM (2) + unique spaCy (1) = 3 entities
        # Note: "Quantum Computing" is in both, so it should be deduplicated
        entity_names = {e["name"] for e in result["entities"]}
        assert "Quantum Computing" in entity_names
        assert "Gemma" in entity_names
        assert "Google" in entity_names
        assert len(result["entities"]) == 3

        # Verify Montague was still called for hybrid/propositions
        integrator.montague_parser.parse_text.assert_called_once()
        assert len(result["facts"]) > 0


@pytest.mark.asyncio
async def test_hybrid_pipeline_fallback_to_montague():
    # Mock LLM failure
    with patch.dict(os.environ, {"EXTRACTION_PROVIDER": "llm"}):
        integrator = SemanticIntegrator()

        integrator.llm_extractor = AsyncMock()
        integrator.llm_extractor.extract_semantics.side_effect = Exception(
            "Ollama down"
        )

        montague_analysis = {
            "entities": [{"text": "Apple", "type": "ORG", "confidence": 0.9}],
            "relations": [],
            "propositions": [],
        }
        integrator.montague_parser = MagicMock()
        integrator.montague_parser.initialize = AsyncMock()
        integrator.montague_parser.nlp = True
        integrator.montague_parser.parse_text = AsyncMock(
            return_value=montague_analysis
        )

        await integrator.initialize()

        result = await integrator.process_text_with_semantics("Apple.")

        # Should have fallback results from Montague
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Apple"
