"""Unit tests for LlmExtractor.

Mocks the provider seam (``provider.chat_json``) rather than aiohttp —
the extractor no longer talks HTTP directly, it delegates to an
LlmProvider, so that is the correct boundary to fake.
"""

from unittest.mock import AsyncMock

import pytest

from synapse_mcp.llm import LlmResponseError
from synapse_mcp.semantic.llm_extractor import LlmExtractor


@pytest.mark.asyncio
async def test_extract_semantics_success():
    extractor = LlmExtractor()
    extractor.provider.chat_json = AsyncMock(
        return_value={
            "entities": [{"text": "Apple", "type": "Company", "confidence": 0.9}],
            "relations": [],
        }
    )

    result = await extractor.extract_semantics("Apple is a company.")

    assert len(result.entities) == 1
    assert result.entities[0].text == "Apple"
    assert result.entities[0].type == "Company"


@pytest.mark.asyncio
async def test_extract_semantics_retry_on_invalid_json():
    extractor = LlmExtractor()
    extractor.provider.chat_json = AsyncMock(
        side_effect=[
            LlmResponseError("invalid json"),
            {
                "entities": [{"text": "Orange", "type": "Fruit", "confidence": 0.8}],
                "relations": [],
            },
        ]
    )

    result = await extractor.extract_semantics("Orange is a fruit.")

    assert len(result.entities) == 1
    assert result.entities[0].text == "Orange"
    assert extractor.provider.chat_json.call_count == 2


@pytest.mark.asyncio
async def test_extract_semantics_returns_empty_after_all_retries():
    extractor = LlmExtractor()
    extractor.provider.chat_json = AsyncMock(
        side_effect=LlmResponseError("provider down")
    )

    result = await extractor.extract_semantics("Anything at all.")

    assert result.entities == []
    assert result.relations == []
    assert extractor.provider.chat_json.call_count == 2
