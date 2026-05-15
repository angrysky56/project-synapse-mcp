from unittest.mock import AsyncMock, patch

import pytest

from synapse_mcp.semantic.llm_extractor import ExtractionResult, LlmExtractor


@pytest.mark.asyncio
async def test_extract_semantics_success():
    extractor = LlmExtractor()

    mock_response = {
        "message": {
            "content": '{"entities": [{"text": "Apple", "type": "Company", "confidence": 0.9}], "relations": []}'
        }
    }

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = mock_response
        mock_resp.__aenter__.return_value = mock_resp
        mock_post.return_value = mock_resp

        result = await extractor.extract_semantics("Apple is a company.")

        assert len(result.entities) == 1
        assert result.entities[0].text == "Apple"
        assert result.entities[0].type == "Company"


@pytest.mark.asyncio
async def test_extract_semantics_retry_on_invalid_json():
    extractor = LlmExtractor()

    invalid_response = {"message": {"content": "invalid json"}}
    valid_response = {
        "message": {
            "content": '{"entities": [{"text": "Orange", "type": "Fruit", "confidence": 0.8}], "relations": []}'
        }
    }

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp1 = AsyncMock()
        mock_resp1.status = 200
        mock_resp1.json.return_value = invalid_response
        mock_resp1.__aenter__.return_value = mock_resp1

        mock_resp2 = AsyncMock()
        mock_resp2.status = 200
        mock_resp2.json.return_value = valid_response
        mock_resp2.__aenter__.return_value = mock_resp2

        mock_post.side_effect = [mock_resp1, mock_resp2]

        result = await extractor.extract_semantics("Orange is a fruit.")

        assert len(result.entities) == 1
        assert result.entities[0].text == "Orange"
        assert mock_post.call_count == 2
