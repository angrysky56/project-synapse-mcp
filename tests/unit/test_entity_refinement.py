
import pytest
from synapse_mcp.semantic.montague_parser import MontagueParser

@pytest.mark.asyncio
async def test_entity_refinement():
    parser = MontagueParser()
    await parser.initialize()
    
    # Test cases: (text, original_type, expected_type)
    test_cases = [
        ("Eidetic Learning", "Organization", "Concept"),
        ("Maximum Occupancy Principle", "Organization", "Concept"),
        ("Backpropagation Algorithm", "Product", "Method"),
        ("Standard Neural Network", "Organization", "Concept"),
        ("Google", "Organization", "Organization"), # Should stay Org
        ("Python", "Product", "Product"), # Should stay Product if not matching Method hints
    ]
    
    for text, orig_type, expected_type in test_cases:
        refined = parser._refine_entity_type(text, orig_type)
        assert refined == expected_type, f"Failed for {text}: expected {expected_type}, got {refined}"

@pytest.mark.asyncio
async def test_normalize_new_types():
    parser = MontagueParser()
    assert parser._normalize_entity_type("CONCEPT") == "Concept"
    assert parser._normalize_entity_type("METHOD") == "Method"
