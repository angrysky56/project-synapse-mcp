#!/usr/bin/env python3
"""
Debug script to test semantic integrator functionality and isolate regex issues.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def test_semantic_integrator():
    """Test the semantic integrator in isolation."""
    print("🔍 Testing semantic integrator...")

    try:
        from synapse_mcp.data_pipeline.semantic_integrator import SemanticIntegrator
        from synapse_mcp.semantic.montague_parser import MontagueParser

        # Initialize components
        parser = MontagueParser()
        await parser.initialize()

        integrator = SemanticIntegrator(parser)
        await integrator.initialize()

        # Test simple text
        test_text = "Machine learning is powerful."
        result = await integrator.process_text_with_semantics(test_text, "test")

        print(f"✅ Semantic integrator test passed!")
        print(f"   Entities: {len(result['entities'])}")
        print(f"   Relationships: {len(result['relationships'])}")
        print(f"   Facts: {len(result['facts'])}")

        if result["entities"]:
            print(f"   Sample entity: {result['entities'][0]}")
        if result["facts"]:
            print(f"   Sample fact: {result['facts'][0]['content']}")

        return True

    except Exception as e:
        print(f"❌ Semantic integrator test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_knowledge_graph_connection():
    """Test knowledge graph connection."""
    print("\n🔍 Testing knowledge graph connection...")

    try:
        from synapse_mcp.core.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        await kg.connect()

        print("✅ Knowledge graph connection successful!")

        await kg.close()
        return True

    except Exception as e:
        print(f"❌ Knowledge graph connection failed: {e}")
        return False


async def test_basic_regex_patterns():
    """Test the regex patterns we're using."""
    print("\n🔍 Testing regex patterns...")

    try:
        import re

        # Test patterns from our code
        test_text = "Machine learning algorithms can identify patterns."

        # Test whitespace normalization
        pattern1 = r"\s+"
        result1 = re.sub(pattern1, " ", test_text)
        print(f"✅ Whitespace pattern works: '{result1}'")

        # Test character cleaning
        pattern2 = r"[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/\&]"
        result2 = re.sub(pattern2, "", test_text)
        print(f"✅ Character cleaning pattern works: '{result2}'")

        # Test quote normalization
        pattern3 = r'[""]'
        result3 = re.sub(pattern3, '"', test_text)
        print(f"✅ Quote normalization pattern works: '{result3}'")

        # Test sentence splitting
        pattern4 = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(pattern4, test_text)
        print(f"✅ Sentence splitting pattern works: {sentences}")

        return True

    except Exception as e:
        print(f"❌ Regex pattern test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all debug tests."""
    print("🧠 Project Synapse - Debug Testing")
    print("=" * 50)

    tests = [
        test_basic_regex_patterns,
        test_knowledge_graph_connection,
        test_semantic_integrator,
    ]

    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All debug tests passed ({passed}/{total})!")
    else:
        print(f"⚠️  Some debug tests failed ({passed}/{total})")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
