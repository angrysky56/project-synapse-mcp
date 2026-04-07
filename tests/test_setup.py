#!/usr/bin/env python3
"""
Quick test script to verify Project Synapse setup and basic functionality.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def test_basic_imports():
    """Test that all core modules can be imported."""
    print("🧪 Testing basic imports...")
    
    try:
        from synapse_mcp.utils.logging_config import setup_logging
        print("✅ Logging config import OK")
        
        from synapse_mcp.data_pipeline.text_processor import TextProcessor
        print("✅ Text processor import OK")
        
        from synapse_mcp.semantic.montague_parser import MontagueParser
        print("✅ Montague parser import OK")
        
        from synapse_mcp.core.knowledge_graph import KnowledgeGraph
        print("✅ Knowledge graph import OK")
        
        from synapse_mcp.zettelkasten.insight_engine import InsightEngine
        print("✅ Insight engine import OK")
        
        from synapse_mcp.knowledge.knowledge_types import Entity, Fact, Insight
        print("✅ Knowledge types import OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


async def test_text_processor():
    """Test text processor basic functionality."""
    print("\n🧪 Testing text processor...")
    
    try:
        from synapse_mcp.data_pipeline.text_processor import TextProcessor
        
        processor = TextProcessor()
        await processor.initialize()
        
        # Test basic text processing
        test_text = "This is a test sentence. It contains some information."
        result = await processor.process_text(test_text, "test")
        
        assert 'text_id' in result
        assert 'sentences' in result
        assert 'facts' in result
        assert len(result['facts']) > 0
        
        print("✅ Text processor functionality OK")
        return True
        
    except Exception as e:
        print(f"❌ Text processor test failed: {e}")
        return False


async def test_montague_parser():
    """Test Montague parser without spaCy if not available."""
    print("\n🧪 Testing Montague parser...")
    
    try:
        from synapse_mcp.semantic.montague_parser import MontagueParser
        
        parser = MontagueParser()
        
        # Test if spaCy is available
        try:
            await parser.initialize()
            
            # Test basic parsing
            test_text = "John likes apples."
            result = await parser.parse_text(test_text)
            
            assert 'entities' in result
            assert 'relations' in result
            assert 'logical_form' in result
            
            print("✅ Montague parser with spaCy OK")
            
        except OSError:
            print("⚠️  spaCy model not available - run: uv run python -m spacy download en_core_web_sm")
            print("✅ Montague parser structure OK")
            
        return True
        
    except Exception as e:
        print(f"❌ Montague parser test failed: {e}")
        return False


async def test_knowledge_types():
    """Test knowledge type classes."""
    print("\n🧪 Testing knowledge types...")
    
    try:
        from synapse_mcp.knowledge.knowledge_types import Entity, Fact, Insight, KnowledgeUtils
        
        # Test entity creation
        entity = Entity(
            id="test_entity",
            name="Test Entity",
            type="Test",
            confidence=0.9
        )
        
        assert entity.id == "test_entity"
        assert entity.confidence == 0.9
        
        # Test fact creation
        fact = Fact(
            id="test_fact",
            content="This is a test fact",
            confidence=0.8
        )
        
        assert fact.content == "This is a test fact"
        
        # Test utilities
        entity_id = KnowledgeUtils.generate_entity_id("Test Name", "Person")
        assert entity_id.startswith("person_")
        
        print("✅ Knowledge types functionality OK")
        return True
        
    except Exception as e:
        print(f"❌ Knowledge types test failed: {e}")
        return False


async def test_server_structure():
    """Test that server can be imported and structured correctly."""
    print("\n🧪 Testing server structure...")
    
    try:
        from synapse_mcp.server import mcp, SynapseServer
        
        # Check that MCP server is configured
        assert mcp is not None
        assert hasattr(mcp, 'run')
        
        # Check that server class exists
        server = SynapseServer()
        assert hasattr(server, 'initialize')
        assert hasattr(server, 'cleanup')
        
        print("✅ Server structure OK")
        return True
        
    except Exception as e:
        print(f"❌ Server structure test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧠 Project Synapse - Quick Setup Test")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_text_processor,
        test_montague_parser,
        test_knowledge_types,
        test_server_structure
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
        print(f"🎉 All tests passed ({passed}/{total})!")
        print("\n✅ Project Synapse is ready to run")
        print("\nNext steps:")
        print("1. Start Neo4j: sudo systemctl start neo4j")
        print("2. Configure .env file")
        print("3. Run server: uv run python -m synapse_mcp.server")
        return 0
    else:
        print(f"⚠️  Some tests failed ({passed}/{total})")
        print("\nPlease fix the issues above before running the server.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
