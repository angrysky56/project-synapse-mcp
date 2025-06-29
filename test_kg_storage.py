#!/usr/bin/env python3
"""
Test knowledge graph storage directly to isolate the regex error.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def test_knowledge_graph_storage():
    """Test knowledge graph storage with simple data."""
    print("🔍 Testing knowledge graph storage...")
    
    try:
        from synapse_mcp.core.knowledge_graph import KnowledgeGraph
        
        # Test connection
        print("Step 1: Connecting to knowledge graph...")
        kg = KnowledgeGraph()
        await kg.connect()
        print("✅ Connection successful")
        
        # Create simple test data (mimicking what semantic integrator would create)
        print("Step 2: Creating test data...")
        test_data = {
            'text_id': 'test_123',
            'source': 'test',
            'metadata': {'test': True},
            'entities': [
                {
                    'id': 'test_entity',
                    'name': 'Test Entity',
                    'type': 'Entity',
                    'confidence': 0.9,
                    'source': 'test',
                    'properties': {}
                }
            ],
            'relationships': [],
            'facts': [
                {
                    'id': 'test_fact',
                    'content': 'This is a test fact.',
                    'logical_form': '',
                    'confidence': 0.9,
                    'source': 'test',
                    'metadata': {},
                    'entities': ['test_entity']
                }
            ]
        }
        print("✅ Test data created")
        
        # Test storage
        print("Step 3: Storing test data...")
        result = await kg.store_processed_data(test_data)
        print("✅ Storage successful!")
        print(f"   Storage stats: {result}")
        
        await kg.close()
        return True
        
    except Exception as e:
        print(f"❌ Knowledge graph storage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the knowledge graph storage test."""
    print("🧠 Project Synapse - Knowledge Graph Storage Test")
    print("=" * 50)
    
    success = await test_knowledge_graph_storage()
    
    if success:
        print("\n✅ Knowledge graph storage works!")
        print("The issue must be elsewhere in the integration.")
    else:
        print("\n❌ Found the source of the error!")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
