#!/usr/bin/env python3
"""
Test script to verify the insight engine fixes and add sample data.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from synapse_mcp.core.knowledge_graph import KnowledgeGraph
from synapse_mcp.zettelkasten.insight_engine import InsightEngine


async def test_insight_engine():
    """Test the insight engine with sample data."""
    
    # Initialize knowledge graph
    kg = KnowledgeGraph()
    await kg.connect()
    
    print("📊 Adding sample data to knowledge graph...")
    
    # Add some sample entities and relationships
    sample_data = {
        'entities': [
            {
                'id': 'python',
                'name': 'Python',
                'type': 'Programming Language',
                'confidence': 1.0,
                'source': 'test_data',
                'properties': {
                    'original_label': 'Python',
                    'start_char': 0,
                    'end_char': 6
                }
            },
            {
                'id': 'machine_learning',
                'name': 'Machine Learning',
                'type': 'Technology',
                'confidence': 1.0,
                'source': 'test_data',
                'properties': {
                    'original_label': 'Machine Learning',
                    'start_char': 0,
                    'end_char': 16
                }
            },
            {
                'id': 'neural_networks',
                'name': 'Neural Networks',
                'type': 'Algorithm',
                'confidence': 1.0,
                'source': 'test_data',
                'properties': {
                    'original_label': 'Neural Networks',
                    'start_char': 0,
                    'end_char': 15
                }
            },
            {
                'id': 'tensorflow',
                'name': 'TensorFlow',
                'type': 'Library',
                'confidence': 1.0,
                'source': 'test_data',
                'properties': {
                    'original_label': 'TensorFlow',
                    'start_char': 0,
                    'end_char': 10
                }
            }
        ],
        'relationships': [
            {
                'source_id': 'tensorflow',
                'target_id': 'python',
                'type': 'IMPLEMENTED_IN',
                'confidence': 0.95,
                'source': 'test_data',
                'properties': {
                    'predicate': 'implemented in',
                    'source_span': 'TensorFlow is implemented in Python'
                }
            },
            {
                'source_id': 'tensorflow',
                'target_id': 'machine_learning',
                'type': 'USED_FOR',
                'confidence': 0.9,
                'source': 'test_data',
                'properties': {
                    'predicate': 'used for',
                    'source_span': 'TensorFlow is used for machine learning'
                }
            },
            {
                'source_id': 'neural_networks',
                'target_id': 'machine_learning',
                'type': 'PART_OF',
                'confidence': 0.85,
                'source': 'test_data',
                'properties': {
                    'predicate': 'part of',
                    'source_span': 'Neural networks are part of machine learning'
                }
            }
        ],
        'facts': [
            {
                'id': 'fact_1',
                'content': 'TensorFlow is a popular machine learning framework implemented in Python.',
                'logical_form': 'POPULAR(tensorflow) ∧ ML_FRAMEWORK(tensorflow) ∧ IMPLEMENTED_IN(tensorflow, python)',
                'confidence': 0.9,
                'source': 'test_data',
                'entities': ['tensorflow', 'python', 'machine_learning'],
                'metadata': {
                    'extraction_method': 'manual',
                    'entities': ['tensorflow', 'python', 'machine_learning']
                }
            },
            {
                'id': 'fact_2',
                'content': 'Neural networks are a fundamental component of modern machine learning systems.',
                'logical_form': 'FUNDAMENTAL(neural_networks) ∧ COMPONENT_OF(neural_networks, machine_learning)',
                'confidence': 0.95,
                'source': 'test_data',
                'entities': ['neural_networks', 'machine_learning'],
                'metadata': {
                    'extraction_method': 'manual',
                    'entities': ['neural_networks', 'machine_learning']
                }
            }
        ]
    }
    
    # Store the sample data
    stats = await kg.store_processed_data(sample_data)
    print(f"✅ Stored: {stats['entities_count']} entities, {stats['relationships_count']} relationships, {stats['facts_count']} facts")
    
    # Initialize insight engine
    print("🧠 Initializing insight engine...")
    insight_engine = InsightEngine(kg, None)  # No Montague parser needed for this test
    await insight_engine.initialize()
    
    # Test insight generation
    print("🔍 Generating insights...")
    try:
        insights = await insight_engine.generate_insights(confidence_threshold=0.5)
        print(f"✅ Generated {len(insights)} insights successfully!")
        
        for i, insight in enumerate(insights, 1):
            print(f"\n📝 Insight {i}:")
            print(f"   Title: {insight['title']}")
            print(f"   Confidence: {insight['confidence']:.2f}")
            print(f"   Pattern Type: {insight['pattern_type']}")
            print(f"   Content Preview: {insight['content'][:100]}...")
    
    except Exception as e:
        print(f"❌ Error generating insights: {e}")
        import traceback
        traceback.print_exc()
    
    # Test knowledge query
    print("\n🔍 Testing knowledge queries...")
    try:
        query_results = await kg.query_semantic("TensorFlow Python machine learning")
        print(f"✅ Query returned {len(query_results)} results")
        
        for result in query_results[:2]:  # Show first 2 results
            print(f"   - {result['statement'][:80]}...")
    
    except Exception as e:
        print(f"❌ Error querying knowledge: {e}")
    
    # Get statistics
    print("\n📊 Knowledge Graph Statistics:")
    try:
        stats = await kg.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")
    
    # Cleanup
    await insight_engine.cleanup()
    await kg.close()
    print("\n✅ Test completed!")


if __name__ == "__main__":
    asyncio.run(test_insight_engine())
