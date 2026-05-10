import sys
import os
import asyncio
from datetime import datetime
from typing import Any

# Add src to path
sys.path.append(os.path.abspath('src'))

from synapse_mcp.data_pipeline.semantic_integrator import SemanticIntegrator
from synapse_mcp.semantic.montague_parser import MontagueParser

class MockKnowledgeGraph:
    async def store_processed_data(self, data):
        print(f"DEBUG: Storing {len(data['entities'])} entities and {len(data['relationships'])} relationships")
        return {
            "entities_count": len(data["entities"]),
            "relationships_count": len(data["relationships"]),
            "facts_count": len(data["facts"]),
            "new_nodes": len(data["entities"]),
            "new_edges": len(data["relationships"])
        }

async def main():
    parser = MontagueParser()
    await parser.initialize()
    
    integrator = SemanticIntegrator(parser)
    await integrator.initialize()
    
    kg = MockKnowledgeGraph()
    
    text = """
# The Markovian Thinker

Amirhossein Sarath, Aaron, and Sarath are authors.
They developed Delethink using RL and Qwen3.
Quick Start:
1. Run pip install.
Config Files  :
- settings.json
Simply use the tool.
    """
    
    print("--- INGESTING TEXT ---")
    processed_data = await integrator.process_text_with_semantics(text, "test_source", {})
    
    print(f"Entities in processed_data: {len(processed_data['entities'])}")
    for ent in processed_data['entities']:
        print(f"  {ent['name']} ({ent['type']})")
        
    print(f"Relationships in processed_data: {len(processed_data['relationships'])}")
    for rel in processed_data['relationships']:
        print(f"  {rel['source_id']} --[{rel['type']}]--> {rel['target_id']} ({rel['properties'].get('predicate')})")
        
    result = await kg.store_processed_data(processed_data)
    print(f"Final Result: {result}")

if __name__ == '__main__':
    asyncio.run(main())
