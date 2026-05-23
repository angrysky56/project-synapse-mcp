import asyncio
from synapse_mcp.data_pipeline.semantic_integrator import SemanticIntegrator

async def main():
    integrator = SemanticIntegrator()
    await integrator.initialize()
    text = "The Markovian Thinker was published by DeepMind. Qwen3 uses Reinforcement Learning. RL is a concept."
    
    res = await integrator.process_text_with_semantics(text, source="test")
    
    print("Graph Entities:")
    for e in res["entities"]:
        print(f"  - {e['name']} ({e['type']})")
        
    print("\nGraph Relations:")
    for r in res["relationships"]:
        print(f"  - {r['source_id']} -> {r['type']} -> {r['target_id']}")

asyncio.run(main())
