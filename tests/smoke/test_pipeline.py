import asyncio
from synapse_mcp.data_pipeline.semantic_integrator import SemanticIntegrator

async def main():
    text = """
    # The Markovian Thinker
    LLM Wiki
    Amirhossein Kazemnejad, Sarath Chandar, Aaron Courville
    RL and Qwen3
    ## Quick Start
    ### Config Files
    Delethink
    Simply put, The Markovian Thinker is a technique that uses RL to improve Qwen3.
    """
    
    integrator = SemanticIntegrator()
    await integrator.initialize()
    
    res = await integrator.process_text_with_semantics(text, source="test")
    
    print("Graph Entities:")
    for e in res["entities"]:
        print(f"  - {e['name']} ({e['type']})")
        
    print("\nGraph Relations:")
    for r in res["relationships"]:
        print(f"  - {r['source_id']} -> {r['type']} -> {r['target_id']}")

if __name__ == "__main__":
    asyncio.run(main())
