import asyncio
from synapse_mcp.data_pipeline.text_processor import TextProcessor
from synapse_mcp.semantic.montague_parser import MontagueParser
from synapse_mcp.data_pipeline.semantic_integrator import SemanticIntegrator

async def main():
    text = """
    The Markovian Thinker
    LLM Wiki
    Amirhossein Kazemnejad, Sarath Chandar, Aaron Courville
    RL and Qwen3
    Quick Start
    Config Files
    Delethink
    Simply put, The Markovian Thinker is a technique that uses RL to improve Qwen3.
    """
    
    mp = MontagueParser()
    await mp.initialize()
    res = await mp.parse_text(text)
    
    si = SemanticIntegrator(None)
    processed = {
        "entities": [],
        "relationships": [],
        "facts": [],
        "metadata": {}
    }
    await si._integrate_semantic_analysis(res, "test.md", processed)
    
    print(f"Rels count: {len(processed['relationships'])}")
    for rel in processed["relationships"]:
        print(f"Rel: {rel}")

if __name__ == "__main__":
    asyncio.run(main())
