import asyncio
from synapse_mcp.semantic.montague_parser import MontagueParser

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
    print("Entities:")
    for ent in res.get("entities", []):
        print(f"  - {ent['text']} ({ent['type']})")
        
    print("Relations:")
    for rel in res.get("relations", []):
        print(f"  - {rel['subject']} [{rel['predicate']}] {rel['object']}")

if __name__ == "__main__":
    asyncio.run(main())
