import asyncio
import spacy
from synapse_mcp.semantic.montague_parser import MontagueParser

async def test():
    parser = MontagueParser()
    await parser.initialize()
    
    text = "The Markovian Thinker modifies the Qwen3 model to use reinforcement learning."
    doc = parser.nlp(text)
    
    relations = await parser._extract_relations(doc)
    print("Extracted relations:")
    for r in relations:
        print(r)

if __name__ == "__main__":
    asyncio.run(test())
