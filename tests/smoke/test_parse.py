import asyncio
from synapse_mcp.semantic.montague_parser import MontagueParser

async def main():
    parser = MontagueParser()
    await parser.initialize()
    text = "The Markovian Thinker was published by DeepMind. Qwen3 uses Reinforcement Learning. RL is a concept."
    res = await parser.parse_text(text)
    print("Entities:", [e['text'] for e in res["entities"]])
    print("Relations:", res["relations"])

asyncio.run(main())
