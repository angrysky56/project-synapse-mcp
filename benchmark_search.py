import asyncio
import os

from dotenv import load_dotenv

from synapse_mcp.core.knowledge_graph import KnowledgeGraph
from synapse_mcp.utils.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)


async def test_search():
    kg = KnowledgeGraph()
    await kg.connect()

    query = "Eidetic Learning in neural networks"
    print(f"Searching for: '{query}'")

    results = await kg.query_hybrid(query, max_results=5)

    print(f"\nFound {len(results)} results:")
    for i, res in enumerate(results, start=1):
        sources = ", ".join(res.get("retrieval_sources", []))
        print(f"{i}. [{res['rrf_score']:.4f}] [{sources}] {res['statement'][:100]}...")

    await kg.close()


if __name__ == "__main__":
    asyncio.run(test_search())
