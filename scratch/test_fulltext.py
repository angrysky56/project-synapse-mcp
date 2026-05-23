import asyncio
import os

from neo4j import AsyncGraphDatabase


async def main():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "synapse_password")

    try:
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        async with driver.session() as session:
            print("Testing fact_fulltext query...")
            result = await session.run(
                "CALL db.index.fulltext.queryNodes('fact_fulltext', 'catastrophic') YIELD node, score RETURN node.content, score"
            )
            records = [r async for r in result]
            print(f"Results: {len(records)}")
            for r in records:
                print(f"Score: {r['score']}, Content: {r['node.content'][:50]}...")
        await driver.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
