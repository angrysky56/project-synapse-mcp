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
            result = await session.run(
                "MATCH (n) RETURN labels(n) as labels, count(*) as count"
            )
            records = [r async for r in result]
            for r in records:
                print(f"Labels: {r['labels']}, Count: {r['count']}")
        await driver.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
