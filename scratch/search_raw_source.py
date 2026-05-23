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
            print("Searching for nodes with source containing 'raw/'...")
            result = await session.run(
                "MATCH (n) WHERE n.source CONTAINS 'raw/' RETURN labels(n) as l, n.name as n, n.content as c, n.source as s LIMIT 10"
            )
            records = [r async for r in result]
            print(f"Results: {len(records)}")
            for r in records:
                print(f"Label: {r['l']}, Name: {r['n']}, Source: {r['s']}")
        await driver.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
