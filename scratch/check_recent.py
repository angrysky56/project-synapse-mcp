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
            print("Most recent 10 nodes:")
            result = await session.run(
                "MATCH (n) RETURN labels(n) as l, properties(n) as p ORDER BY n.created_at DESC LIMIT 10"
            )
            records = [r async for r in result]
            for r in records:
                label = r["l"][0] if r["l"] else "NoLabel"
                name = r["p"].get("name") or r["p"].get("content") or r["p"].get("id")
                print(f"Label: {label}, Name/Content: {str(name)[:100]}")
        await driver.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
