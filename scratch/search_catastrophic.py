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
            print("Searching for 'catastrophic' in Fact nodes...")
            result = await session.run("MATCH (f:Fact) WHERE toLower(f.content) CONTAINS 'catastrophic' RETURN f.content LIMIT 5")
            records = [r async for r in result]
            print(f"Results: {len(records)}")
            for r in records:
                print(f"Content: {r['f.content'][:100]}...")
        await driver.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
