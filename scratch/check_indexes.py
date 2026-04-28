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
            result = await session.run("SHOW INDEXES")
            records = [r async for r in result]
            for r in records:
                if 'fulltext' in r['name']:
                    print(f"Index: {r['name']}, State: {r['state']}, Type: {r['type']}")
        await driver.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
