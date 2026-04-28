import asyncio
import os
from neo4j import AsyncGraphDatabase

async def main():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "00000000"
    database = "synapse"
    
    try:
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        async with driver.session(database=database) as session:
            result = await session.run("MATCH (e:Entity) WHERE toLower(e.name) CONTAINS 'eidetic' RETURN e.name, e.type")
            records = [r async for r in result]
            print("Entities:", [(r['e.name'], r['e.type']) for r in records])
        await driver.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
