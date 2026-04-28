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
            result = await session.run("MATCH (a:Entity)-[r]->(b:Entity) RETURN a.name, type(r), b.name LIMIT 10")
            records = [r async for r in result]
            for r in records:
                print(f"{r['a.name']} -[{r['type(r)']}]-> {r['b.name']}")
        await driver.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
