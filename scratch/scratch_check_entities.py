
import asyncio
import os
from neo4j import AsyncGraphDatabase

async def check():
    uri = "bolt://localhost:7687"
    auth = ("neo4j", "00000000")
    driver = AsyncGraphDatabase.driver(uri, auth=auth)
    session = driver.session(database="synapse")
    result = await session.run("MATCH (e:Entity) RETURN count(e)")
    print(f"Total entities in 'synapse': {await result.single()}")
    await driver.close()

if __name__ == "__main__":
    asyncio.run(check())
