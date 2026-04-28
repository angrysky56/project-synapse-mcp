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
            print("Creating fact_fulltext index...")
            await session.run("CREATE FULLTEXT INDEX fact_fulltext IF NOT EXISTS FOR (n:Fact) ON EACH [n.content]")
            print("Creating entity_fulltext index...")
            await session.run("CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.type]")
            print("Indexes created/verified.")
        await driver.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
