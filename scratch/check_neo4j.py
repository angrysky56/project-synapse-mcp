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
            result = await session.run("MATCH (n:Entity) RETURN n.name LIMIT 10")
            records = [r async for r in result]
            print("Entities:", [r['n.name'] for r in records])
            
            result = await session.run("SHOW INDEXES")
            indexes = [r async for r in result]
            print("Indexes:", [i['name'] for i in indexes])
            
            # Try to create the missing index if it doesn't exist
            if 'fact_fulltext' not in [i['name'] for i in indexes]:
                print("Creating fact_fulltext index...")
                await session.run("CALL db.index.fulltext.createNodeIndex('fact_fulltext', ['Fact'], ['content'], {analyzer: 'standard-no-stop-words'})")
                print("Index created.")
                
        await driver.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
