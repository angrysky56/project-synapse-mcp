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
            print(f"Creating indexes in database '{database}'...")
            await session.run("CREATE FULLTEXT INDEX fact_fulltext IF NOT EXISTS FOR (n:Fact) ON EACH [n.content]")
            await session.run("CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.type]")
            
            # Also check if any Fact nodes exist
            result = await session.run("MATCH (n:Fact) RETURN count(n) as c")
            record = await result.single()
            print(f"Fact nodes in '{database}': {record['c']}")
            
            # Check source for test_ingest.md
            result = await session.run("MATCH (n) WHERE n.source = 'raw/test_ingest.md' RETURN labels(n) as l, n.name as n, n.content as c")
            records = [r async for r in result]
            print(f"Nodes from test_ingest.md: {len(records)}")
            for r in records:
                print(f"  Label: {r['l']}, Name/Content: {r['n'] or r['c']}")
                
        await driver.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
