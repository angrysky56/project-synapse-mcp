import asyncio
from neo4j import AsyncGraphDatabase

async def main():
    driver = AsyncGraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "00000000"))
    async with driver.session() as session:
        # Count all nodes
        result = await session.run("MATCH (n) RETURN count(n) as c")
        total_nodes = (await result.single())["c"]
        print(f"Total nodes: {total_nodes}")
        
        result = await session.run("MATCH (e:Entity) RETURN count(e) as c")
        entity_nodes = (await result.single())["c"]
        print(f"Total Entity nodes: {entity_nodes}")
        
        result = await session.run("MATCH ()-[r]->() RETURN type(r) as t, count(r) as c")
        print("Relationships:")
        async for record in result:
            print(f"  {record['t']}: {record['c']}")
            
    await driver.close()

asyncio.run(main())
