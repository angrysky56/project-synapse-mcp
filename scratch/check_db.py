import asyncio
import os

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv() -> None:
        pass


# Load environment configuration from .env file if available
load_dotenv()

from neo4j import AsyncGraphDatabase


async def main():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError("NEO4J_PASSWORD environment variable is required but not set")

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    async with driver.session() as session:
        # Count all nodes
        result = await session.run("MATCH (n) RETURN count(n) as c")
        total_nodes = (await result.single())["c"]
        print(f"Total nodes: {total_nodes}")

        result = await session.run("MATCH (e:Entity) RETURN count(e) as c")
        entity_nodes = (await result.single())["c"]
        print(f"Total Entity nodes: {entity_nodes}")

        result = await session.run(
            "MATCH ()-[r]->() RETURN type(r) as t, count(r) as c"
        )
        print("Relationships:")
        async for record in result:
            print(f"  {record['t']}: {record['c']}")

    await driver.close()


asyncio.run(main())
