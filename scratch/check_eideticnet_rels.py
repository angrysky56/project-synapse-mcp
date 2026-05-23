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
    database = os.getenv("NEO4J_DATABASE", "synapse")

    try:
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        async with driver.session(database=database) as session:
            result = await session.run(
                "MATCH (a:Entity)-[r]->(b:Entity) WHERE a.name = 'EideticNet' OR b.name = 'EideticNet' RETURN a.name, type(r), b.name"
            )
            records = [r async for r in result]
            for r in records:
                print(f"{r['a.name']} -[{r['type(r)']}]-> {r['b.name']}")
        await driver.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
