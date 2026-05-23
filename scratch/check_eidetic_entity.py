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
                "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS 'eidetic' RETURN e.name, e.type"
            )
            records = [r async for r in result]
            print("Entities:", [(r["e.name"], r["e.type"]) for r in records])
        await driver.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
