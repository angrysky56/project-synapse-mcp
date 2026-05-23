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


async def check():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError("NEO4J_PASSWORD environment variable is required but not set")
    auth = (user, password)

    driver = AsyncGraphDatabase.driver(uri, auth=auth)
    session = driver.session(database=os.getenv("NEO4J_DATABASE", "synapse"))
    result = await session.run("MATCH (e:Entity) RETURN count(e)")
    print(f"Total entities in 'synapse': {await result.single()}")
    await driver.close()


if __name__ == "__main__":
    asyncio.run(check())
