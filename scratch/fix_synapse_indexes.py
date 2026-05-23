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
            print(f"Creating indexes in database '{database}'...")
            await session.run(
                "CREATE FULLTEXT INDEX fact_fulltext IF NOT EXISTS FOR (n:Fact) ON EACH [n.content]"
            )
            await session.run(
                "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.type]"
            )

            # Also check if any Fact nodes exist
            result = await session.run("MATCH (n:Fact) RETURN count(n) as c")
            record = await result.single()
            print(f"Fact nodes in '{database}': {record['c']}")

            # Check source for test_ingest.md
            result = await session.run(
                "MATCH (n) WHERE n.source = 'raw/test_ingest.md' RETURN labels(n) as l, n.name as n, n.content as c"
            )
            records = [r async for r in result]
            print(f"Nodes from test_ingest.md: {len(records)}")
            for r in records:
                print(f"  Label: {r['l']}, Name/Content: {r['n'] or r['c']}")

        await driver.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
