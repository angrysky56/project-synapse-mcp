import os

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv() -> None:
        pass


# Load environment configuration from .env file if available
load_dotenv()

from neo4j import GraphDatabase

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")
if not PASSWORD:
    raise ValueError("NEO4J_PASSWORD environment variable is required but not set")
AUTH = (USER, PASSWORD)
DB = os.getenv("NEO4J_DATABASE", "synapse")


def run_query(query, params=None):
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session(database=DB) as session:
        result = session.run(query, params or {})
        return [dict(r) for r in result]
    driver.close()


if __name__ == "__main__":
    orphan_query = "MATCH (e:Entity) OPTIONAL MATCH (e)-[r:RELATES]-() WITH e, count(r) as rel_count WHERE rel_count = 0 RETURN e.type, e.name LIMIT 10"
    orphans = run_query(orphan_query)
    if orphans:
        print("Found orphan entities (no RELATES edges):")
        for orphan in orphans:
            print(f"- Type: {orphan['e.type']}, Name: {orphan['e.name']}")
    else:
        print("No orphan entities found.")
