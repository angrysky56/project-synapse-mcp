import asyncio
import os
import sys
from pathlib import Path

# Add the 'src' directory to sys.path to allow absolute imports when run as a script
src_dir = Path(__file__).resolve().parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

from synapse_mcp.semantic.montague_parser import MontagueParser
from synapse_mcp.utils.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)


async def resolve_entities() -> None:
    """Retroactively refine entity types in the Neo4j knowledge graph."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "synapse_password")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    logger.info(f"Connecting to Neo4j at {uri} (database: {database})")

    try:
        async with AsyncGraphDatabase.driver(uri, auth=(user, password)) as driver:
            async with driver.session(database=database) as session:
                # 1. Fetch all entities
                result = await session.run("MATCH (e:Entity) RETURN e")
                records = await result.data()

                entities = [dict(r["e"]) for r in records]

                print(f"Found {len(entities)} entities to check")

                updates = []
                for ent in entities:
                    name = ent.get("name", "")
                    etype = ent.get("type", "Entity")
                    eid = ent.get("id")

                    if not eid:
                        continue

                    new_type = MontagueParser._refine_entity_type(name, etype)
                    if new_type != etype:
                        print(f"Match found: {name} ({etype}) -> {new_type}")
                        updates.append((eid, new_type, etype))

                print(f"Identified {len(updates)} entities for type refinement")

                # 2. Apply updates
                for eid, new_type, old_type in updates:
                    await session.run(
                        "MATCH (e:Entity {id: $id}) SET e.type = $type, e.updated_at = timestamp()",
                        {"id": eid, "type": new_type},
                    )
                    print(f"Updated entity {eid}: {old_type} -> {new_type}")

                print(f"Refinement complete. Updated {len(updates)} entities.")

    except Exception as e:
        logger.error(f"Maintenance failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(resolve_entities())
