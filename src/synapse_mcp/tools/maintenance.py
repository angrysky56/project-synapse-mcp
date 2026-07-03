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


async def reembed_degraded(dry_run: bool = False) -> None:
    """Find and re-embed nodes whose vectors came from the local fallback.

    The local sentence-transformers fallback produces 384-dim vectors that
    get zero-padded to EMBEDDING_DIMENSION. Real provider vectors (qwen3
    2560-dim) essentially never contain a long run of exact zeros, so a
    node whose embedding tail beyond index 1024 is all zeros was embedded
    by the fallback (or padded from another short model) and is invisible
    to semantic search. This re-embeds those nodes with the configured
    provider. Also repairs nodes with a missing embedding.

    Run only while the embedding provider is healthy.
    """
    from synapse_mcp.core.knowledge_graph import KnowledgeGraph

    # (label, id property, cypher expression producing the text to embed)
    targets = [
        ("Fact", "id", "n.content"),
        ("Entity", "id", "n.name + ' (' + coalesce(n.type, '') + ')'"),
        ("Zettel", "id", "n.content"),
    ]

    kg = KnowledgeGraph()
    await kg.connect()
    assert kg.driver is not None

    total_fixed = 0
    try:
        async with kg.driver.session(database=kg.database) as session:
            for label, id_prop, text_expr in targets:
                find_query = f"""
                MATCH (n:{label})
                WHERE n.embedding IS NULL
                   OR all(x IN n.embedding[1024..] WHERE x = 0.0)
                RETURN n.{id_prop} AS id, {text_expr} AS text
                """
                result = await session.run(find_query)  # type: ignore[arg-type]
                rows = await result.data()
                print(f"{label}: {len(rows)} degraded/missing embeddings found")

                if dry_run:
                    continue

                for row in rows:
                    if not row["text"]:
                        continue
                    vec = await kg._embed_text(row["text"])
                    await session.run(
                        f"MATCH (n:{label} {{{id_prop}: $id}}) "  # type: ignore[arg-type]
                        "CALL db.create.setNodeVectorProperty(n, 'embedding', $vec)",
                        {"id": row["id"], "vec": vec},
                    )
                    total_fixed += 1

        if kg._fallback_embed_count > 0:
            print(
                f"WARNING: the fallback embedder fired {kg._fallback_embed_count} "
                "times during re-embedding — the provider is still unhealthy. "
                "Those nodes remain degraded; re-run later."
            )
        print(f"Re-embedded {total_fixed} nodes." if not dry_run else "Dry run only.")
    finally:
        await kg.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Synapse maintenance tasks")
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="Re-embed nodes stored with degraded fallback vectors",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report degraded nodes, change nothing"
    )
    parser.add_argument(
        "--resolve-entities",
        action="store_true",
        help="Retroactively refine entity types",
    )
    args = parser.parse_args()

    if args.reembed:
        asyncio.run(reembed_degraded(dry_run=args.dry_run))
    elif args.resolve_entities:
        asyncio.run(resolve_entities())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
