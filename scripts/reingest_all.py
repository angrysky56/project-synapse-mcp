#!/usr/bin/env python3
"""
Full re-ingest of all LLM-WIKI clippings through the patched Synapse pipeline.
Wipes the Neo4j graph first, then processes all clipping files.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from collections import Counter

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from neo4j import AsyncGraphDatabase

from synapse_mcp.data_pipeline.semantic_integrator import SemanticIntegrator
from synapse_mcp.core.knowledge_graph import KnowledgeGraph

wiki_vault_path = os.getenv("WIKI_VAULT_PATH")
if wiki_vault_path:
    CLIPPINGS_DIR = Path(wiki_vault_path) / "Clippings"
else:
    CLIPPINGS_DIR = Path.home() / "Documents" / "LLM-WIKI" / "Clippings"

DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "synapse_password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "synapse")


async def wipe_graph(driver) -> int:
    """Wipe all nodes and edges. Returns count of deleted nodes."""
    async with driver.session(database=NEO4J_DATABASE) as session:
        result = await session.run("MATCH (n) DETACH DELETE n RETURN count(n) AS deleted")
        records = await result.data()
        return records[0]["deleted"] if records else 0


async def read_stats(driver) -> dict:
    """Read graph statistics."""
    stats = {}
    async with driver.session(database=NEO4J_DATABASE) as session:
        # Entity type distribution
        result = await session.run(
            "MATCH (e:Entity) RETURN e.type AS type, count(*) AS count ORDER BY count DESC"
        )
        stats["entity_types"] = await result.data()

        # Relationship type distribution
        result = await session.run(
            "MATCH ()-[r]->() RETURN r.type AS type, count(*) AS count ORDER BY count DESC LIMIT 20"
        )
        stats["rel_types"] = await result.data()

        # Total counts
        result = await session.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC")
        stats["totals"] = await result.data()

    return stats


async def ingest_file(
    integrator: SemanticIntegrator,
    kg: KnowledgeGraph,
    filepath: Path,
    idx: int,
    total: int,
) -> dict:
    """Ingest a single clipping file through the pipeline."""
    source_name = filepath.stem
    t0 = time.time()

    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
        # Strip YAML frontmatter if present
        if text.startswith("---"):
            end = text.find("---", 3)
            if end >= 0:
                text = text[end + 3 :].strip()

        if len(text) < 50:
            return {"file": source_name, "status": "skipped", "reason": "too short"}

        # Process through the patched pipeline
        processed = await integrator.process_text_with_semantics(
            text,
            source_name,
            {"type": _detect_type(filepath)},
        )

        if not DRY_RUN:
            # Store in Neo4j
            await kg.store_processed_data(processed)

        elapsed = time.time() - t0
        n_ent = len(processed.get("entities", []))
        n_rel = len(processed.get("relationships", []))
        n_fact = len(processed.get("facts", []))

        print(
            f"  [{idx:3d}/{total}] {source_name[:50]:50s}  "
            f"{n_ent:4d} ent  {n_rel:4d} rel  {n_fact:4d} fact  "
            f"{elapsed:.1f}s"
        )

        return {
            "file": source_name,
            "status": "ok",
            "entities": n_ent,
            "relationships": n_rel,
            "facts": n_fact,
            "seconds": elapsed,
        }

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{idx:3d}/{total}] {source_name[:50]:50s}  ERROR: {e}")
        return {"file": source_name, "status": "error", "error": str(e), "seconds": elapsed}


def _detect_type(filepath: Path) -> str:
    """Infer source type from the Clippings subfolder."""
    parts = filepath.parts
    if "papers" in parts:
        return "paper"
    elif "repositories" in parts:
        return "repository"
    elif "documentation" in parts:
        return "documentation"
    else:
        return "article"


def collect_clippings() -> list[Path]:
    """Find all .md files in the Clippings directory."""
    files = sorted(CLIPPINGS_DIR.rglob("*.md"))
    return files


async def main():
    files = collect_clippings()
    print(f"Found {len(files)} clipping files to re-ingest")
    print(f"Dry run: {DRY_RUN}")
    print()

    # Connect to Neo4j
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    await driver.verify_connectivity()
    print(f"Connected to Neo4j at {NEO4J_URI}, database: {NEO4J_DATABASE}")

    # Initialize Synapse pipeline
    integrator = SemanticIntegrator()
    await integrator.initialize()
    kg = KnowledgeGraph()
    await kg.connect()

    # Step 1: Wipe the graph
    if not DRY_RUN:
        print("\nWiping existing graph...")
        deleted = await wipe_graph(driver)
        print(f"  Deleted {deleted:,} nodes\n")
    else:
        print("\n(Dry run — skipping graph wipe)\n")

    # Step 2: Ingest all files
    print("Re-ingesting all clippings through patched pipeline...")
    print("-" * 80)
    results = []
    t_total = time.time()

    for idx, filepath in enumerate(files, 1):
        result = await ingest_file(integrator, kg, filepath, idx, len(files))
        results.append(result)

    t_total = time.time() - t_total

    # Step 3: Report
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]
    skipped = [r for r in results if r["status"] == "skipped"]

    total_ent = sum(r.get("entities", 0) for r in ok)
    total_rel = sum(r.get("relationships", 0) for r in ok)
    total_fact = sum(r.get("facts", 0) for r in ok)

    print(f"\n{'='*60}")
    print(f"RE-INGEST COMPLETE")
    print(f"{'='*60}")
    print(f"Files: {len(ok)} ok, {len(errors)} errors, {len(skipped)} skipped")
    print(f"Total entities:      {total_ent:,}")
    print(f"Total relationships: {total_rel:,}")
    print(f"Total facts:         {total_fact:,}")
    print(f"Total time:          {t_total:.1f}s ({t_total/60:.1f}m)")
    print(f"Avg per file:        {t_total/max(len(files),1):.1f}s")

    # Graph stats
    if not DRY_RUN:
        print(f"\n--- Neo4j Graph Statistics ---")
        stats = await read_stats(driver)

        print("\nNode totals:")
        for row in stats["totals"]:
            print(f"  {row['label']:20s} {row['count']:,}")

        print("\nEntity type distribution:")
        for row in stats["entity_types"]:
            print(f"  {row['type']:20s} {row['count']:,}")

        print("\nRelationship type distribution:")
        for row in stats["rel_types"]:
            rtype = row["type"] if row["type"] else "(NULL)"
            print(f"  {rtype:20s} {row['count']:,}")

    if errors:
        print(f"\nErrors:")
        for e in errors:
            print(f"  {e['file']}: {e.get('error', 'unknown')}")

    await kg.close()
    await driver.close()


if __name__ == "__main__":
    asyncio.run(main())