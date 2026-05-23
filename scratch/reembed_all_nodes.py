#!/usr/bin/env python3
"""
Re-embed all old 768-dim nodes in Neo4j with Qwen 4B 2560-dim embeddings.

Run from the project-synapse-mcp directory:
    cd /path/to/your/workspace/project-synapse-mcp
    uv run python scratch/reembed_all_nodes.py

Uses the same embedding logic as knowledge_graph.py:
  - Entity:  f"{name} ({type})"
  - Fact:    content property

Batches 50 texts per Ollama API call. Dry-run with --dry-run first.
"""

import argparse
import asyncio
import os
import sys
import time

import aiohttp
from neo4j import AsyncGraphDatabase

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD environment variable is required but not set")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "synapse")
BATCH_SIZE = 50
NEW_DIM = 2560
OLD_DIM = 768


async def ollama_embed_batch(
    session: aiohttp.ClientSession, texts: list[str]
) -> list[list[float]]:
    """Send a batch of texts to Ollama for embedding."""
    url = f"{OLLAMA_URL}/api/embed"
    payload = {"model": OLLAMA_MODEL, "input": texts}
    async with session.post(
        url, json=payload, timeout=aiohttp.ClientTimeout(total=120)
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"Ollama returned {resp.status}: {body[:500]}")
        data = await resp.json()
        return data["embeddings"]


async def get_old_nodes(driver, label: str) -> list[dict]:
    """Fetch all nodes with old 768-dim embeddings."""
    async with driver.session(database=NEO4J_DATABASE) as s:
        result = await s.run(
            f"MATCH (n:{label}) WHERE size(n.embedding) = {OLD_DIM} "
            "RETURN elementId(n) as eid, n.name as name, n.type as type, n.content as content"
        )
        records = [r async for r in result]
    return records


async def update_node_embedding(driver, eid: str, embedding: list[float]):
    """Update a single node's embedding via its elementId."""
    async with driver.session(database=NEO4J_DATABASE) as s:
        await s.run(
            "MATCH (n) WHERE elementId(n) = $eid "
            "CALL db.create.setNodeVectorProperty(n, 'embedding', $vec) "
            "RETURN n",
            {"eid": eid, "vec": embedding},
        )


async def reembed_label(
    driver, session, label: str, dry_run: bool = False, batch_size: int = 50
):
    """Re-embed all old nodes of a given label."""
    print(f"\n{'='*60}")
    print(f"🔍 Fetching {label} nodes with {OLD_DIM}-dim embeddings...")
    nodes = await get_old_nodes(driver, label)
    total = len(nodes)
    print(f"   Found {total} nodes to re-embed")

    if not nodes:
        print(f"   ✅ No work needed for {label}")
        return

    # Build embedding text for each node
    texts = []
    for n in nodes:
        if label == "Entity":
            texts.append(f"{n.get('name', '')} ({n.get('type', '')})")
        elif label == "Fact":
            texts.append(n.get("content", ""))
        else:
            texts.append(n.get("content", n.get("text", "")))

    if dry_run:
        print(
            f"   🏴 DRY RUN — would embed {total} nodes in {((total + batch_size - 1) // batch_size)} batches"
        )
        print(f"   Sample text[0]: {texts[0][:120]}...")
        return

    # Process in batches
    batches = [texts[i : i + batch_size] for i in range(0, total, batch_size)]
    node_batches = [nodes[i : i + batch_size] for i in range(0, total, batch_size)]

    updated = 0
    start = time.time()
    for batch_idx, (batch_texts, batch_nodes) in enumerate(zip(batches, node_batches)):
        try:
            embeddings = await ollama_embed_batch(session, batch_texts)
            for node, emb in zip(batch_nodes, embeddings):
                if len(emb) != NEW_DIM:
                    print(
                        f"   ⚠️  Unexpected embedding dim {len(emb)} for node {node['eid'][:12]}..."
                    )
                    continue
                await update_node_embedding(driver, node["eid"], emb)
                updated += 1
        except Exception as e:
            print(f"   ❌ Batch {batch_idx} failed: {e}")
            continue

        elapsed = time.time() - start
        rate = updated / elapsed if elapsed > 0 else 0
        pct = (updated / total) * 100
        eta = (total - updated) / rate if rate > 0 else 0
        print(
            f"   [{batch_idx+1}/{len(batches)}] {updated}/{total} ({pct:.0f}%) — {rate:.0f} nodes/s — ETA {eta:.0f}s",
            end="\r",
        )

    elapsed = time.time() - start
    print(
        f"\n   ✅ {label}: {updated}/{total} re-embedded in {elapsed:.0f}s ({updated/elapsed:.0f} nodes/s)"
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Re-embed old 768-dim nodes with Qwen 4B 2560-dim embeddings"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Count nodes but don't embed"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["Entity", "Fact"],
        help="Labels to re-embed (default: Entity Fact)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()

    # Verify Ollama is reachable
    print(f"🔌 Checking Ollama at {OLLAMA_URL}...")
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{OLLAMA_URL}/api/tags", timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                tags = await resp.json()
                models = [m["name"] for m in tags.get("models", [])]
                if OLLAMA_MODEL not in models:
                    print(
                        f"⚠️  Model '{OLLAMA_MODEL}' not in {models}. Available: {models}"
                    )
                else:
                    print(f"   ✅ {OLLAMA_MODEL} available")
    except Exception as e:
        print(f"❌ Cannot reach Ollama: {e}")
        sys.exit(1)

    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    async with aiohttp.ClientSession() as http_session:
        for label in args.labels:
            await reembed_label(
                driver,
                http_session,
                label,
                dry_run=args.dry_run,
                batch_size=args.batch_size,
            )

    await driver.close()
    print("\n🏁 Done.")


if __name__ == "__main__":
    asyncio.run(main())
