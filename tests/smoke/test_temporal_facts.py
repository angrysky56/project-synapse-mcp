"""Quick smoke test for the temporal-facts pipeline.

Writes three facts that exercise the bitemporal model, then queries them
back to verify:
  - subject/object as both sides of relations
  - valid_from/valid_to handling
  - causal-window correlation
  - invalidate marks a fact ended without losing history

This is a one-shot script, not a unit test — running it pollutes the DB
with a 'test' source tag so it's easy to clean up:

    MATCH (f:TemporalFact {source: 'test:temporal_facts_smoke'})
    DETACH DELETE f
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

load_dotenv(Path(__file__).parent / ".env")

from synapse_mcp.core.temporal_facts import TemporalFact, TemporalFactStore

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "synapse_password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "synapse")

SOURCE = "test:temporal_facts_smoke"


async def main() -> None:
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    store = TemporalFactStore(driver, NEO4J_DATABASE)
    await store.initialize_schema()

    # Clean any prior smoke-test data.
    async with driver.session(database=NEO4J_DATABASE) as s:
        await s.run(
            "MATCH (f:TemporalFact {source: $src}) DETACH DELETE f", src=SOURCE
        )

    today = datetime.now(timezone.utc).replace(microsecond=0)
    yesterday = today - timedelta(days=1)
    two_days_ago = today - timedelta(days=2)
    three_days_ago = today - timedelta(days=3)

    # Three causally-suggestive facts.
    await store.add(
        TemporalFact(
            subject="Ty",
            predicate="started_taking",
            object="new_medication",
            valid_from=three_days_ago,
            source=SOURCE,
        )
    )
    await store.add(
        TemporalFact(
            subject="Ty",
            predicate="ate",
            object="aged_cheese",
            valid_from=two_days_ago,
            source=SOURCE,
        )
    )
    await store.add(
        TemporalFact(
            subject="Ty",
            predicate="experienced",
            object="headache",
            valid_from=yesterday,
            source=SOURCE,
        )
    )

    print("=" * 60)
    print("STATS")
    print("=" * 60)
    print(await store.stats())

    print()
    print("=" * 60)
    print("TIMELINE for Ty")
    print("=" * 60)
    for r in await store.timeline(entity_name="Ty"):
        print(
            f"  {r['valid_from']}  "
            f"({r['subject']}) -[{r['predicate']}]-> ({r['object']})"
        )

    print()
    print("=" * 60)
    print("CAUSAL-WINDOW: what happened in the 7d before the headache?")
    print("=" * 60)
    rows = await store.causal_chain(
        effect_entity="headache",
        before=yesterday + timedelta(hours=1),
        within_days=7,
    )
    for r in rows:
        print(
            f"  {r['days_before']}d before: "
            f"({r['cause_subject']})-[{r['cause_predicate']}]->"
            f"({r['cause_object']})"
        )

    print()
    print("=" * 60)
    print("INVALIDATE: stopping the medication")
    print("=" * 60)
    n = await store.invalidate("Ty", "started_taking", "new_medication")
    print(f"  invalidated {n} facts")

    print()
    print("=" * 60)
    print("RECALL: facts about Ty AS OF the headache day")
    print("=" * 60)
    rows = await store.query_entity("Ty", as_of=yesterday)
    for r in rows:
        vt = r["valid_to"]
        print(
            f"  ({r['subject']})-[{r['predicate']}]->({r['object']}) "
            f"valid_from={r['valid_from']} valid_to={vt}"
        )

    print()
    print("Done. Smoke-test data tagged with source=" + SOURCE)
    print("To clean up:")
    print(
        f"  MATCH (f:TemporalFact {{source: '{SOURCE}'}}) DETACH DELETE f"
    )

    await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
