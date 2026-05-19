"""
temporal_facts.py — bitemporal fact store for Synapse.

Adds time-aware facts alongside Synapse's existing atemporal Fact nodes.
Every TemporalFact carries:
  - valid_from / valid_to: when the fact is true in the world
  - observed_at: when WE learned about it (audit trail)

Edges into the existing entity graph:
  (:TemporalFact)-[:SUBJECT]->(:Entity)
  (:TemporalFact)-[:OBJECT]->(:Entity)
  (:TemporalFact)-[:MENTIONS]->(:Entity)

Shares Neo4j driver with KnowledgeGraph — no second bolt pool.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neo4j import AsyncDriver

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TemporalFact:
    """A single timestamped assertion.

    subject/predicate/object are entity-name strings (resolved to canonical
    IDs at write time using KnowledgeUtils.generate_entity_id, same as the
    main pipeline).

    valid_from is required; valid_to is optional (still-true facts have no
    end). observed_at is set automatically on insert — never trust callers.
    """

    subject: str
    predicate: str
    object: str
    valid_from: datetime
    valid_to: datetime | None = None
    confidence: float = 1.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_id(self) -> str:
        """Stable content-addressed ID. Same SPO + valid_from collapses to
        the same node so re-imports are idempotent."""
        h = hashlib.sha256(
            f"{self.subject}|{self.predicate}|{self.object}|"
            f"{self.valid_from.isoformat()}".encode()
        ).hexdigest()[:16]
        return f"tfact_{h}"


class TemporalFactStore:
    """Write and query temporal facts against Neo4j.

    Constructed with an existing AsyncDriver so we share the connection pool
    with the rest of Synapse. Uses managed transactions (execute_write) so
    we benefit from driver retry on transient errors.
    """

    def __init__(self, driver: AsyncDriver, database: str = "synapse") -> None:
        self.driver = driver
        self.database = database

    async def initialize_schema(self) -> None:
        """Idempotent. Safe to call on every server start."""
        queries = [
            "CREATE CONSTRAINT temporal_fact_id IF NOT EXISTS "
            "FOR (f:TemporalFact) REQUIRE f.id IS UNIQUE",
            "CREATE INDEX temporal_fact_valid_from IF NOT EXISTS "
            "FOR (f:TemporalFact) ON (f.valid_from)",
            "CREATE INDEX temporal_fact_valid_to IF NOT EXISTS "
            "FOR (f:TemporalFact) ON (f.valid_to)",
            "CREATE INDEX temporal_fact_observed_at IF NOT EXISTS "
            "FOR (f:TemporalFact) ON (f.observed_at)",
            "CREATE INDEX temporal_fact_source IF NOT EXISTS "
            "FOR (f:TemporalFact) ON (f.source)",
        ]
        async with self.driver.session(database=self.database) as session:
            for q in queries:
                try:
                    await session.run(q)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    msg = str(e).lower()
                    if "already exists" in msg or "equivalent" in msg:
                        continue
                    logger.warning("TemporalFact schema: %s", e)
        logger.info("TemporalFact schema ready")

    async def add(self, fact: TemporalFact) -> str:
        """Insert (or upsert by content hash) one fact. Returns the fact id.

        MERGEs the subject/object entities so the fact lands even if those
        names aren't separately extracted yet — they get an
        UnresolvedReference type until promotion later.
        """
        from ..knowledge.knowledge_types import KnowledgeUtils

        fact_id = fact.to_id()
        observed_at = datetime.now(timezone.utc)

        subject_id = KnowledgeUtils.generate_entity_id(
            fact.subject, "UnresolvedReference"
        )
        object_id = KnowledgeUtils.generate_entity_id(
            fact.object, "UnresolvedReference"
        )

        params = {
            "fid": fact_id,
            "subject": fact.subject,
            "predicate": fact.predicate,
            "object": fact.object,
            "valid_from": fact.valid_from.isoformat(),
            "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
            "observed_at": observed_at.isoformat(),
            "confidence": float(fact.confidence),
            "source": fact.source,
            "metadata_json": json.dumps(fact.metadata) if fact.metadata else "",
            "subject_id": subject_id,
            "subject_name": fact.subject,
            "object_id": object_id,
            "object_name": fact.object,
        }

        query = """
            MERGE (f:TemporalFact {id: $fid})
            ON CREATE SET
                f.subject = $subject,
                f.predicate = $predicate,
                f.object = $object,
                f.valid_from = datetime($valid_from),
                f.valid_to = CASE WHEN $valid_to IS NULL THEN NULL
                                  ELSE datetime($valid_to) END,
                f.observed_at = datetime($observed_at),
                f.confidence = $confidence,
                f.source = $source,
                f.metadata = $metadata_json
            ON MATCH SET
                f.confidence = $confidence,
                f.valid_to = CASE WHEN $valid_to IS NULL THEN f.valid_to
                                  ELSE datetime($valid_to) END
            MERGE (s:Entity {id: $subject_id})
            ON CREATE SET s.name = $subject_name,
                          s.type = 'UnresolvedReference',
                          s.source = $source
            MERGE (o:Entity {id: $object_id})
            ON CREATE SET o.name = $object_name,
                          o.type = 'UnresolvedReference',
                          o.source = $source
            MERGE (f)-[:SUBJECT]->(s)
            MERGE (f)-[:OBJECT]->(o)
            MERGE (f)-[:MENTIONS]->(s)
            MERGE (f)-[:MENTIONS]->(o)
            RETURN f.id AS fid
        """

        async with self.driver.session(database=self.database) as session:

            async def _work(tx):
                result = await tx.run(query, **params)
                record = await result.single()
                return record["fid"] if record else None

            return await session.execute_write(_work)

    async def invalidate(
        self,
        subject: str,
        predicate: str,
        object_: str,
        ended: datetime | None = None,
    ) -> int:
        """Mark a still-true fact as no longer true.

        Sets valid_to rather than deleting. Returns # of facts updated.
        """
        end_time = ended or datetime.now(timezone.utc)
        query = """
            MATCH (f:TemporalFact {
                subject: $subject,
                predicate: $predicate,
                object: $object
            })
            WHERE f.valid_to IS NULL
            SET f.valid_to = datetime($ended)
            RETURN count(f) AS n
        """
        async with self.driver.session(database=self.database) as session:

            async def _work(tx):
                result = await tx.run(
                    query,
                    subject=subject,
                    predicate=predicate,
                    object=object_,
                    ended=end_time.isoformat(),
                )
                record = await result.single()
                return record["n"] if record else 0

            return await session.execute_write(_work)

    async def query_entity(
        self,
        entity_name: str,
        as_of: datetime | None = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Return temporal facts about entity_name.

        as_of filters to facts valid at that timestamp.
        direction: 'outgoing' / 'incoming' / 'both'.
        """
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError(
                f"direction must be outgoing|incoming|both, got {direction}"
            )

        as_of_clause = ""
        if as_of is not None:
            as_of_clause = (
                "AND f.valid_from <= datetime($as_of) "
                "AND (f.valid_to IS NULL OR f.valid_to > datetime($as_of)) "
            )

        if direction == "outgoing":
            where_clause = "WHERE f.subject = $name " + as_of_clause
        elif direction == "incoming":
            where_clause = "WHERE f.object = $name " + as_of_clause
        else:
            where_clause = (
                "WHERE (f.subject = $name OR f.object = $name) " + as_of_clause
            )

        query = f"""
            MATCH (f:TemporalFact)
            {where_clause}
            RETURN f.id AS id, f.subject AS subject, f.predicate AS predicate,
                   f.object AS object, f.valid_from AS valid_from,
                   f.valid_to AS valid_to, f.observed_at AS observed_at,
                   f.confidence AS confidence, f.source AS source,
                   f.metadata AS metadata
            ORDER BY f.valid_from DESC
        """

        params: dict[str, Any] = {"name": entity_name}
        if as_of is not None:
            params["as_of"] = as_of.isoformat()

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, **params)
            return [dict(record) async for record in result]

    async def timeline(
        self, entity_name: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Chronological facts. Scoped to entity if given, else global."""
        if entity_name:
            query = """
                MATCH (f:TemporalFact)
                WHERE f.subject = $name OR f.object = $name
                RETURN f.id AS id, f.subject AS subject,
                       f.predicate AS predicate, f.object AS object,
                       f.valid_from AS valid_from, f.valid_to AS valid_to,
                       f.source AS source
                ORDER BY f.valid_from ASC
                LIMIT $limit
            """
            params: dict[str, Any] = {"name": entity_name, "limit": limit}
        else:
            query = """
                MATCH (f:TemporalFact)
                RETURN f.id AS id, f.subject AS subject,
                       f.predicate AS predicate, f.object AS object,
                       f.valid_from AS valid_from, f.valid_to AS valid_to,
                       f.source AS source
                ORDER BY f.valid_from ASC
                LIMIT $limit
            """
            params = {"limit": limit}

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, **params)
            return [dict(record) async for record in result]

    async def causal_chain(
        self,
        effect_entity: str,
        before: datetime,
        within_days: int = 30,
    ) -> list[dict[str, Any]]:
        """Temporal-correlation search.

        Returns facts whose valid_from is within [before - within_days, before]
        that share at least one entity (via MENTIONS) with a fact about
        effect_entity. Surfaces candidates — does NOT make causal claims.

        The 'track what you ate to find the headache cause' pattern.
        """
        query = """
            MATCH (effect:TemporalFact)
            WHERE (effect.subject = $entity OR effect.object = $entity)
              AND effect.valid_from <= datetime($before)
              AND effect.valid_from >= datetime($before) - duration({days: $window})

            MATCH (cause:TemporalFact)
            WHERE cause.valid_from <= effect.valid_from
              AND cause.valid_from >= datetime($before) - duration({days: $window})
              AND cause.id <> effect.id

            OPTIONAL MATCH path = (cause)-[:MENTIONS]->(:Entity)<-[:MENTIONS]-(effect)
            WITH cause, effect, count(path) AS shared
            WHERE shared > 0

            RETURN
              effect.subject AS effect_subject,
              effect.predicate AS effect_predicate,
              effect.object AS effect_object,
              effect.valid_from AS effect_when,
              cause.subject AS cause_subject,
              cause.predicate AS cause_predicate,
              cause.object AS cause_object,
              cause.valid_from AS cause_when,
              shared AS shared_entities,
              duration.between(cause.valid_from, effect.valid_from).days AS days_before
            ORDER BY effect.valid_from DESC, days_before ASC
            LIMIT 50
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                query,
                entity=effect_entity,
                before=before.isoformat(),
                window=within_days,
            )
            return [dict(record) async for record in result]

    async def stats(self) -> dict[str, Any]:
        """Quick health-check / overview numbers."""
        query = """
            MATCH (f:TemporalFact)
            WITH count(f) AS total,
                 sum(CASE WHEN f.valid_to IS NULL THEN 1 ELSE 0 END) AS still_true,
                 min(f.valid_from) AS earliest,
                 max(f.valid_from) AS latest
            RETURN total, still_true, earliest, latest
        """
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            record = await result.single()
            return (
                dict(record)
                if record
                else {
                    "total": 0,
                    "still_true": 0,
                    "earliest": None,
                    "latest": None,
                }
            )
