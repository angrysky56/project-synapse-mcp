"""
Enhanced Mock Knowledge Graph with DuckDB Backend

Provides a practical, embedded alternative to Neo4j for development and testing.
Follows the servant-never-master principle with zero-setup requirements.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None
    DUCKDB_AVAILABLE = False

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class MockKnowledgeGraph:
    """
    Lightweight knowledge graph using DuckDB for embedded analytics.

    Provides the same interface as KnowledgeGraph but with zero setup requirements.
    Falls back to in-memory storage if DuckDB unavailable.
    """

    def __init__(self):
        self.conn: Any | None = None
        # Will be set to DuckDBPyConnection instance later if available
        self.in_memory_fallback = False
        self.entities = {}
        self.relationships = []
        self.facts = {}

        # Database path in project data directory
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        self.db_path = data_dir / "synapse_knowledge.duckdb"

    async def connect(self) -> None:
        """Initialize DuckDB connection with graceful fallback."""
        if not DUCKDB_AVAILABLE:
            logger.warning("DuckDB not available, using in-memory fallback")
            self.in_memory_fallback = True
            return

        try:
            if DUCKDB_AVAILABLE and duckdb is not None:
                self.conn = duckdb.connect(str(self.db_path))
                await self._initialize_schema()
                logger.info(f"Connected to DuckDB knowledge store: {self.db_path}")
            else:
                logger.warning("DuckDB is not available, cannot connect.")
                self.in_memory_fallback = True
        except Exception as e:
            logger.warning(f"DuckDB connection failed, using in-memory fallback: {e}")
            self.in_memory_fallback = True

    async def _initialize_schema(self) -> None:
        """Initialize DuckDB schema for knowledge storage."""
        if not self.conn:
            return

        schema_queries = [
            """CREATE TABLE IF NOT EXISTS entities (
                id VARCHAR PRIMARY KEY,
                name VARCHAR,
                type VARCHAR,
                confidence DOUBLE,
                source VARCHAR,
                properties JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",

            """CREATE TABLE IF NOT EXISTS relationships (
                id VARCHAR PRIMARY KEY,
                source_id VARCHAR,
                target_id VARCHAR,
                type VARCHAR,
                confidence DOUBLE,
                source VARCHAR,
                properties JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",

            """CREATE TABLE IF NOT EXISTS facts (
                id VARCHAR PRIMARY KEY,
                content TEXT,
                logical_form TEXT,
                confidence DOUBLE,
                source VARCHAR,
                entities VARCHAR[],
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",

            # Indexes for performance
            "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)",
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id)",
            "CREATE INDEX IF NOT EXISTS idx_facts_content ON facts(content)",
        ]

        for query in schema_queries:
            try:
                self.conn.execute(query)
            except Exception as e:
                logger.debug(f"Schema query warning (may already exist): {e}")

    async def store_processed_data(self, data: dict) -> dict:
        """Store processed semantic data with graceful degradation."""
        stats = {
            'entities_count': 0,
            'relationships_count': 0,
            'facts_count': 0,
            'new_nodes': 0,
            'new_edges': 0
        }

        try:
            if self.in_memory_fallback:
                return await self._store_in_memory(data, stats)
            else:
                return await self._store_in_duckdb(data, stats)
        except Exception as e:
            logger.error(f"Storage failed: {e}")
            return stats

    async def _store_in_memory(self, data: dict, stats: dict) -> dict:
        """Fallback storage in memory."""
        # Store entities
        for entity in data.get('entities', []):
            self.entities[entity['id']] = entity
            stats['entities_count'] += 1
            stats['new_nodes'] += 1

        # Store relationships
        for rel in data.get('relationships', []):
            rel_id = f"{rel['source_id']}_{rel['type']}_{rel['target_id']}"
            rel['id'] = rel_id
            self.relationships.append(rel)
            stats['relationships_count'] += 1
            stats['new_edges'] += 1

        # Store facts
        for fact in data.get('facts', []):
            self.facts[fact['id']] = fact
            stats['facts_count'] += 1

        logger.info(f"Stored in memory: {stats}")
        return stats

    async def _store_in_duckdb(self, data: dict, stats: dict) -> dict:
        """Store data in DuckDB."""
        if not self.conn:
            return await self._store_in_memory(data, stats)

        # Store entities
        for entity in data.get('entities', []):
            self.conn.execute("""
                INSERT OR REPLACE INTO entities
                (id, name, type, confidence, source, properties)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                entity['id'], entity['name'], entity['type'],
                entity['confidence'], entity['source'], entity.get('properties', {})
            ))
            stats['entities_count'] += 1
            stats['new_nodes'] += 1

        # Store relationships
        for rel in data.get('relationships', []):
            rel_id = f"{rel['source_id']}_{rel['type']}_{rel['target_id']}"
            self.conn.execute("""
                INSERT OR REPLACE INTO relationships
                (id, source_id, target_id, type, confidence, source, properties)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                rel_id, rel['source_id'], rel['target_id'], rel['type'],
                rel['confidence'], rel['source'], rel.get('properties', {})
            ))
            stats['relationships_count'] += 1
            stats['new_edges'] += 1

        # Store facts
        for fact in data.get('facts', []):
            self.conn.execute("""
                INSERT OR REPLACE INTO facts
                (id, content, logical_form, confidence, source, entities, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                fact['id'], fact['content'], fact.get('logical_form', ''),
                fact['confidence'], fact['source'], fact.get('entities', []),
                fact.get('metadata', {})
            ))
            stats['facts_count'] += 1

        logger.info(f"Stored in DuckDB: {stats}")
        return stats

    async def query_semantic(self, query: str, max_results: int = 10) -> list[dict]:
        """Query knowledge base with simple text matching."""
        if self.in_memory_fallback:
            return await self._query_in_memory(query, max_results)
        else:
            return await self._query_duckdb(query, max_results)

    async def _query_in_memory(self, query: str, max_results: int) -> list[dict]:
        """Query in-memory storage."""
        results = []
        query_lower = query.lower()

        # Search facts
        for fact in self.facts.values():
            if query_lower in fact['content'].lower():
                results.append({
                    'statement': fact['content'],
                    'source': fact['source'],
                    'confidence': fact['confidence'],
                    'type': 'fact'
                })

        # Search entities
        for entity in self.entities.values():
            if query_lower in entity['name'].lower():
                results.append({
                    'statement': f"Entity: {entity['name']} ({entity['type']})",
                    'source': entity['source'],
                    'confidence': entity['confidence'],
                    'type': 'entity'
                })

        return results[:max_results]

    async def _query_duckdb(self, query: str, max_results: int) -> list[dict]:
        """Query DuckDB storage."""
        if not self.conn:
            return await self._query_in_memory(query, max_results)

        results = []

        # Search facts
        fact_results = self.conn.execute("""
            SELECT content, source, confidence FROM facts
            WHERE content LIKE ?
            ORDER BY confidence DESC
            LIMIT ?
        """, (f"%{query}%", max_results)).fetchall()

        for row in fact_results:
            results.append({
                'statement': row[0],
                'source': row[1],
                'confidence': row[2],
                'type': 'fact'
            })

        # Search entities if space remains
        if len(results) < max_results:
            entity_results = self.conn.execute("""
                SELECT name, type, source, confidence FROM entities
                WHERE name LIKE ?
                ORDER BY confidence DESC
                LIMIT ?
            """, (f"%{query}%", max_results - len(results))).fetchall()

            for row in entity_results:
                results.append({
                    'statement': f"Entity: {row[0]} ({row[1]})",
                    'source': row[2],
                    'confidence': row[3],
                    'type': 'entity'
                })

        return results

    async def get_statistics(self) -> dict[str, Any]:
        """Get knowledge base statistics."""
        if self.in_memory_fallback:
            return {
                'entity_count': len(self.entities),
                'relationship_count': len(self.relationships),
                'fact_count': len(self.facts),
                'density': 0.0,
                'texts_processed': len(self.facts),
                'processing_rate': 0.0,
                'error_rate': 0.0,
                'db_status': 'in_memory',
                'memory_usage': 0.0,
                'active_connections': 1
            }

        if not self.conn:
            return {'error': 'No database connection'}

        try:
            entity_count = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            relationship_count = self.conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
            fact_count = self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

            return {
                'entity_count': entity_count,
                'relationship_count': relationship_count,
                'fact_count': fact_count,
                'density': relationship_count / max(entity_count, 1),
                'texts_processed': fact_count
            }
        except Exception:
            return {'error': 'Database query failed'}
