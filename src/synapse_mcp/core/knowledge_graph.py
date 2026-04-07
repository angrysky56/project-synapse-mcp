"""
Knowledge Graph management using Neo4j for Project Synapse.

Upgraded for Neo4j 2026.x with:
  - Native VECTOR type and vector indexes for ANN semantic search
  - Fulltext indexes for hybrid BM25 + vector search
  - Local embeddings via sentence-transformers or Ollama (no paid APIs)
  - db.create.setNodeVectorProperty for efficient vector storage
"""

import os

from neo4j import AsyncGraphDatabase

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# Embedding config — all local, no paid APIs
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", "768"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
# Set EMBEDDING_PROVIDER=ollama and OLLAMA_EMBED_MODEL=mxbai-embed-large to use Ollama
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class KnowledgeGraph:
    """
    Neo4j 2026.x knowledge graph with local vector embeddings.

    Embedding pipeline: Python (sentence-transformers or Ollama)
        → db.create.setNodeVectorProperty() → Neo4j VECTOR index
        → db.index.vector.queryNodes() for ANN search
    """

    def __init__(self):
        self.driver = None
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "synapse_password")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self._local_embedder = None  # lazy-loaded

    async def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            async with self.driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            logger.info("Connected to Neo4j database")
            await self._initialize_schema()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    async def close(self) -> None:
        """Close database connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")

    # ------------------------------------------------------------------
    # Schema initialisation (Neo4j 2026.x)
    # ------------------------------------------------------------------

    async def _initialize_schema(self) -> None:
        """Create constraints, property indexes, vector indexes, and fulltext indexes."""
        if self.driver is None:
            raise RuntimeError("Driver not initialized.")

        schema_queries = [
            # Uniqueness constraints
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT fact_id_unique IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT zettel_id_unique IF NOT EXISTS FOR (z:Zettel) REQUIRE z.id IS UNIQUE",
            # Property indexes
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX fact_source_index IF NOT EXISTS FOR (f:Fact) ON (f.source)",
            "CREATE INDEX zettel_topic_index IF NOT EXISTS FOR (z:Zettel) ON (z.topic)",
        ]

        # Vector indexes for ANN semantic search
        vector_queries = [
            f"""CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
                FOR (e:Entity) ON e.embedding
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {EMBEDDING_DIM},
                    `vector.similarity_function`: 'cosine'
                }}}}""",
            f"""CREATE VECTOR INDEX fact_embedding IF NOT EXISTS
                FOR (f:Fact) ON f.embedding
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {EMBEDDING_DIM},
                    `vector.similarity_function`: 'cosine'
                }}}}""",
            f"""CREATE VECTOR INDEX zettel_embedding IF NOT EXISTS
                FOR (z:Zettel) ON z.embedding
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {EMBEDDING_DIM},
                    `vector.similarity_function`: 'cosine'
                }}}}""",
        ]

        # Fulltext indexes for BM25 keyword search
        fulltext_queries = [
            """CALL db.index.fulltext.createNodeIndex(
                'entity_fulltext', ['Entity'], ['name', 'type'],
                {analyzer: 'standard-no-stop-words'})""",
            """CALL db.index.fulltext.createNodeIndex(
                'fact_fulltext', ['Fact'], ['content'],
                {analyzer: 'standard-no-stop-words'})""",
        ]

        async with self.driver.session(database=self.database) as session:
            for query in schema_queries + vector_queries:
                try:
                    await session.run(query)
                except Exception as e:
                    logger.warning(f"Schema query skipped: {e}")
            for query in fulltext_queries:
                try:
                    await session.run(query)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Fulltext index skipped: {e}")
        logger.info("Schema initialisation complete")

    # ------------------------------------------------------------------
    # Local embedding (no paid APIs)
    # ------------------------------------------------------------------

    def _get_local_embedder(self):
        """Lazy-load sentence-transformers model."""
        if self._local_embedder is None:
            from sentence_transformers import SentenceTransformer
            self._local_embedder = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
        return self._local_embedder

    async def _embed_text(self, text: str) -> list[float]:
        """Generate embedding locally. Supports sentence-transformers or Ollama."""
        import asyncio

        if EMBEDDING_PROVIDER == "ollama":
            return await self._embed_via_ollama(text)

        # Default: sentence-transformers (runs on GPU if available)
        model = self._get_local_embedder()
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(None, model.encode, text)
        return vec.tolist()

    async def _embed_via_ollama(self, text: str) -> list[float]:
        """Get embedding from local Ollama instance."""
        import aiohttp
        url = f"{OLLAMA_BASE_URL}/api/embed"
        payload = {"model": OLLAMA_EMBED_MODEL, "input": text}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()
                    return data["embeddings"][0]
        except Exception as e:
            logger.warning(f"Ollama embed failed, falling back to local: {e}")
            import asyncio
            model = self._get_local_embedder()
            loop = asyncio.get_event_loop()
            vec = await loop.run_in_executor(None, model.encode, text)
            return vec.tolist()

    # ------------------------------------------------------------------
    # Store operations
    # ------------------------------------------------------------------

    async def store_processed_data(self, processed_data: dict) -> dict:
        """Store processed semantic data in the knowledge graph."""
        if self.driver is None:
            raise RuntimeError("Driver not initialized.")

        async with self.driver.session(database=self.database) as session:
            tx = await session.begin_transaction()
            try:
                stats = {
                    'entities_count': 0, 'relationships_count': 0,
                    'facts_count': 0, 'new_nodes': 0, 'new_edges': 0,
                }
                if 'entities' in processed_data:
                    for entity in processed_data['entities']:
                        was_created = await self._store_entity(tx, entity)
                        stats['entities_count'] += 1
                        if was_created:
                            stats['new_nodes'] += 1
                if 'relationships' in processed_data:
                    for rel in processed_data['relationships']:
                        was_created = await self._store_relationship(tx, rel)
                        stats['relationships_count'] += 1
                        if was_created:
                            stats['new_edges'] += 1
                if 'facts' in processed_data:
                    for fact in processed_data['facts']:
                        was_created = await self._store_fact(tx, fact)
                        stats['facts_count'] += 1
                        if was_created:
                            stats['new_nodes'] += 1
                await tx.commit()
                logger.info(
                    f"Stored {stats['entities_count']} entities "
                    f"({stats['new_nodes']} new), "
                    f"{stats['relationships_count']} rels "
                    f"({stats['new_edges']} new), "
                    f"{stats['facts_count']} facts"
                )
                return stats
            finally:
                await tx.close()

    async def _store_entity(self, tx, entity: dict) -> bool:
        """Store an entity node with vector embedding."""
        props = entity.get('properties', {})
        flat: dict = {}
        for k, v in props.items():
            if isinstance(v, str | int | float | bool):
                flat[k] = v
            elif isinstance(v, list):
                flat[k] = ', '.join(str(i) for i in v)
            else:
                flat[k] = str(v)

        # Generate embedding for entity
        embed_text = f"{entity['name']} ({entity.get('type', '')})"
        embedding = await self._embed_text(embed_text)

        query = """
        MERGE (e:Entity {id: $id})
        ON CREATE SET e.created_at = timestamp(), e.was_created = true
        ON MATCH SET e.was_created = false
        SET e.name = $name, e.type = $type,
            e.confidence = $confidence, e.source = $source,
            e.original_label = $original_label,
            e.start_char = $start_char, e.end_char = $end_char,
            e.updated_at = timestamp()
        RETURN e.was_created as created
        """
        result = await tx.run(query, {
            'id': entity['id'], 'name': entity['name'],
            'type': entity.get('type', 'Unknown'),
            'confidence': entity.get('confidence', 1.0),
            'source': entity.get('source', 'unknown'),
            'original_label': flat.get('original_label', ''),
            'start_char': flat.get('start_char', -1),
            'end_char': flat.get('end_char', -1),
        })
        record = await result.single()
        was_created = record['created'] if record else False

        # Store vector embedding on node
        await tx.run(
            "MATCH (e:Entity {id: $id}) "
            "CALL db.create.setNodeVectorProperty(e, 'embedding', $vec)",
            {'id': entity['id'], 'vec': embedding},
        )
        return was_created

    async def _store_relationship(self, tx, relationship: dict) -> bool:
        """Store a relationship between entities."""
        props = relationship.get('properties', {})
        flat: dict = {}
        for k, v in props.items():
            if isinstance(v, str | int | float | bool):
                flat[k] = v
            elif isinstance(v, list):
                flat[k] = ', '.join(str(i) for i in v)
            else:
                flat[k] = str(v)

        query = """
        MATCH (a:Entity {id: $source_id})
        MATCH (b:Entity {id: $target_id})
        MERGE (a)-[r:RELATES {type: $rel_type}]->(b)
        ON CREATE SET r.created_at = timestamp(), r.was_created = true
        ON MATCH SET r.was_created = false
        SET r.confidence = $confidence, r.source = $source,
            r.predicate = $predicate, r.source_span = $source_span,
            r.updated_at = timestamp()
        RETURN r.was_created as created
        """
        result = await tx.run(query, {
            'source_id': relationship['source_id'],
            'target_id': relationship['target_id'],
            'rel_type': relationship['type'],
            'confidence': relationship.get('confidence', 1.0),
            'source': relationship.get('source', 'unknown'),
            'predicate': flat.get('predicate', ''),
            'source_span': flat.get('source_span', ''),
        })
        record = await result.single()
        return record['created'] if record else False

    async def _store_fact(self, tx, fact: dict) -> bool:
        """Store a semantic fact with vector embedding."""
        metadata = fact.get('metadata', {})
        flat: dict = {}
        for k, v in metadata.items():
            if isinstance(v, str | int | float | bool):
                flat[k] = v
            elif isinstance(v, list):
                flat[k] = ', '.join(str(i) for i in v)
            elif isinstance(v, dict):
                import json
                flat[k] = json.dumps(v)
            else:
                flat[k] = str(v)

        embedding = await self._embed_text(fact['content'])

        query = """
        MERGE (f:Fact {id: $id})
        ON CREATE SET f.created_at = timestamp(), f.was_created = true
        ON MATCH SET f.was_created = false
        SET f.content = $content, f.logical_form = $logical_form,
            f.confidence = $confidence, f.source = $source,
            f.extraction_method = $extraction_method,
            f.entity_list = $entity_list,
            f.updated_at = timestamp()
        RETURN f.was_created as created
        """
        result = await tx.run(query, {
            'id': fact['id'], 'content': fact['content'],
            'logical_form': fact.get('logical_form', ''),
            'confidence': fact.get('confidence', 1.0),
            'source': fact.get('source', 'unknown'),
            'extraction_method': flat.get('extraction_method', 'unknown'),
            'entity_list': flat.get('entities', ''),
        })
        record = await result.single()
        was_created = record['created'] if record else False

        # Store vector embedding
        await tx.run(
            "MATCH (f:Fact {id: $id}) "
            "CALL db.create.setNodeVectorProperty(f, 'embedding', $vec)",
            {'id': fact['id'], 'vec': embedding},
        )

        # Link fact to mentioned entities
        if 'entities' in fact:
            for entity_id in fact['entities']:
                await tx.run(
                    "MATCH (f:Fact {id: $fid}) "
                    "MATCH (e:Entity {id: $eid}) "
                    "MERGE (f)-[:MENTIONS]->(e)",
                    {'fid': fact['id'], 'eid': entity_id},
                )
        return was_created

    # ------------------------------------------------------------------
    # Query: vector semantic search (replaces old CONTAINS matching)
    # ------------------------------------------------------------------

    async def query_semantic(self, query: str, max_results: int = 10) -> list[dict]:
        """ANN vector search over Facts using Neo4j vector index."""
        if self.driver is None:
            raise RuntimeError("Driver not initialized.")

        query_embedding = await self._embed_text(query)

        vector_query = """
        CALL db.index.vector.queryNodes('fact_embedding', $limit, $vec)
        YIELD node AS f, score
        RETURN f.content AS statement, f.source AS source,
               f.confidence AS confidence, score
        ORDER BY score DESC
        """
        async with self.driver.session(database=self.database) as session:
            result = await session.run(vector_query, {
                'vec': query_embedding, 'limit': max_results,
            })
            facts: list[dict] = []
            async for record in result:
                facts.append({
                    'statement': record['statement'],
                    'source': record['source'],
                    'confidence': record['confidence'],
                    'similarity': record['score'],
                })
            return facts

    async def query_hybrid(self, query: str, max_results: int = 10) -> list[dict]:
        """Hybrid search: vector ANN + fulltext BM25 keyword boost."""
        if self.driver is None:
            raise RuntimeError("Driver not initialized.")

        query_embedding = await self._embed_text(query)

        # Get vector results
        vec_results = await self.query_semantic(query, max_results * 2)

        # Get fulltext BM25 results
        ft_query = """
        CALL db.index.fulltext.queryNodes('fact_fulltext', $query)
        YIELD node AS f, score AS ftScore
        RETURN f.id AS id, f.content AS statement, f.source AS source,
               f.confidence AS confidence, ftScore
        LIMIT $limit
        """
        async with self.driver.session(database=self.database) as session:
            result = await session.run(ft_query, {
                'query': query, 'limit': max_results,
            })
            ft_hits: dict[str, float] = {}
            async for record in result:
                ft_hits[record['statement']] = record['ftScore']

        # Merge: boost vector results that also matched fulltext
        for fact in vec_results:
            ft_boost = ft_hits.get(fact['statement'], 0.0)
            fact['hybrid_score'] = fact['similarity'] + (ft_boost * 0.1)

        vec_results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        return vec_results[:max_results]

    async def explore_entity_connections(
        self, entity: str, depth: int = 2,
        connection_types: list[str] | None = None,
    ) -> list[dict]:
        """Graph traversal to discover connections around an entity."""
        max_depth = min(max(depth, 1), 5)
        query = f"""
        MATCH path = (start:Entity {{name: $entity}})-[r*1..{max_depth}]-(connected:Entity)
        WHERE start <> connected
        RETURN connected.name as target_entity,
               type(last(relationships(path))) as relationship_type,
               length(path) as depth,
               [rel in relationships(path) | type(rel)] as path_types
        ORDER BY depth, connected.name
        LIMIT 100
        """
        if self.driver is None:
            raise RuntimeError("Driver not initialized.")

        async with self.driver.session(database=self.database) as session:
            from typing import LiteralString, cast
            result = await session.run(cast(LiteralString, query), {
                'entity': entity,
            })
            connections: list[dict] = []
            async for record in result:
                conn = {
                    'target_entity': record['target_entity'],
                    'relationship_type': record['relationship_type'],
                    'depth': record['depth'],
                    'path': ' → '.join(record['path_types']),
                }
                if connection_types is None or record['relationship_type'] in connection_types:
                    connections.append(conn)
            return connections

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def get_statistics(self) -> dict:
        """Get current knowledge graph statistics."""
        queries = {
            'entity_count': "MATCH (e:Entity) RETURN count(e) as count",
            'relationship_count': "MATCH ()-[r]->() RETURN count(r) as count",
            'fact_count': "MATCH (f:Fact) RETURN count(f) as count",
        }
        stats: dict = {}
        if self.driver is None:
            raise RuntimeError("Driver not initialized.")
        async with self.driver.session(database=self.database) as session:
            for stat_name, query in queries.items():
                from typing import LiteralString, cast
                result = await session.run(cast(LiteralString, query))
                record = await result.single()
                stats[stat_name] = record['count'] if record else 0

        stats['density'] = 0.0
        if stats['entity_count'] > 1:
            max_edges = stats['entity_count'] * (stats['entity_count'] - 1)
            stats['density'] = stats['relationship_count'] / max_edges if max_edges > 0 else 0.0
        stats.update({
            'texts_processed': 0,
            'processing_rate': 0.0,
            'error_rate': 0.0,
            'db_status': 'Connected',
            'memory_usage': 0.0,
            'active_connections': 1,
        })
        return stats

    # ------------------------------------------------------------------
    # Insight (Zettel) storage
    # ------------------------------------------------------------------

    async def store_insight(self, insight: dict) -> str:
        """Store a generated insight (Zettel) with vector embedding."""
        embedding = await self._embed_text(insight['content'])

        query = """
        CREATE (z:Zettel {
            id: $id, title: $title, content: $content,
            topic: $topic, confidence: $confidence,
            pattern_type: $pattern_type, created_at: timestamp()
        })
        RETURN z.id as zettel_id
        """
        if self.driver is None:
            raise RuntimeError("Driver not initialized.")
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, {
                'id': insight['zettel_id'],
                'title': insight['title'],
                'content': insight['content'],
                'topic': insight.get('topic', ''),
                'confidence': insight['confidence'],
                'pattern_type': insight['pattern_type'],
            })
            record = await result.single()
            if record is None or record.get('zettel_id') is None:
                raise RuntimeError("Failed to store insight.")
            zettel_id = record['zettel_id']

            # Store vector embedding
            await session.run(
                "MATCH (z:Zettel {id: $id}) "
                "CALL db.create.setNodeVectorProperty(z, 'embedding', $vec)",
                {'id': zettel_id, 'vec': embedding},
            )

            # Link to supporting evidence
            if 'evidence' in insight:
                for evidence in insight['evidence']:
                    await self._link_insight_to_evidence(session, zettel_id, evidence)

            logger.info(f"Stored insight: {zettel_id}")
            return zettel_id

    async def _link_insight_to_evidence(self, session, zettel_id: str, evidence: dict) -> None:
        """Link an insight to its supporting evidence."""
        await session.run(
            "MATCH (z:Zettel {id: $zid}) "
            "MATCH (f:Fact {id: $fid}) "
            "MERGE (z)-[r:SUPPORTED_BY]->(f) "
            "SET r.weight = $weight",
            {'zid': zettel_id, 'fid': evidence['fact_id'],
             'weight': evidence.get('weight', 1.0)},
        )

    async def get_insights_by_topic(self, topic: str) -> list[dict]:
        """Retrieve insights related to a topic using vector similarity."""
        if self.driver is None:
            raise RuntimeError("Driver not initialized.")

        # Use vector search on Zettel nodes for semantic topic matching
        topic_embedding = await self._embed_text(topic)
        query = """
        CALL db.index.vector.queryNodes('zettel_embedding', $limit, $vec)
        YIELD node AS z, score
        OPTIONAL MATCH (z)-[:SUPPORTED_BY]->(f:Fact)
        RETURN z.id as zettel_id, z.title as title,
               z.content as content, z.confidence as confidence,
               z.pattern_type as pattern_type, z.created_at as created_at,
               score,
               collect({fact_id: f.id, statement: f.content, source: f.source}) as evidence
        ORDER BY score DESC
        """
