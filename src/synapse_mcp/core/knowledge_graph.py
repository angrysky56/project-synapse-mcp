"""
Knowledge Graph management using Neo4j for Project Synapse.

This module provides the core graph database functionality for storing
entities, relationships, and semantic facts extracted from text.
"""

import os

from neo4j import AsyncGraphDatabase

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class KnowledgeGraph:
    """
    Neo4j-based knowledge graph for storing semantic information.

    Implements the Knowledge Cortex component of Project Synapse,
    managing entities, relationships, and semantic facts.
    """

    def __init__(self):
        self.driver = None
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "synapse_password")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

    async def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )

            # Test connection
            if self.driver is None:
                raise RuntimeError("KnowledgeGraph driver is not initialized. Call connect() first.")
            async with self.driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 as test")
                await result.single()

            logger.info("Successfully connected to Neo4j database")
            await self._initialize_schema()

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    async def close(self) -> None:
        """Close database connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")

    async def _initialize_schema(self) -> None:
        """Initialize database schema and constraints."""
        if self.driver is None:
            raise RuntimeError("KnowledgeGraph driver is not initialized. Call connect() first.")

        schema_queries = [
            # Entity constraints
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT fact_id_unique IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT zettel_id_unique IF NOT EXISTS FOR (z:Zettel) REQUIRE z.id IS UNIQUE",

            # Indexes for performance
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX fact_content_index IF NOT EXISTS FOR (f:Fact) ON (f.content)",
            "CREATE INDEX zettel_topic_index IF NOT EXISTS FOR (z:Zettel) ON (z.topic)",
        ]

        async with self.driver.session(database=self.database) as session:
            for query in schema_queries:
                try:
                    await session.run(query)  # type: ignore[arg-type]
                    logger.debug(f"Executed schema query: {query}")
                except Exception as e:
                    logger.warning(f"Schema query failed (may already exist): {e}")

    async def store_processed_data(self, processed_data: dict) -> dict:
        """
        Store processed semantic data in the knowledge graph.

        Args:
            processed_data: dictionary containing entities, relationships, and facts

        Returns:
            dictionary with storage statistics
        """

        if self.driver is None:
            raise RuntimeError("KnowledgeGraph driver is not initialized. Call connect() first.")
        async with self.driver.session(database=self.database) as session:
            # Start transaction for atomic operations
            tx = await session.begin_transaction()
            try:
                stats = {
                    'entities_count': 0,
                    'relationships_count': 0,
                    'facts_count': 0,
                    'new_nodes': 0,
                    'new_edges': 0
                }

                # Store entities
                if 'entities' in processed_data:
                    for entity in processed_data['entities']:
                        await self._store_entity(tx, entity)
                        stats['entities_count'] += 1

                # Store relationships
                if 'relationships' in processed_data:
                    for rel in processed_data['relationships']:
                        await self._store_relationship(tx, rel)
                        stats['relationships_count'] += 1

                # Store semantic facts
                if 'facts' in processed_data:
                    for fact in processed_data['facts']:
                        await self._store_fact(tx, fact)
                        stats['facts_count'] += 1

                await tx.commit()
                logger.info(f"Stored {stats['entities_count']} entities, "
                           f"{stats['relationships_count']} relationships, "
                           f"{stats['facts_count']} facts")
                return stats
            finally:
                await tx.close()

    async def _store_entity(self, tx, entity: dict) -> None:
        """Store an entity node in the graph."""
        # Flatten properties to ensure all values are primitive types
        properties = entity.get('properties', {})
        flattened_props = {}

        for key, value in properties.items():
            if isinstance(value, str | int | float | bool):
                flattened_props[key] = value
            elif isinstance(value, list):
                flattened_props[key] = ', '.join(str(item) for item in value)
            else:
                flattened_props[key] = str(value)

        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.type = $type,
            e.confidence = $confidence,
            e.source = $source,
            e.original_label = $original_label,
            e.start_char = $start_char,
            e.end_char = $end_char,
            e.created_at = timestamp()
        RETURN e
        """

        await tx.run(query, {
            'id': entity['id'],
            'name': entity['name'],
            'type': entity.get('type', 'Unknown'),
            'confidence': entity.get('confidence', 1.0),
            'source': entity.get('source', 'unknown'),
            'original_label': flattened_props.get('original_label', ''),
            'start_char': flattened_props.get('start_char', -1),
            'end_char': flattened_props.get('end_char', -1)
        })

    async def _store_relationship(self, tx, relationship: dict) -> None:
        """Store a relationship between entities."""
        # Flatten properties to ensure all values are primitive types
        properties = relationship.get('properties', {})
        flattened_props = {}

        for key, value in properties.items():
            if isinstance(value, str | int | float | bool):
                flattened_props[key] = value
            elif isinstance(value, list):
                flattened_props[key] = ', '.join(str(item) for item in value)
            else:
                flattened_props[key] = str(value)

        query = """
        MATCH (a:Entity {id: $source_id})
        MATCH (b:Entity {id: $target_id})
        MERGE (a)-[r:RELATES {type: $rel_type}]->(b)
        SET r.confidence = $confidence,
            r.source = $source,
            r.predicate = $predicate,
            r.source_span = $source_span,
            r.created_at = timestamp()
        RETURN r
        """

        await tx.run(query, {
            'source_id': relationship['source_id'],
            'target_id': relationship['target_id'],
            'rel_type': relationship['type'],
            'confidence': relationship.get('confidence', 1.0),
            'source': relationship.get('source', 'unknown'),
            'predicate': flattened_props.get('predicate', ''),
            'source_span': flattened_props.get('source_span', '')
        })

    async def _store_fact(self, tx, fact: dict) -> None:
        """Store a semantic fact as a node with connections to entities."""
        # Completely flatten metadata to only simple key-value pairs
        flattened_metadata = {}
        metadata = fact.get('metadata', {})

        for key, value in metadata.items():
            if isinstance(value, str | int | float | bool):
                flattened_metadata[key] = value
            elif isinstance(value, list):
                # Convert list to comma-separated string
                flattened_metadata[key] = ', '.join(str(item) for item in value)
            elif isinstance(value, dict):
                # Convert dict to JSON string
                import json
                flattened_metadata[key] = json.dumps(value)
            else:
                # Convert anything else to string
                flattened_metadata[key] = str(value)

        query = """
        CREATE (f:Fact {
            id: $id,
            content: $content,
            logical_form: $logical_form,
            confidence: $confidence,
            source: $source,
            extraction_method: $extraction_method,
            entity_list: $entity_list,
            created_at: timestamp()
        })
        RETURN f
        """

        # Extract specific metadata fields as separate properties
        extraction_method = flattened_metadata.get('extraction_method', 'unknown')
        entity_list = flattened_metadata.get('entities', '')

        await tx.run(query, {
            'id': fact['id'],
            'content': fact['content'],
            'logical_form': fact.get('logical_form', ''),
            'confidence': fact.get('confidence', 1.0),
            'source': fact.get('source', 'unknown'),
            'extraction_method': extraction_method,
            'entity_list': entity_list
        })

        # Connect fact to related entities
        if 'entities' in fact:
            for entity_id in fact['entities']:
                connect_query = """
                MATCH (f:Fact {id: $fact_id})
                MATCH (e:Entity {id: $entity_id})
                MERGE (f)-[:MENTIONS]->(e)
                """
                await tx.run(connect_query, {
                    'fact_id': fact['id'],
                    'entity_id': entity_id
                })

    async def query_semantic(self, query: str, max_results: int = 10) -> list[dict]:
        """
        Query the knowledge graph using semantic search.

        Args:
            query: Natural language query
            max_results: Maximum number of results

        Returns:
            list of relevant facts and entities
        """
        # For now, implement basic text matching
        # TODO: Implement proper semantic embedding-based search

        search_query = """
        MATCH (f:Fact)
        WHERE f.content CONTAINS $query_text
        RETURN f.content as statement, f.source as source, f.confidence as confidence
        ORDER BY f.confidence DESC
        LIMIT $limit
        """

        if self.driver is None:
            raise RuntimeError("KnowledgeGraph driver is not initialized. Call connect() first.")
        async with self.driver.session(database=self.database) as session:
            result = await session.run(search_query, {
                'query_text': query.lower(),
                'limit': max_results
            })

            facts = []
            async for record in result:
                facts.append({
                    'statement': record['statement'],
                    'source': record['source'],
                    'confidence': record['confidence']
                })

            return facts

    async def explore_entity_connections(
        self,
        entity: str,
        depth: int = 2,
        connection_types: list[str] | None = None
    ) -> list[dict]:
        """
        Explore connections around an entity using graph traversal.

        Args:
            entity: Entity name to start from
            depth: Maximum traversal depth
            connection_types: Specific relationship types to follow

        Returns:
            list of connected entities with relationship information
        """
        # Construct query dynamically to avoid parameter issues with path length
        # Limit depth to reasonable values for performance
        max_depth = min(max(depth, 1), 5)

        # Build the query string with literal depth values
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
            raise RuntimeError("KnowledgeGraph driver is not initialized. Call connect() first.")

        async with self.driver.session(database=self.database) as session:
            from typing import LiteralString, cast
            result = await session.run(cast(LiteralString, query), {
                'entity': entity
            })

            connections = []
            async for record in result:
                connection = {
                    'target_entity': record['target_entity'],
                    'relationship_type': record['relationship_type'],
                    'depth': record['depth'],
                    'path': ' → '.join(record['path_types'])
                }

                # Filter by connection types if specified
                if connection_types is None or record['relationship_type'] in connection_types:
                    connections.append(connection)

            return connections

    async def get_statistics(self) -> dict:
        """
        Get current knowledge graph statistics.

        Returns:
            dictionary with various graph metrics
        """
        queries = {
            'entity_count': "MATCH (e:Entity) RETURN count(e) as count",
            'relationship_count': "MATCH ()-[r]->() RETURN count(r) as count",
            'fact_count': "MATCH (f:Fact) RETURN count(f) as count",
        }

        stats = {}

        if self.driver is None:
            raise RuntimeError("KnowledgeGraph driver is not initialized. Call connect() first.")
        async with self.driver.session(database=self.database) as session:
            for stat_name, query in queries.items():
                from typing import LiteralString, cast
                result = await session.run(cast(LiteralString, query))
                record = await result.single()
                stats[stat_name] = record['count'] if record else 0

        # Calculate additional metrics
        stats['density'] = 0.0
        if stats['entity_count'] > 1:
            max_edges = stats['entity_count'] * (stats['entity_count'] - 1)
            stats['density'] = stats['relationship_count'] / max_edges if max_edges > 0 else 0.0

        # Add system health metrics (mock values for now)
        stats.update({
            'texts_processed': 0,  # TODO: Implement tracking
            'processing_rate': 0.0,
            'error_rate': 0.0,
            'db_status': 'Connected',
            'memory_usage': 0.0,
            'active_connections': 1
        })

        return stats

    async def store_insight(self, insight: dict) -> str:
        """
        Store a generated insight (Zettel) in the knowledge graph.

        Args:
            insight: dictionary containing insight data

        Returns:
            Zettel ID of the stored insight
        """
        query = """
        CREATE (z:Zettel {
            id: $id,
            title: $title,
            content: $content,
            topic: $topic,
            confidence: $confidence,
            pattern_type: $pattern_type,
            created_at: timestamp(),
            metadata: $metadata
        })
        RETURN z.id as zettel_id
        """

        if self.driver is None:
            raise RuntimeError("KnowledgeGraph driver is not initialized. Call connect() first.")
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, {
                'id': insight['zettel_id'],
                'title': insight['title'],
                'content': insight['content'],
                'topic': insight.get('topic', ''),
                'confidence': insight['confidence'],
                'pattern_type': insight['pattern_type'],
                'metadata': insight.get('metadata', {})
            })

            record = await result.single()
            if record is None or record.get('zettel_id') is None:
                logger.error("Failed to store insight: No Zettel ID returned from database.")
                raise RuntimeError("Failed to store insight: No Zettel ID returned from database.")
            zettel_id = record['zettel_id']

            # Link insight to supporting evidence
            if 'evidence' in insight:
                for evidence in insight['evidence']:
                    await self._link_insight_to_evidence(session, zettel_id, evidence)

            logger.info(f"Stored insight with Zettel ID: {zettel_id}")
            return zettel_id

    async def _link_insight_to_evidence(self, session, zettel_id: str, evidence: dict) -> None:
        """Link an insight to its supporting evidence."""
        query = """
        MATCH (z:Zettel {id: $zettel_id})
        MATCH (f:Fact {id: $fact_id})
        MERGE (z)-[r:SUPPORTED_BY]->(f)
        SET r.weight = $weight
        """

        await session.run(query, {
            'zettel_id': zettel_id,
            'fact_id': evidence['fact_id'],
            'weight': evidence.get('weight', 1.0)
        })

    async def get_insights_by_topic(self, topic: str) -> list[dict]:
        """
        Retrieve insights related to a specific topic.

        Args:
            topic: Topic to search for

        Returns:
            list of insights with evidence trails
        """
        query = """
        MATCH (z:Zettel)
        WHERE z.topic CONTAINS $topic OR z.content CONTAINS $topic
        OPTIONAL MATCH (z)-[:SUPPORTED_BY]->(f:Fact)
        RETURN z.id as zettel_id,
               z.title as title,
               z.content as content,
               z.confidence as confidence,
               z.pattern_type as pattern_type,
               z.created_at as created_at,
               collect({
                   fact_id: f.id,
                   statement: f.content,
                   source: f.source
               }) as evidence
        ORDER BY z.confidence DESC
        """

        if self.driver is None:
            raise RuntimeError("KnowledgeGraph driver is not initialized. Call connect() first.")
        if self.driver is None:
            raise RuntimeError("KnowledgeGraph driver is not initialized. Call connect() first.")
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, {'topic': topic})

            insights = []
            async for record in result:
                insights.append({
                    'zettel_id': record['zettel_id'],
                    'title': record['title'],
                    'content': record['content'],
                    'confidence': record['confidence'],
                    'pattern_type': record['pattern_type'],
                    'created_at': record['created_at'],
                    'evidence': [e for e in record['evidence'] if e['fact_id']]
                })

            return insights
