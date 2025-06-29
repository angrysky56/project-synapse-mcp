"""
Zettelkasten-inspired Insight Engine for Project Synapse.

This module implements the autonomous synthesis engine that identifies patterns
and generates novel insights from the knowledge graph.
"""

import asyncio
import os
import uuid
from datetime import datetime

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class InsightEngine:
    async def get_insights_by_topic(self, topic: str, max_results: int = 20) -> list[dict]:
        """
        Retrieve all insights related to a specific topic.

        Args:
            topic: The topic to retrieve insights for
            max_results: Maximum number of insights to return

        Returns:
            list of insights related to the specified topic
        """
        query = """
        MATCH (z:Zettel)
        WHERE toLower(z.topic) CONTAINS $topic OR toLower(z.title) CONTAINS $topic OR toLower(z.content) CONTAINS $topic
        OPTIONAL MATCH (z)-[:SUPPORTED_BY]->(f:Fact)
        RETURN z.id as zettel_id, z.title as title, z.content as content,
               z.confidence as confidence, z.pattern_type as pattern_type,
               z.created_at as created_at,
               collect({fact_id: f.id, statement: f.content, source: f.source}) as evidence
        ORDER BY z.confidence DESC
        LIMIT $limit
        """

        insights = []
        async with self.knowledge_graph.driver.session(
            database=self.knowledge_graph.database
        ) as session:
            result = await session.run(query, {
                'topic': topic.lower(),
                'limit': max_results
            })

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
    """
    Zettelkasten-inspired engine for autonomous insight generation.

    Implements pattern detection and knowledge synthesis using graph algorithms
    and machine learning to identify non-obvious connections and generate insights.
    """

    def __init__(self, knowledge_graph, montague_parser):
        self.knowledge_graph = knowledge_graph
        self.montague_parser = montague_parser
        self.graph = nx.DiGraph()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.is_running = False
        self.processing_interval = 300  # 5 minutes

        # Insight generation settings
        self.confidence_threshold = float(os.getenv('INSIGHT_CONFIDENCE_THRESHOLD', '0.8'))
        self.link_threshold = float(os.getenv('LINK_THRESHOLD', '0.7'))

    async def initialize(self) -> None:
        """Initialize the insight engine."""
        try:
            await self._build_analysis_graph()
            logger.info("Insight engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize insight engine: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up the insight engine."""
        self.is_running = False
        logger.info("Insight engine cleanup completed")

    async def start_autonomous_processing(self) -> None:
        """Start autonomous insight generation in background."""
        self.is_running = True
        logger.info("Starting autonomous insight processing")

        while self.is_running:
            try:
                await self._autonomous_processing_cycle()
                await asyncio.sleep(self.processing_interval)
            except asyncio.CancelledError:
                logger.info("Autonomous processing cancelled")
                break
            except Exception as e:
                logger.error(f"Error in autonomous processing: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _autonomous_processing_cycle(self) -> None:
        """Run one cycle of autonomous insight generation."""
        logger.debug("Running autonomous insight generation cycle")

        # Rebuild analysis graph with latest data
        await self._build_analysis_graph()

        # Detect patterns using various algorithms
        patterns = await self._detect_patterns()

        # Generate insights from detected patterns
        for pattern in patterns:
            try:
                insight = await self._generate_insight_from_pattern(pattern)
                if insight and insight['confidence'] >= self.confidence_threshold:
                    # Store insight in knowledge graph
                    zettel_id = await self.knowledge_graph.store_insight(insight)
                    logger.info(f"Generated new insight with Zettel ID: {zettel_id}")
            except Exception as e:
                logger.error(f"Failed to generate insight from pattern: {e}")

    async def _build_analysis_graph(self) -> None:
        """Build NetworkX graph from knowledge graph for analysis."""
        self.graph.clear()

        # Get all entities and relationships from knowledge graph
        query = """
        MATCH (a)-[r]->(b)
        WHERE a.id IS NOT NULL AND b.id IS NOT NULL
        RETURN a.id as source, b.id as target, type(r) as rel_type, r.confidence as confidence
        """

        try:
            async with self.knowledge_graph.driver.session(
                database=self.knowledge_graph.database
            ) as session:
                result = await session.run(query)

                edge_count = 0
                async for record in result:
                    source = record['source']
                    target = record['target']

                    # Only add edge if both source and target are valid
                    if source is not None and target is not None:
                        self.graph.add_edge(
                            source,
                            target,
                            rel_type=record['rel_type'] or 'RELATES',
                            weight=record['confidence'] or 1.0
                        )
                        edge_count += 1

            logger.debug(f"Built analysis graph with {self.graph.number_of_nodes()} nodes, "
                        f"{self.graph.number_of_edges()} edges")

        except Exception as e:
            logger.warning(f"Could not build analysis graph from knowledge base: {e}")
            logger.debug("Starting with empty analysis graph")

    async def _detect_patterns(self) -> list[dict]:
        """Detect patterns in the knowledge graph using various algorithms."""
        patterns = []

        if self.graph.number_of_nodes() < 2:
            logger.debug("Graph too small for pattern detection, skipping")
            return patterns

        # Community detection
        try:
            communities = await self._detect_communities()
            patterns.extend(communities)
        except Exception as e:
            logger.debug(f"Community detection failed: {e}")

        # Centrality analysis
        try:
            central_nodes = await self._analyze_centrality()
            patterns.extend(central_nodes)
        except Exception as e:
            logger.debug(f"Centrality analysis failed: {e}")

        # Path analysis
        try:
            interesting_paths = await self._find_interesting_paths()
            patterns.extend(interesting_paths)
        except Exception as e:
            logger.debug(f"Path analysis failed: {e}")

        # Clustering by semantic similarity
        try:
            semantic_clusters = await self._cluster_by_semantics()
            patterns.extend(semantic_clusters)
        except Exception as e:
            logger.debug(f"Semantic clustering failed: {e}")

        logger.debug(f"Detected {len(patterns)} patterns")
        return patterns

    async def _detect_communities(self) -> list[dict]:
        """Detect communities/clusters in the graph."""
        try:
            # Convert to undirected for community detection
            undirected = self.graph.to_undirected()

            if undirected.number_of_nodes() < 3:
                return []

            # Use Louvain algorithm for community detection
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(undirected)

                # Group nodes by community
                communities = {}
                for node, comm_id in partition.items():
                    if comm_id not in communities:
                        communities[comm_id] = []
                    communities[comm_id].append(node)

                patterns = []
                for comm_id, nodes in communities.items():
                    if len(nodes) >= 3:  # Only consider meaningful communities
                        patterns.append({
                            'type': 'community',
                            'pattern_id': f"community_{comm_id}",
                            'nodes': nodes,
                            'size': len(nodes),
                            'confidence': 0.7
                        })

                return patterns

            except ImportError:
                logger.debug("python-louvain not available, using simple clustering")
                # Fallback to simple connected components
                components = list(nx.connected_components(undirected))
                patterns = []

                for i, component in enumerate(components):
                    if len(component) >= 3:
                        patterns.append({
                            'type': 'community',
                            'pattern_id': f"component_{i}",
                            'nodes': list(component),
                            'size': len(component),
                            'confidence': 0.6
                        })

                return patterns

        except Exception as e:
            logger.debug(f"Community detection failed: {e}")
            return []

    async def _analyze_centrality(self) -> list[dict]:
        """Analyze node centrality to find important entities."""
        patterns = []

        try:
            if self.graph.number_of_nodes() < 2:
                return patterns

            # Calculate different centrality measures
            centralities = {}

            try:
                centralities['betweenness'] = nx.betweenness_centrality(self.graph)
            except Exception as e:
                logger.debug(f"Betweenness centrality failed: {e}")

            try:
                centralities['pagerank'] = nx.pagerank(self.graph)
            except Exception as e:
                logger.debug(f"PageRank centrality failed: {e}")

            try:
                centralities['eigenvector'] = nx.eigenvector_centrality_numpy(self.graph)
            except Exception as e:
                logger.debug(f"Eigenvector centrality failed: {e}")

            for measure_name, centrality_dict in centralities.items():
                # Find top nodes for each centrality measure
                top_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:5]

                for node, score in top_nodes:
                    if score > 0.1:  # Threshold for significance
                        patterns.append({
                            'type': 'centrality',
                            'pattern_id': f"{measure_name}_{node}",
                            'central_node': node,
                            'centrality_type': measure_name,
                            'score': score,
                            'confidence': min(score * 2, 1.0)  # Scale to confidence
                        })

            return patterns

        except Exception as e:
            logger.debug(f"Centrality analysis failed: {e}")
            return []

    async def _find_interesting_paths(self) -> list[dict]:
        """Find interesting paths between entities."""
        patterns = []

        try:
            nodes = list(self.graph.nodes())

            # Sample pairs of nodes to avoid exponential complexity
            import random
            if len(nodes) > 20:
                sample_pairs = random.sample([(a, b) for a in nodes for b in nodes if a != b], 50)
            else:
                sample_pairs = [(a, b) for a in nodes for b in nodes if a != b]

            for source, target in sample_pairs:
                try:
                    if nx.has_path(self.graph, source, target):
                        # Find shortest path
                        path = nx.shortest_path(self.graph, source, target)

                        # Consider paths of length 3-5 as interesting
                        if 3 <= len(path) <= 5:
                            patterns.append({
                                'type': 'path',
                                'pattern_id': f"path_{source}_{target}",
                                'path': path,
                                'length': len(path),
                                'source': source,
                                'target': target,
                                'confidence': 1.0 / len(path)  # Shorter paths higher confidence
                            })
                except nx.NetworkXNoPath:
                    continue

            return patterns

        except Exception as e:
            logger.error(f"Path analysis failed: {e}")
            return []

    async def _cluster_by_semantics(self) -> list[dict]:
        """Cluster entities based on semantic similarity."""
        patterns = []

        try:
            # Get entity descriptions from knowledge graph
            query = """
            MATCH (e:Entity)
            RETURN e.id as entity_id, e.name as name,
                   coalesce(e.description, e.name) as description
            LIMIT 100
            """

            entities = []
            descriptions = []

            async with self.knowledge_graph.driver.session(
                database=self.knowledge_graph.database
            ) as session:
                result = await session.run(query)

                async for record in result:
                    entities.append(record['entity_id'])
                    descriptions.append(record['description'])

            if len(descriptions) < 3:
                return patterns

            # Calculate TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(descriptions)

            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Find clusters of similar entities
            from sklearn.cluster import AgglomerativeClustering
            n_clusters = min(5, len(entities) // 3)

            if n_clusters >= 2:
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                cluster_labels = clustering.fit_predict(similarity_matrix)

                # Group entities by cluster
                clusters = {}
                for i, label in enumerate(cluster_labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(entities[i])

                for cluster_id, cluster_entities in clusters.items():
                    if len(cluster_entities) >= 2:
                        patterns.append({
                            'type': 'semantic_cluster',
                            'pattern_id': f"semantic_cluster_{cluster_id}",
                            'entities': cluster_entities,
                            'size': len(cluster_entities),
                            'confidence': 0.6
                        })

            return patterns

        except Exception as e:
            logger.error(f"Semantic clustering failed: {e}")
            return []

    async def _generate_insight_from_pattern(self, pattern: dict) -> dict | None:
        """Generate an insight from a detected pattern."""
        try:
            pattern_type = pattern['type']

            if pattern_type == 'community':
                return await self._generate_community_insight(pattern)
            elif pattern_type == 'centrality':
                return await self._generate_centrality_insight(pattern)
            elif pattern_type == 'path':
                return await self._generate_path_insight(pattern)
            elif pattern_type == 'semantic_cluster':
                return await self._generate_semantic_insight(pattern)

            return None

        except Exception as e:
            logger.error(f"Failed to generate insight from pattern {pattern.get('pattern_id')}: {e}")
            return None

    async def _generate_community_insight(self, pattern: dict) -> dict:
        """Generate insight from a community pattern."""
        nodes = pattern['nodes']

        # Get entity names for the nodes in the community
        entity_names = await self._get_entity_names(nodes)

        insight_content = f"""Discovered a cluster of {len(nodes)} interconnected entities that form a coherent knowledge community.

Key entities in this cluster:
{', '.join(entity_names[:5])}{'...' if len(entity_names) > 5 else ''}

This clustering suggests these entities share common themes, relationships, or contextual significance that may not be immediately obvious from individual examination. The strength of their interconnections indicates they should be considered together when analyzing related topics."""

        return {
            'zettel_id': f"insight_{uuid.uuid4().hex[:8]}",
            'title': f"Knowledge Community: {entity_names[0]} cluster",
            'content': insight_content,
            'topic': entity_names[0] if entity_names else 'unknown',
            'confidence': pattern['confidence'],
            'pattern_type': 'community_detection',
            'evidence': await self._get_evidence_for_nodes(nodes),
            'metadata': {
                'community_size': len(nodes),
                'entity_count': len(entity_names),
                'pattern_id': pattern['pattern_id']
            }
        }

    async def _generate_centrality_insight(self, pattern: dict) -> dict:
        """Generate insight from a centrality pattern."""
        central_node = pattern['central_node']
        centrality_type = pattern['centrality_type']
        score = pattern['score']

        entity_name = await self._get_entity_name(central_node)

        centrality_descriptions = {
            'betweenness': 'acts as a critical bridge between different parts of the knowledge network',
            'pagerank': 'has high importance based on the network of relationships pointing to it',
            'eigenvector': 'is connected to other highly important entities in the network'
        }

        description = centrality_descriptions.get(centrality_type, 'shows high centrality')

        insight_content = f"""The entity '{entity_name}' demonstrates exceptional structural importance in the knowledge network.

Centrality Analysis:
- Measure: {centrality_type.title()} centrality
- Score: {score:.3f}
- Interpretation: This entity {description}

This high centrality suggests that '{entity_name}' plays a pivotal role in connecting different domains of knowledge and may be a key concept for understanding broader relationships within the knowledge base."""

        return {
            'zettel_id': f"insight_{uuid.uuid4().hex[:8]}",
            'title': f"Central Entity: {entity_name}",
            'content': insight_content,
            'topic': entity_name,
            'confidence': pattern['confidence'],
            'pattern_type': 'centrality_analysis',
            'evidence': await self._get_evidence_for_nodes([central_node]),
            'metadata': {
                'centrality_type': centrality_type,
                'centrality_score': score,
                'entity_id': central_node
            }
        }

    async def _generate_path_insight(self, pattern: dict) -> dict:
        """Generate insight from an interesting path pattern."""
        path = pattern['path']
        source = pattern['source']
        target = pattern['target']

        source_name = await self._get_entity_name(source)
        target_name = await self._get_entity_name(target)
        path_names = await self._get_entity_names(path)

        insight_content = f"""Discovered an interesting connection pathway between '{source_name}' and '{target_name}'.

Connection Path:
{' → '.join(path_names)}

This path reveals a non-obvious relationship that connects these seemingly distant entities through {len(path) - 2} intermediate concept(s). Such paths often indicate:
- Hidden semantic relationships
- Potential areas for knowledge synthesis
- Opportunities for interdisciplinary insights

The existence of this path suggests that research or analysis involving '{source_name}' might benefit from considering connections to '{target_name}' and vice versa."""

        return {
            'zettel_id': f"insight_{uuid.uuid4().hex[:8]}",
            'title': f"Connection Path: {source_name} ↔ {target_name}",
            'content': insight_content,
            'topic': f"{source_name}+{target_name}",
            'confidence': pattern['confidence'],
            'pattern_type': 'path_analysis',
            'evidence': await self._get_evidence_for_nodes(path),
            'metadata': {
                'path_length': len(path),
                'source_entity': source,
                'target_entity': target,
                'intermediate_entities': path[1:-1]
            }
        }

    async def _generate_semantic_insight(self, pattern: dict) -> dict:
        """Generate insight from semantic clustering."""
        entities = pattern['entities']
        entity_names = await self._get_entity_names(entities)

        insight_content = f"""Identified a semantic cluster of {len(entities)} entities that share conceptual similarity despite not being directly connected.

Clustered Entities:
{', '.join(entity_names)}

This semantic clustering suggests these entities:
- Share underlying conceptual themes
- May benefit from being analyzed together
- Could reveal hidden patterns when considered as a group
- Represent different aspects of a common domain

The semantic similarity indicates potential for knowledge synthesis across these entities, even in the absence of explicit relationships."""

        return {
            'zettel_id': f"insight_{uuid.uuid4().hex[:8]}",
            'title': f"Semantic Cluster: {entity_names[0]} group",
            'content': insight_content,
            'topic': entity_names[0] if entity_names else 'semantic_cluster',
            'confidence': pattern['confidence'],
            'pattern_type': 'semantic_clustering',
            'evidence': await self._get_evidence_for_nodes(entities),
            'metadata': {
                'cluster_size': len(entities),
                'entity_list': entities
            }
        }

    async def _get_entity_names(self, entity_ids: list[str]) -> list[str]:
        """Get entity names for a list of entity IDs."""
        if not entity_ids:
            return []

        query = """
        MATCH (e:Entity)
        WHERE e.id IN $entity_ids
        RETURN e.name as name
        """

        names = []
        async with self.knowledge_graph.driver.session(
            database=self.knowledge_graph.database
        ) as session:
            result = await session.run(query, {'entity_ids': entity_ids})
            async for record in result:
                names.append(record['name'])

        return names

    async def _get_entity_name(self, entity_id: str) -> str:
        """Get entity name for a single entity ID."""
        names = await self._get_entity_names([entity_id])
        return names[0] if names else entity_id

    async def _get_evidence_for_nodes(self, nodes: list[str]) -> list[dict]:
        """Get supporting evidence (facts) for a list of nodes."""
        query = """
        MATCH (f:Fact)-[:MENTIONS]->(e:Entity)
        WHERE e.id IN $node_ids
        RETURN f.id as fact_id, f.content as statement, f.source as source
        LIMIT 10
        """

        evidence = []
        async with self.knowledge_graph.driver.session(
            database=self.knowledge_graph.database
        ) as session:
            result = await session.run(query, {'node_ids': nodes})
            async for record in result:
                evidence.append({
                    'fact_id': record['fact_id'],
                    'statement': record['statement'],
                    'source': record['source'],
                    'weight': 1.0
                })

        return evidence

    async def generate_insights(
        self,
        topic: str | None = None,
        confidence_threshold: float = 0.8
    ) -> list[dict]:
        """
        Generate insights on-demand for a specific topic or generally.

        Args:
            topic: Optional topic to focus on
            confidence_threshold: Minimum confidence for insights

        Returns:
            list of generated insights
        """
        logger.info(f"Generating insights for topic: {topic or 'general'}")

        # Refresh analysis graph
        await self._build_analysis_graph()

        # Detect patterns
        patterns = await self._detect_patterns()

        # Filter patterns by topic if specified
        if topic:
            patterns = await self._filter_patterns_by_topic(patterns, topic)

        # Generate insights from patterns
        insights = []
        for pattern in patterns:
            try:
                insight = await self._generate_insight_from_pattern(pattern)
                if insight and insight['confidence'] >= confidence_threshold:
                    insights.append(insight)
            except Exception as e:
                logger.error(f"Failed to generate insight: {e}")

        logger.info(f"Generated {len(insights)} insights")
        return insights

    async def _filter_patterns_by_topic(self, patterns: list[dict], topic: str) -> list[dict]:
        """Filter patterns to those relevant to a specific topic."""
        # Simple implementation - can be enhanced with semantic similarity
        filtered = []
        topic_lower = topic.lower()

        for pattern in patterns:
            if pattern['type'] == 'community':
                entity_names = await self._get_entity_names(pattern['nodes'])
                if any(topic_lower in name.lower() for name in entity_names):
                    filtered.append(pattern)
            elif pattern['type'] == 'centrality':
                entity_name = await self._get_entity_name(pattern['central_node'])
                if topic_lower in entity_name.lower():
                    filtered.append(pattern)
            # Add more filtering logic for other pattern types

        return filtered

    async def search_insights(self, query: str, max_results: int = 10) -> list[dict]:
        """Search for existing insights based on a query."""
        search_query = """
        MATCH (z:Zettel)
        WHERE z.content CONTAINS $query OR z.title CONTAINS $query OR z.topic CONTAINS $query
        OPTIONAL MATCH (z)-[:SUPPORTED_BY]->(f:Fact)
        RETURN z.id as zettel_id, z.title as title, z.content as content,
               z.confidence as confidence, z.pattern_type as pattern_type,
               z.created_at as created_at,
               collect({fact_id: f.id, statement: f.content, source: f.source}) as evidence
        ORDER BY z.confidence DESC
        LIMIT $limit
        """

        insights = []
        async with self.knowledge_graph.driver.session(
            database=self.knowledge_graph.database
        ) as session:
            result = await session.run(search_query, {
                'query': query.lower(),
                'limit': max_results
            })

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

    async def get_statistics(self) -> dict:
        """Get insight engine statistics."""
        # Get counts from knowledge graph
        query = """
        MATCH (z:Zettel)
        RETURN count(z) as total_insights,
               avg(z.confidence) as avg_confidence,
               max(z.created_at) as last_processing
        """

        stats = {
            'total_insights': 0,
            'active_patterns': 0,
            'avg_confidence': 0.0,
            'last_processing': 'Never'
        }

        async with self.knowledge_graph.driver.session(
            database=self.knowledge_graph.database
        ) as session:
            result = await session.run(query)
            record = await result.single()

            if record:
                stats['total_insights'] = record['total_insights'] or 0
                stats['avg_confidence'] = record['avg_confidence'] or 0.0
                if record['last_processing']:
                    stats['last_processing'] = datetime.fromtimestamp(
                        record['last_processing'] / 1000
                    ).strftime('%Y-%m-%d %H:%M:%S')

        stats['active_patterns'] = len(await self._detect_patterns())

        return stats
