"""
Zettelkasten-inspired Insight Engine for Project Synapse.

This module implements the autonomous synthesis engine that identifies patterns
and generates novel insights from the knowledge graph.
"""

import asyncio
import os
import random
import uuid
from datetime import datetime
from typing import Any

import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import community as community_louvain
except ImportError:
    community_louvain = None

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class InsightEngine:
    """
    Zettelkasten-inspired engine for autonomous insight generation.

    Implements pattern detection and knowledge synthesis using graph algorithms
    and machine learning to identify non-obvious connections and generate insights.
    """

    @logger.timer()
    async def get_insights_by_topic(
        self, topic: str, max_results: int = 20
    ) -> list[dict[str, Any]]:
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
        WHERE toLower(z.topic) CONTAINS $topic
           OR toLower(z.title) CONTAINS $topic
           OR toLower(z.content) CONTAINS $topic
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
            result = await session.run(
                query, {"topic": topic.lower(), "limit": max_results}
            )

            async for record in result:
                insights.append(
                    {
                        "zettel_id": record["zettel_id"],
                        "title": record["title"],
                        "content": record["content"],
                        "confidence": record["confidence"],
                        "pattern_type": record["pattern_type"],
                        "created_at": record["created_at"],
                        "evidence": [e for e in record["evidence"] if e["fact_id"]],
                    }
                )

        return insights

    def __init__(self, knowledge_graph: Any, montague_parser: Any) -> None:
        self.knowledge_graph = knowledge_graph
        self.montague_parser = montague_parser
        self.graph = nx.DiGraph()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        self.is_running = False
        self.processing_interval = int(os.getenv("PATTERN_DETECTION_INTERVAL", "300"))

        # Insight generation settings
        self.confidence_threshold = float(
            os.getenv("INSIGHT_CONFIDENCE_THRESHOLD", "0.8")
        )
        self.link_threshold = float(os.getenv("LINK_THRESHOLD", "0.7"))
        self.logger = logger

    @logger.timer()
    async def initialize(self) -> None:
        """Initialize the insight engine."""
        try:
            await self._build_analysis_graph()
            logger.info("Insight engine initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize insight engine: %s", e)
            raise

    async def cleanup(self) -> None:
        """Clean up the insight engine."""
        self.is_running = False
        logger.info("Insight engine cleanup completed")

    async def start_autonomous_processing(self) -> None:
        """Start autonomous insight generation in background.

        Note on startup delay: ``_autonomous_processing_cycle`` is heavy
        (graph-wide NetworkX analysis on every node + edge). If it fires
        immediately when the MCP server starts, the event loop stays saturated
        long enough that the MCP stdio reader can't keep up with the client,
        the client decides the server is dead, and the connection drops with
        ``anyio.BrokenResourceError``. The initial sleep lets the server
        finish its handshake and serve a few requests cleanly before insight
        work begins competing for the event loop.
        """
        self.is_running = True
        logger.info("Starting autonomous insight processing")

        initial_delay = int(os.getenv("SYNAPSE_INSIGHT_STARTUP_DELAY", "120"))
        if initial_delay > 0:
            logger.info(
                f"First insight cycle will run after {initial_delay}s startup delay"
            )
            try:
                await asyncio.sleep(initial_delay)
            except asyncio.CancelledError:
                logger.info("Autonomous processing cancelled before first cycle")
                return

        while self.is_running:
            try:
                await self._autonomous_processing_cycle()
                await asyncio.sleep(self.processing_interval)
            except asyncio.CancelledError:
                logger.info("Autonomous processing cancelled")
                break
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Error in autonomous processing: %s", e)
                await asyncio.sleep(60)  # Wait before retrying

    @logger.timer()
    async def _autonomous_processing_cycle(self) -> None:
        """Run one cycle of autonomous insight generation."""
        logger.debug("Running autonomous insight generation cycle")

        # Rebuild analysis graph with latest data
        await self._build_analysis_graph()

        # Detect patterns using various algorithms
        patterns: list[dict[str, Any]] = await self._detect_patterns()

        # Generate insights from detected patterns
        for pattern in patterns:
            try:
                insight = await self._generate_insight_from_pattern(pattern)
                if insight and insight["confidence"] >= self.confidence_threshold:
                    # Store insight in knowledge graph
                    zettel_id = await self.knowledge_graph.store_insight(insight)
                    logger.info("Generated new insight with Zettel ID: %s", zettel_id)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Failed to generate insight from pattern: %s", e)

    async def _build_analysis_graph(self) -> None:
        """Build NetworkX graph from knowledge graph for analysis."""
        self.graph.clear()

        # Get all entities and relationships from knowledge graph
        query = """
        MATCH (a)-[r]->(b)
        WHERE a.id IS NOT NULL AND b.id IS NOT NULL
        RETURN a.id as source, b.id as target, type(r) as rel_type,
               coalesce(r.confidence, 1.0) as confidence
        """

        try:
            async with self.knowledge_graph.driver.session(
                database=self.knowledge_graph.database
            ) as session:
                result = await session.run(query)

                edge_count = 0
                async for record in result:
                    source = record["source"]
                    target = record["target"]

                    # Only add edge if both source and target are valid
                    if source is not None and target is not None:
                        self.graph.add_edge(
                            source,
                            target,
                            rel_type=record["rel_type"] or "RELATES",
                            weight=record["confidence"] or 1.0,
                        )
                        edge_count += 1

            logger.debug(
                "Built analysis graph with %d nodes, %d edges",
                self.graph.number_of_nodes(),
                self.graph.number_of_edges(),
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Could not build analysis graph from knowledge base: %s", e)
            logger.debug("Starting with empty analysis graph")

    @logger.timer()
    async def _detect_patterns(self) -> list[dict[str, Any]]:
        """Detect patterns in the knowledge graph using various algorithms."""
        patterns: list[dict[str, Any]] = []

        if self.graph.number_of_nodes() < 2:
            logger.debug("Graph too small for pattern detection, skipping")
            return patterns

        # Community detection
        try:
            communities = await self._detect_communities()
            patterns.extend(communities)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("Community detection failed: %s", e)

        # Centrality analysis
        try:
            central_nodes = await self._analyze_centrality()
            patterns.extend(central_nodes)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("Centrality analysis failed: %s", e)

        # Path analysis
        try:
            interesting_paths = await self._find_interesting_paths()
            patterns.extend(interesting_paths)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("Path analysis failed: %s", e)

        # Clustering by semantic similarity
        try:
            semantic_clusters = await self._cluster_by_semantics()
            patterns.extend(semantic_clusters)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("Semantic clustering failed: %s", e)

        logger.debug("Detected %d patterns", len(patterns))
        return patterns

    @logger.timer()
    async def _detect_communities(self) -> list[dict[str, Any]]:
        """Detect communities/clusters in the graph."""
        try:
            # Convert to undirected for community detection
            undirected = self.graph.to_undirected()

            if undirected.number_of_nodes() < 3:
                return []

            patterns: list[dict[str, Any]] = []

            # Use Louvain algorithm for community detection
            if community_louvain:
                partition = community_louvain.best_partition(undirected)

                # Group nodes by community
                communities: dict[int, list[str]] = {}
                for node, comm_id in partition.items():
                    if comm_id not in communities:
                        communities[comm_id] = []
                    communities[comm_id].append(node)

                for comm_id, nodes in communities.items():
                    if len(nodes) >= 3:  # Only consider meaningful communities
                        patterns.append(
                            {
                                "type": "community",
                                "pattern_id": f"community_{comm_id}",
                                "nodes": nodes,
                                "size": len(nodes),
                                "confidence": 0.7,
                            }
                        )

                return patterns
            else:
                logger.debug("python-louvain not available, using simple clustering")
                # Fallback to simple connected components
                components = list(nx.connected_components(undirected))

                for i, component in enumerate(components):
                    if len(component) >= 3:
                        patterns.append(
                            {
                                "type": "community",
                                "pattern_id": f"component_{i}",
                                "nodes": list(component),
                                "size": len(component),
                                "confidence": 0.6,
                            }
                        )

                return patterns

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("Community detection failed: %s", e)
            return []

    @logger.timer()
    async def _analyze_centrality(self) -> list[dict[str, Any]]:
        """Analyze node centrality to find important entities."""
        patterns: list[dict[str, Any]] = []

        try:
            if self.graph.number_of_nodes() < 2:
                return patterns

            # Calculate different centrality measures
            centralities: dict[str, dict[Any, float]] = {}

            try:
                centralities["betweenness"] = nx.betweenness_centrality(self.graph)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.debug("Betweenness centrality failed: %s", e)

            try:
                centralities["pagerank"] = nx.pagerank(self.graph)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.debug("PageRank centrality failed: %s", e)

            try:
                centralities["eigenvector"] = nx.eigenvector_centrality_numpy(
                    self.graph
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.debug("Eigenvector centrality failed: %s", e)

            for measure_name, centrality_dict in centralities.items():
                # Find top nodes for each centrality measure
                top_nodes = sorted(
                    centrality_dict.items(), key=lambda x: x[1], reverse=True
                )[:5]

                for node, score in top_nodes:
                    if score > 0.1:  # Threshold for significance
                        patterns.append(
                            {
                                "type": "centrality",
                                "pattern_id": f"{measure_name}_{node}",
                                "central_node": node,
                                "centrality_type": measure_name,
                                "score": score,
                                "confidence": min(
                                    score * 2, 1.0
                                ),  # Scale to confidence
                            }
                        )

            return patterns

        except Exception as e:
            logger.debug(f"Centrality analysis failed: {e}")
            return []

    @logger.timer()
    async def _find_interesting_paths(self) -> list[dict[str, Any]]:
        """Find interesting paths between entities."""
        patterns: list[dict[str, Any]] = []

        try:
            nodes = list(self.graph.nodes())

            # Sample pairs of nodes to avoid exponential complexity
            if len(nodes) > 20:
                # Use SystemRandom for security linting compliance if needed
                secure_random = random.SystemRandom()
                sample_pairs = secure_random.sample(
                    [(a, b) for a in nodes for b in nodes if a != b], 50
                )
            else:
                sample_pairs = [(a, b) for a in nodes for b in nodes if a != b]

            for source, target in sample_pairs:
                try:
                    if nx.has_path(self.graph, source, target):
                        # Find shortest path
                        path = nx.shortest_path(self.graph, source, target)

                        # Consider paths of length 3-5 as interesting
                        if 3 <= len(path) <= 5:
                            patterns.append(
                                {
                                    "type": "path",
                                    "pattern_id": f"path_{source}_{target}",
                                    "path": path,
                                    "length": len(path),
                                    "source": source,
                                    "target": target,
                                    "confidence": 1.0
                                    / len(path),  # Shorter paths = higher confidence
                                }
                            )
                except nx.NetworkXNoPath:
                    continue

            return patterns

        except Exception as e:
            logger.error(f"Path analysis failed: {e}")
            return []

    @logger.timer()
    async def _cluster_by_semantics(self) -> list[dict[str, Any]]:
        """Cluster entities based on semantic similarity."""
        patterns: list[dict[str, Any]] = []

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
                    entity_id = record["entity_id"]
                    description = record["description"]

                    # Only add entities with valid descriptions
                    is_valid = (
                        entity_id
                        and description
                        and isinstance(description, str)
                        and description.strip()
                    )
                    if is_valid:
                        entities.append(entity_id)
                        descriptions.append(description.strip())

            if len(descriptions) < 3:
                logger.debug(
                    "Not enough entities with descriptions for clustering: %d",
                    len(descriptions),
                )
                return patterns

            # Calculate TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(descriptions)

            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Find clusters of similar entities
            n_clusters = min(5, len(entities) // 3)

            if n_clusters >= 2:
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                cluster_labels = clustering.fit_predict(similarity_matrix)

                # Group entities by cluster
                clusters: dict[int, list[str]] = {}
                for i, label in enumerate(cluster_labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(entities[i])

                for cluster_id, cluster_entities in clusters.items():
                    # Ensure we have valid entities
                    if len(cluster_entities) >= 2 and cluster_entities:
                        # Filter out any None values from cluster_entities
                        valid_entities = [e for e in cluster_entities if e is not None]
                        if valid_entities:
                            patterns.append(
                                {
                                    "type": "semantic_cluster",
                                    "pattern_id": f"semantic_cluster_{cluster_id}",
                                    "entities": valid_entities,
                                    "size": len(valid_entities),
                                    "confidence": 0.6,
                                }
                            )

            return patterns

        except Exception as e:
            logger.error(f"Semantic clustering failed: {e}")
            return []

    async def _generate_insight_from_pattern(
        self, pattern: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Generate an insight from a detected pattern."""
        try:
            pattern_type = pattern["type"]

            if pattern_type == "community":
                return await self._generate_community_insight(pattern)
            elif pattern_type == "centrality":
                return await self._generate_centrality_insight(pattern)
            elif pattern_type == "path":
                return await self._generate_path_insight(pattern)
            elif pattern_type == "semantic_cluster":
                return await self._generate_semantic_insight(pattern)

            return None

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Failed to generate insight from pattern %s: %s",
                pattern.get("pattern_id"),
                e,
            )
            return None

    async def _generate_community_insight(
        self, pattern: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate insight from a community pattern."""
        nodes = pattern["nodes"]

        # Get entity names for the nodes in the community
        entity_names = await self._get_entity_names(nodes)

        # Fallback to node IDs if no entity names found
        if not entity_names:
            entity_names = [f"entity_{node}" for node in nodes[:5] if node]

        # Ensure we have at least one valid name
        if not entity_names:
            entity_names = [f"community_{pattern.get('pattern_id', 'unknown')}"]

        # Ensure we have at least one name for the title
        primary_name = (
            entity_names[0]
            if entity_names
            else f"community_{pattern.get('pattern_id', 'unknown')}"
        )

        # Clean entity names for display (filter out any remaining None/empty values)
        display_names = [
            name
            for name in entity_names
            if name and isinstance(name, str) and name.strip()
        ]
        if not display_names:
            display_names = [primary_name]

        insight_content = (
            f"Discovered a cluster of {len(nodes)} interconnected entities that "
            f"form a coherent knowledge community.\n\n"
            f"Key entities in this cluster:\n"
            f"{', '.join(display_names[:5])}{'...' if len(display_names) > 5 else ''}\n\n"
            f"This clustering suggests these entities share common themes, relationships, "
            f"or contextual significance that may not be immediately obvious from individual "
            f"examination. The strength of their interconnections indicates they should be "
            f"considered together when analyzing related topics."
        )

        return {
            "zettel_id": f"insight_{uuid.uuid4().hex[:8]}",
            "title": f"Knowledge Community: {primary_name} cluster",
            "content": insight_content,
            "topic": primary_name,
            "confidence": pattern["confidence"],
            "pattern_type": "community_detection",
            "evidence": await self._get_evidence_for_nodes(nodes),
            "metadata": {
                "community_size": len(nodes),
                "entity_count": len(display_names),
                "pattern_id": pattern["pattern_id"],
            },
        }

    async def _generate_centrality_insight(
        self, pattern: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate insight from a centrality pattern."""
        central_node = pattern["central_node"]
        centrality_type = pattern["centrality_type"]
        score = pattern["score"]

        entity_name = await self._get_entity_name(central_node)

        # Fallback to node ID if no entity name found
        if not entity_name or entity_name == central_node:
            entity_name = f"entity_{central_node}"

        centrality_descriptions = {
            "betweenness": "acts as a critical bridge between different parts of the knowledge "
            "network",
            "pagerank": "has high importance based on the network of relationships pointing to it",
            "eigenvector": "is connected to other highly important entities in the network",
        }

        description = centrality_descriptions.get(
            centrality_type, "shows high centrality"
        )

        insight_content = (
            f"The entity '{entity_name}' demonstrates exceptional structural importance in the "
            f"knowledge network.\n\n"
            f"Centrality Analysis:\n"
            f"- Measure: {centrality_type.title()} centrality\n"
            f"- Score: {score:.3f}\n"
            f"- Interpretation: This entity {description}\n\n"
            f"This high centrality suggests that '{entity_name}' plays a pivotal role in "
            f"connecting different domains of knowledge and may be a key concept for "
            f"understanding broader relationships within the knowledge base."
        )

        return {
            "zettel_id": f"insight_{uuid.uuid4().hex[:8]}",
            "title": f"Central Entity: {entity_name}",
            "content": insight_content,
            "topic": entity_name,
            "confidence": pattern["confidence"],
            "pattern_type": "centrality_analysis",
            "evidence": await self._get_evidence_for_nodes([central_node]),
            "metadata": {
                "centrality_type": centrality_type,
                "centrality_score": score,
                "entity_id": central_node,
            },
        }

    async def _generate_path_insight(self, pattern: dict[str, Any]) -> dict[str, Any]:
        """Generate insight from an interesting path pattern."""
        path = pattern["path"]
        source = pattern["source"]
        target = pattern["target"]

        source_name = await self._get_entity_name(source)
        target_name = await self._get_entity_name(target)
        path_names = await self._get_entity_names(path)

        # Fallback to node IDs if no entity names found
        if not source_name or source_name == source:
            source_name = f"entity_{source}"
        if not target_name or target_name == target:
            target_name = f"entity_{target}"
        if not path_names:
            path_names = [f"entity_{node}" for node in path]

        insight_content = (
            f"Discovered an interesting connection pathway between '{source_name}' and "
            f"'{target_name}'.\n\n"
            f"Connection Path:\n"
            f"{' → '.join(path_names)}\n\n"
            f"This path reveals a non-obvious relationship that connects these seemingly "
            f"distant entities through {len(path) - 2} intermediate concept(s). Such "
            f"paths often indicate:\n"
            f"- Hidden semantic relationships\n"
            f"- Potential areas for knowledge synthesis\n"
            f"- Opportunities for interdisciplinary insights\n\n"
            f"The existence of this path suggests that research or analysis involving "
            f"'{source_name}' might benefit from considering connections to '{target_name}' "
            f"and vice versa."
        )

        return {
            "zettel_id": f"insight_{uuid.uuid4().hex[:8]}",
            "title": f"Connection Path: {source_name} ↔ {target_name}",
            "content": insight_content,
            "topic": f"{source_name}+{target_name}",
            "confidence": pattern["confidence"],
            "pattern_type": "path_analysis",
            "evidence": await self._get_evidence_for_nodes(path),
            "metadata": {
                "path_length": len(path),
                "source_entity": source,
                "target_entity": target,
                "intermediate_entities": path[1:-1],
            },
        }

    async def _generate_semantic_insight(
        self, pattern: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate insight from semantic clustering."""
        entities = pattern["entities"]
        entity_names = await self._get_entity_names(entities)

        # Fallback to entity IDs if no entity names found
        if not entity_names:
            entity_names = [f"entity_{entity}" for entity in entities if entity]

        # Ensure we have at least one valid name
        if not entity_names:
            entity_names = [f"cluster_{pattern.get('pattern_id', 'unknown')}"]

        # Ensure we have at least one name for the title and topic
        default_name = f"semantic_cluster_{pattern.get('pattern_id', 'unknown')}"
        primary_name = entity_names[0] if entity_names else default_name

        # Clean entity names for display (filter out any remaining None/empty values)
        display_names = [
            name
            for name in entity_names
            if name and isinstance(name, str) and name.strip()
        ]
        if not display_names:
            display_names = [primary_name]

        insight_content = (
            f"Identified a semantic cluster of {len(entities)} entities that "
            f"share conceptual similarity despite not being directly connected.\n\n"
            f"Clustered Entities:\n"
            f"{', '.join(display_names)}\n\n"
            f"This semantic clustering suggests these entities:\n"
            f"- Share underlying conceptual themes\n"
            f"- May benefit from being analyzed together\n"
            f"- Could reveal hidden patterns when considered as a group\n"
            f"- Represent different aspects of a common domain\n\n"
            f"The semantic similarity indicates potential for knowledge synthesis across "
            f"these entities, even in the absence of explicit relationships."
        )

        return {
            "zettel_id": f"insight_{uuid.uuid4().hex[:8]}",
            "title": f"Semantic Cluster: {primary_name} group",
            "content": insight_content,
            "topic": primary_name,
            "confidence": pattern["confidence"],
            "pattern_type": "semantic_clustering",
            "evidence": await self._get_evidence_for_nodes(entities),
            "metadata": {"cluster_size": len(entities), "entity_list": entities},
        }

    async def _get_entity_names(self, entity_ids: list[str]) -> list[str]:
        """Get entity names for a list of entity IDs."""
        if not entity_ids:
            return []

        query = """
        MATCH (e:Entity)
        WHERE e.id IN $entity_ids
        RETURN e.name as name, e.id as id
        """

        names = []
        async with self.knowledge_graph.driver.session(
            database=self.knowledge_graph.database
        ) as session:
            try:
                result = await session.run(query, {"entity_ids": entity_ids})
                async for record in result:
                    # Filter out None values and ensure we have valid strings
                    name = record["name"]
                    if name is not None and isinstance(name, str) and name.strip():
                        names.append(name.strip())
                    else:
                        # Fallback to entity ID if name is invalid
                        entity_id = record["id"]
                        if entity_id and isinstance(entity_id, str):
                            # Clean up entity ID to make it more readable
                            readable_name = (
                                entity_id.replace("_", " ").replace("-", " ").title()
                            )
                            names.append(readable_name)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.debug("Could not retrieve entity names: %s", e)
                # Fallback: return cleaned entity IDs
                for entity_id in entity_ids:
                    if entity_id and isinstance(entity_id, str):
                        readable_name = (
                            entity_id.replace("_", " ").replace("-", " ").title()
                        )
                        names.append(readable_name)

        return names

    async def _get_entity_name(self, entity_id: str) -> str:
        """Get entity name for a single entity ID."""
        names = await self._get_entity_names([entity_id])
        # Return the first valid name or fallback to a cleaned entity_id
        if names:
            return names[0]
        # Fallback: clean up the entity_id to make it more readable
        if entity_id:
            return entity_id.replace("_", " ").replace("-", " ").title()
        return "Unknown Entity"

    async def _get_evidence_for_nodes(self, nodes: list[str]) -> list[dict[str, Any]]:
        """Get supporting evidence (facts) for a list of nodes."""
        if not nodes:
            return []

        # First check if there are any Facts and MENTIONS relationships
        check_query = """
        MATCH (f:Fact)
        OPTIONAL MATCH (f)-[r:MENTIONS]->(e:Entity)
        WHERE e.id IN $node_ids
        RETURN count(f) as fact_count, count(r) as mention_count
        LIMIT 1
        """

        query = """
        MATCH (f:Fact)-[:MENTIONS]->(e:Entity)
        WHERE e.id IN $node_ids
        RETURN f.id as fact_id, f.content as statement, f.source as source
        LIMIT 10
        """

        evidence: list[dict[str, Any]] = []
        async with self.knowledge_graph.driver.session(
            database=self.knowledge_graph.database
        ) as session:
            try:
                # Check if we have facts and mentions relationships
                check_result = await session.run(check_query, {"node_ids": nodes})
                check_record = await check_result.single()

                if check_record and check_record["mention_count"] > 0:
                    # We have MENTIONS relationships, use the normal query
                    result = await session.run(query, {"node_ids": nodes})
                    async for record in result:
                        evidence.append(
                            {
                                "fact_id": record["fact_id"],
                                "statement": record["statement"],
                                "source": record["source"],
                                "weight": 1.0,
                            }
                        )
                else:
                    # No MENTIONS relationships exist yet, try alternative evidence
                    logger.debug("No MENTIONS relationships found, using alternatives")

                    # Look for facts that might contain these node IDs
                    alt_query = """
                    MATCH (f:Fact)
                    WHERE any(node_id IN $node_ids
                          WHERE f.content CONTAINS node_id
                             OR f.entity_list CONTAINS node_id)
                    RETURN f.id as fact_id, f.content as statement, f.source as source
                    LIMIT 5
                    """

                    try:
                        alt_result = await session.run(alt_query, {"node_ids": nodes})
                        async for record in alt_result:
                            evidence.append(
                                {
                                    "fact_id": record["fact_id"],
                                    "statement": record["statement"],
                                    "source": record["source"],
                                    "weight": 0.5,  # Lower weight for indirect evidence
                                }
                            )
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        logger.debug("Alternative evidence query failed: %s", e)

            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.debug("Evidence retrieval failed: %s", e)
                # Return empty evidence rather than crashing

        return evidence

    async def generate_insights(
        self, topic: str | None = None, confidence_threshold: float = 0.8
    ) -> list[dict[str, Any]]:
        """
        Generate insights on-demand for a specific topic or generally.

        Args:
            topic: Optional topic to focus on
            confidence_threshold: Minimum confidence for insights

        Returns:
            list of generated insights
        """
        logger.info("Generating insights for topic: %s", topic or "general")

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
                if insight and insight["confidence"] >= confidence_threshold:
                    insights.append(insight)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Failed to generate insight: %s", e)

        logger.info("Generated %d insights", len(insights))
        return insights

    async def _filter_patterns_by_topic(
        self, patterns: list[dict[str, Any]], topic: str
    ) -> list[dict[str, Any]]:
        """Filter patterns to those relevant to a specific topic."""
        # Simple implementation - can be enhanced with semantic similarity
        filtered = []
        topic_lower = topic.lower()

        for pattern in patterns:
            if pattern["type"] == "community":
                entity_names = await self._get_entity_names(pattern["nodes"])
                if any(topic_lower in name.lower() for name in entity_names):
                    filtered.append(pattern)
            elif pattern["type"] == "centrality":
                entity_name = await self._get_entity_name(pattern["central_node"])
                if topic_lower in entity_name.lower():
                    filtered.append(pattern)
            # Add more filtering logic for other pattern types

        return filtered

    async def search_insights(
        self, query: str, max_results: int = 10
    ) -> list[dict[str, Any]]:
        """Search for existing insights based on a query."""
        search_query = """
        MATCH (z:Zettel)
        WHERE z.content CONTAINS $query
           OR z.title CONTAINS $query
           OR z.topic CONTAINS $query
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
            result = await session.run(
                search_query, {"query": query.lower(), "limit": max_results}
            )

            async for record in result:
                insights.append(
                    {
                        "zettel_id": record["zettel_id"],
                        "title": record["title"],
                        "content": record["content"],
                        "confidence": record["confidence"],
                        "pattern_type": record["pattern_type"],
                        "created_at": record["created_at"],
                        "evidence": [e for e in record["evidence"] if e["fact_id"]],
                    }
                )

        return insights

    async def get_statistics(self) -> dict[str, Any]:
        """Get insight engine statistics."""
        # Get counts from knowledge graph
        query = """
        MATCH (z:Zettel)
        RETURN count(z) as total_insights,
               avg(z.confidence) as avg_confidence,
               max(z.created_at) as last_processing
        """

        stats: dict[str, Any] = {
            "total_insights": 0,
            "active_patterns": 0,
            "avg_confidence": 0.0,
            "last_processing": "Never",
        }

        async with self.knowledge_graph.driver.session(
            database=self.knowledge_graph.database
        ) as session:
            result = await session.run(query)
            record = await result.single()

            if record:
                stats["total_insights"] = record["total_insights"] or 0
                stats["avg_confidence"] = record["avg_confidence"] or 0.0
                if record["last_processing"]:
                    last_processing_val = record["last_processing"]
                    stats["last_processing"] = datetime.fromtimestamp(
                        last_processing_val / 1000
                    ).strftime("%Y-%m-%d %H:%M:%S")

        stats["active_patterns"] = len(await self._detect_patterns())

        return stats
