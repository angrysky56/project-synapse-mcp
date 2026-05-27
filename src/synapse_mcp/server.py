"""
Project Synapse MCP Server

Main server implementation providing autonomous knowledge synthesis capabilities
through the Model Context Protocol (MCP).
"""

# pylint: disable=too-many-lines

import asyncio
import atexit
import os
import re
import shutil
import signal

# trunk-ignore(bandit/B404)
import subprocess
import sys
import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts import base

# Import our core modules
from .core.knowledge_graph import KnowledgeGraph
from .core.temporal_facts import TemporalFact, TemporalFactStore
from .data_pipeline.semantic_integrator import SemanticIntegrator
from .data_pipeline.text_processor import TextProcessor
from .semantic.montague_parser import MontagueParser
from .utils.logging_config import quiet_chatty_loggers, setup_logging
from .wiki.wiki_adapter import WikiAdapter
from .zettelkasten.insight_engine import InsightEngine

# Load environment variables
load_dotenv()

# Configure logging for MCP (stderr only)
logger = setup_logging(__name__)
# Mute Neo4j's INFO-level "index/constraint already exists, IF NOT EXISTS
# worked correctly" notifications — they're spam, not signal.
quiet_chatty_loggers()


class SynapseServer:
    """Main Project Synapse server class managing all components."""

    def __init__(self) -> None:
        self.knowledge_graph: KnowledgeGraph | None = None
        self.montague_parser: MontagueParser | None = None
        self.insight_engine: InsightEngine | None = None
        self.text_processor: TextProcessor | None = None
        self.semantic_integrator: SemanticIntegrator | None = None
        self.wiki_adapter: WikiAdapter | None = None
        # Bitemporal episodic-memory store. Shares Neo4j driver with the
        # main knowledge graph; initialised after the KG connects.
        self.temporal_facts: TemporalFactStore | None = None
        self.background_tasks: set = set()
        self.logger = logger

    def set_context(self, ctx: Context) -> None:
        """Propagate MCP context to all components for status updates."""
        self.logger.set_context(ctx)
        components = [
            self.knowledge_graph,
            self.wiki_adapter,
            self.insight_engine,
            self.text_processor,
            self.semantic_integrator,
            self.montague_parser,
        ]
        for comp in components:
            if comp and hasattr(comp, "logger") and hasattr(comp.logger, "set_context"):
                comp.logger.set_context(ctx)

    async def initialize(self) -> None:
        """Initialize all server components."""
        logger.info("Initializing Project Synapse server components")

        try:
            # Initialize knowledge graph
            self.knowledge_graph = KnowledgeGraph()
            await self.knowledge_graph.connect()

            # Initialize temporal-fact store (shares the KG's Neo4j driver
            # so we don't open a second bolt pool).
            if self.knowledge_graph.driver is not None:
                self.temporal_facts = TemporalFactStore(
                    self.knowledge_graph.driver,
                    self.knowledge_graph.database,
                )
                await self.temporal_facts.initialize_schema()

            # Initialize semantic parser
            self.montague_parser = MontagueParser()
            await self.montague_parser.initialize()

            # Initialize text processor
            self.text_processor = TextProcessor()
            await self.text_processor.initialize()

            # Initialize semantic integrator
            self.semantic_integrator = SemanticIntegrator(self.montague_parser)
            await self.semantic_integrator.initialize()

            # Initialize insight engine
            self.insight_engine = InsightEngine(
                knowledge_graph=self.knowledge_graph,
                montague_parser=self.montague_parser,
            )
            await self.insight_engine.initialize()

            # Initialize wiki adapter
            self.wiki_adapter = WikiAdapter()
            await self.wiki_adapter.initialize()

            # Final health check
            await self.check_health()
            logger.info("All components initialized successfully")

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to initialize server components: %s", e)
            raise

    async def check_health(self) -> dict[str, Any]:
        """Check health of all server components."""
        health_status: dict[str, Any] = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {},
        }

        # Knowledge Graph check
        try:
            if self.knowledge_graph:
                await self.knowledge_graph.check_health()
                health_status["components"]["knowledge_graph"] = "healthy"
            else:
                health_status["components"]["knowledge_graph"] = "not_initialized"
                health_status["status"] = "unhealthy"
        except Exception as e:  # pylint: disable=broad-exception-caught
            health_status["components"]["knowledge_graph"] = f"unhealthy: {str(e)}"
            health_status["status"] = "unhealthy"

        # Wiki Adapter check
        try:
            if self.wiki_adapter:
                await self.wiki_adapter.check_health()
                health_status["components"]["wiki_adapter"] = "healthy"
            else:
                health_status["components"]["wiki_adapter"] = "not_initialized"
                health_status["status"] = "unhealthy"
        except Exception as e:  # pylint: disable=broad-exception-caught
            health_status["components"]["wiki_adapter"] = f"unhealthy: {str(e)}"
            health_status["status"] = "unhealthy"

        if health_status["status"] == "unhealthy":
            logger.error("Server health check failed: %s", health_status)
        else:
            logger.info("Server health check passed")

        return health_status

    async def cleanup(self) -> None:
        """Clean up all server resources."""
        logger.info("Cleaning up Project Synapse server")

        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        self.background_tasks.clear()

        # Close component connections
        if self.knowledge_graph:
            await self.knowledge_graph.close()
        if self.insight_engine:
            await self.insight_engine.cleanup()

        logger.info("Server cleanup completed")


# Global server instance
synapse_server = SynapseServer()


@asynccontextmanager
async def lifespan_context(_mcp_app: FastMCP) -> AsyncGenerator[dict[str, Any], None]:
    """Manage server lifecycle with proper cleanup."""
    logger.info("Starting Project Synapse MCP server")

    try:
        # Initialize server
        await synapse_server.initialize()

        # Start background insight generation if insight_engine is initialized and enabled
        if synapse_server.insight_engine is not None:
            if os.getenv("SYNAPSE_AUTONOMOUS_INSIGHTS", "off").lower() == "on":
                insight_task = asyncio.create_task(
                    synapse_server.insight_engine.start_autonomous_processing()
                )
                synapse_server.background_tasks.add(insight_task)
                logger.info(
                    "Autonomous insight engine started (SYNAPSE_AUTONOMOUS_INSIGHTS=on)"
                )
            else:
                logger.info(
                    "Autonomous insight engine disabled by default "
                    "(SYNAPSE_AUTONOMOUS_INSIGHTS=off)"
                )

        yield {"synapse": synapse_server}

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Server startup failed: %s", e)
        raise
    finally:
        await synapse_server.cleanup()


# Initialize MCP server with lifespan management
mcp = FastMCP(name="project-synapse", lifespan=lifespan_context)
# =============================================================================
# MCP TOOLS - Knowledge Synthesis and Retrieval
# =============================================================================


@mcp.tool()
async def debug_test() -> str:
    """Simple test tool to check if MCP server is working."""
    try:
        logger.info("Debug test tool called")
        return "✅ MCP server is working correctly!"
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Debug test failed: %s", e, exc_info=True)
        return f"❌ Debug test failed: {type(e).__name__}: {str(e)}"


@mcp.tool()
async def ingest_text(
    ctx: Context,
    text: str,
    source: str = "user_input",
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Ingest and process text into the knowledge graph using semantic analysis.

    This tool performs the core knowledge synthesis pipeline:
    1. Semantic parsing using Montague Grammar
    2. Entity extraction and relationship identification
    3. Storage in the Neo4j knowledge graph
    4. Automatic insight generation triggers

    Args:
        text: Raw text to process and analyze
        source: Source identifier for provenance tracking
        metadata: Additional metadata about the text

    Returns:
        Processing summary with entities and relationships extracted
    """
    try:
        await ctx.info(f"Ingesting text from source: {source}")
        logger.info("Starting text ingestion: %s...", text[:50])

        synapse = ctx.request_context.lifespan_context["synapse"]
        synapse.set_context(ctx)
        logger.info("Retrieved synapse server instance")

        # Process text through semantic pipeline
        logger.info("About to call semantic integrator...")
        try:
            processed_data = (
                await synapse.semantic_integrator.process_text_with_semantics(
                    text, source, metadata or {}
                )
            )
            logger.info("Semantic processing completed successfully")
        except Exception as semantic_error:
            logger.error("Semantic processing failed: %s", semantic_error)
            traceback.print_exc()
            raise

        # Store in knowledge graph
        logger.info("About to store in knowledge graph...")
        try:
            result = await synapse.knowledge_graph.store_processed_data(processed_data)
            logger.info("Knowledge graph storage completed successfully")
        except Exception as storage_error:
            logger.error("Knowledge graph storage failed: %s", storage_error)
            traceback.print_exc()
            raise

        await ctx.info("Text ingestion completed successfully")

        return (
            "Text ingestion completed successfully.\n\n"
            "Extracted:\n"
            f"- {result['entities_count']} entities\n"
            f"- {result['relationships_count']} relationships\n"
            f"- {result['facts_count']} semantic facts\n\n"
            f"Knowledge graph updated with {result['new_nodes']} new nodes and "
            f"{result['new_edges']} new edges.\n"
            "Automatic insight generation triggered."
        )

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Text ingestion failed: %s", e, exc_info=True)
        return f"❌ Text ingestion failed [{type(e).__name__}]: {str(e)}"


@mcp.tool()
async def generate_insights(
    ctx: Context, topic: str | None = None, confidence_threshold: float = 0.8
) -> str:
    """
    Trigger autonomous insight generation using the Zettelkasten engine.

    This tool activates the autonomous synthesis engine to identify patterns
    and generate novel insights from the existing knowledge graph.

    Args:
        topic: Optional topic to focus insight generation on
        confidence_threshold: Minimum confidence level for insights (0.0-1.0)

    Returns:
        Generated insights with confidence scores and evidence trails
    """
    try:
        await ctx.info("Generating autonomous insights")

        synapse = ctx.request_context.lifespan_context["synapse"]
        synapse.set_context(ctx)

        insights = await synapse.insight_engine.generate_insights(
            topic=topic, confidence_threshold=confidence_threshold
        )

        if not insights:
            return "No new insights generated above the confidence threshold."

        result_buffer = ["🧠 **Autonomous Insights Generated**\n\n"]

        for i, insight in enumerate(insights, 1):
            result_buffer.append(
                f"**Insight {i}** (Confidence: {insight['confidence']:.2f})\n"
                f"{insight['content']}\n\n"
                f"*Evidence Trail:* {len(insight['evidence'])} supporting facts\n"
                f"*Pattern Type:* {insight['pattern_type']}\n"
                f"*Zettel ID:* {insight['zettel_id']}\n\n---\n\n"
            )

        await ctx.info(f"Generated {len(insights)} new insights")
        return "".join(result_buffer)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Insight generation failed: %s", e, exc_info=True)
        return f"❌ Insight generation failed [{type(e).__name__}]: {str(e)}"


# ----------------------------------------------------------------------
# Gap analysis (GBrain pattern — no LLM)
# ----------------------------------------------------------------------

def _extract_query_dimensions(query: str) -> list[str]:
    """Rough NLP-free dimension extraction from query tokens."""
    q = query.lower()
    dims: list[str] = []

    temporal_signals = [
        (r"\b(?:when|what year|what month|what day|how long|timeframe|period|duration)\b", "temporal"),
        (r"\b(?:since|before|after|between|until|ago|recent|latest|current)\b", "temporal"),
        (r"\b(?:202[0-9]|19[0-9]{2})\b", "temporal"),
        (r"\b(?:yesterday|last week|last month|this week|this month|today)\b", "temporal"),
    ]
    for pattern, label in temporal_signals:
        if re.search(pattern, q):
            dims.append(label)
            break

    quantity_signals = [
        (r"\b(?:how many|how much|how often|percentage|percent|ratio|rate|count|number|amount)\b", "quantitative"),
        (r"\b(?:mrr|arr|revenue|users?|customers?|dau|mau|growth|cost|price|value|metric|kpi)\b", "quantitative"),
    ]
    for pattern, label in quantity_signals:
        if re.search(pattern, q):
            dims.append(label)
            break

    people_signals = [
        (r"\b(?:who|whom|people|team|members?|employees?|founder|ceo|cto|cofounder)\b", "people"),
        (r"\b(?:company|organization|org|team|startup|firm)\b", "organization"),
    ]
    for pattern, label in people_signals:
        if re.search(pattern, q):
            dims.append(label)
            break

    if re.search(r"\b(?:why|cause|because|reason|result|effect|impact|due to|leads to)\b", q):
        dims.append("causal")

    if re.search(r"\b(?:how relate|connected|relationship|between|link|compared|versus|vs)\b", q):
        dims.append("relational")

    # Temporal: override only if explicit time signal (not just "last" alone)
    # Don't flag "last quarter" as temporal unless it's paired with actual date signal
    has_explicit_time = bool(
        re.search(r"\b(?:202[0-9]|19[0-9]{2}|yesterday|last week|last month|this week|this month|today)\b", q)
    )
    if "temporal" in dims and not has_explicit_time:
        dims.remove("temporal")

    return dims


def _analyze_gaps(
    query: str, facts: list[dict[str, Any]], insights: list[dict[str, Any]]
) -> list[str]:
    """Detect knowledge gaps the query asks about but the KB doesn't satisfy."""
    import re as _re

    if not query.strip():
        return []

    dims = _extract_query_dimensions(query)
    if not dims:
        return []

    gaps: list[str] = []
    # Keep originals for person-name detection; lower for keyword checks
    fact_texts_lower = {f.get("statement", "").lower() for f in facts}
    insight_texts_lower = {i.get("content", "").lower() for i in insights}
    fact_texts_raw = {f.get("statement", "") for f in facts}
    insight_texts_raw = {i.get("content", "") for i in insights}
    all_text_lower = " ".join(fact_texts_lower | insight_texts_lower)
    all_text_raw = " ".join(fact_texts_raw | insight_texts_raw)

    for dim in dims:
        if dim == "temporal":
            if not _re.search(r"\b(202[0-9]|19[0-9]{2}|[A-Z][a-z]+ \d{1,2},? \d{4})\b", all_text_lower):
                gaps.append(
                    "No temporal anchors (dates/periods) found for this query. "
                    "Consider adding a timeline entry or `date:` field to relevant entity pages."
                )
        elif dim == "quantitative":
            if not _re.search(r"\b\d+(?:\.\d+)?[%$]?\b", all_text_lower):
                gaps.append(
                    "No quantitative data found. If you have metrics, try adding "
                    "a `gbrain-facts` block with metric: value unit: period: fields."
                )
        elif dim == "people":
            # Title-case two-word sequences (first and last names) in raw text
            if not _re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", all_text_raw):
                gaps.append(
                    "No person entities found. If the query involves people, "
                    "consider enriching relevant entity pages."
                )
        elif dim == "causal":
            if not _re.search(r"\b(because|caused|leads to|result|therefore|due to|reason)\b", all_text_lower):
                gaps.append(
                    "No causal links found. Try using synapse_remember with "
                    "predicate: caused to record causal chains."
                )
        elif dim == "relational":
            if not _re.search(r"\b(connects?|relates?|linked|associated|related)\b", all_text_lower):
                gaps.append(
                    "No relationship data found. Try explore_connections to map "
                    "the relationship graph for the relevant entities."
                )

    return gaps


@mcp.tool()
async def query_knowledge(
    ctx: Context, query: str, include_insights: bool = True, max_results: int = 10
) -> str:
    """
    Query the knowledge graph for facts and insights using natural language.

    This tool provides the conversational interface to the knowledge base,
    prioritizing synthesized insights over raw facts.

    Args:
        query: Natural language query
        include_insights: Whether to include AI-generated insights
        max_results: Maximum number of results to return

    Returns:
        Query results with facts, insights, and reasoning trails
    """
    try:
        await ctx.info(f"Processing knowledge query: {query[:50]}...")

        synapse = ctx.request_context.lifespan_context["synapse"]
        synapse.set_context(ctx)

        # --- Stage 1: Entity extraction → graph seed ---
        # Extract named entities from query and pull directly connected facts.
        # These are high-precision hits that bypass vector similarity entirely.
        entity_facts: list[dict[str, Any]] = []
        if synapse.knowledge_graph:
            query_entities = await synapse.knowledge_graph.extract_query_entities(query)
            if query_entities:
                await ctx.info(f"Query entities found: {', '.join(query_entities)}")
                entity_facts = await synapse.knowledge_graph.query_by_entities(
                    query_entities, depth=1
                )

        # --- Stage 2: Hybrid RRF retrieval (vector ANN + BM25) ---
        facts = await synapse.knowledge_graph.query_hybrid(
            query, max_results=max_results
        )

        # --- Stage 3: Wikilink graph expansion ---
        # Find which wiki pages matched the search, then surface their wikilink
        # neighbours as conceptually adjacent context.
        wiki_context: list[str] = []
        if synapse.wiki_adapter:
            wiki_hits = await synapse.wiki_adapter.search_pages(query, subdir="wiki")
            if wiki_hits:
                hit_slugs = [h["name"] for h in wiki_hits[:3]]
                neighbours = await synapse.wiki_adapter.get_wikilink_neighbors(
                    hit_slugs
                )
                linked: list[str] = []
                for slug, links_list in neighbours.items():
                    if links_list:
                        linked_str = ", ".join(f"[[{link}]]" for link in links_list[:4])
                        linked.append(f"[[{slug}]] → {linked_str}")
                wiki_context = linked

        # --- Stage 4: Insights (Zettelkasten-first) ---
        insights: list[dict[str, Any]] = []
        if include_insights:
            insights = await synapse.insight_engine.search_insights(
                query, max_results=max_results // 2
            )

        # --- Merge entity hits to front of fact list ---
        entity_statements = {e["statement"] for e in entity_facts}
        merged_facts = list(entity_facts) + [
            f for f in facts if f["statement"] not in entity_statements
        ]

        # --- Format output ---
        result_buffer = ["🔍 **Knowledge Query Results**\n\n"]

        if insights:
            result_buffer.append("**💡 Relevant Insights:**\n\n")
            for insight in insights:
                result_buffer.append(
                    f"- **{insight['title']}** (Confidence: {insight['confidence']:.2f})\n"
                    f"  {insight['content']}\n"
                    f"  *Evidence:* {len(insight['evidence'])} supporting facts\n\n"
                )

        if merged_facts:
            result_buffer.append("**📊 Factual Information:**\n\n")
            for fact in merged_facts[:max_results]:
                path = fact.get("retrieval_path", "hybrid")
                score = fact.get("rrf_score") or fact.get("similarity", 0)
                tag = f" `[{path}]`" if path == "entity_graph" else ""
                result_buffer.append(
                    f"- {fact['statement']}{tag}\n"
                    f"  *Source:* {fact['source']} | *Score:* {score:.4f}\n\n"
                )

        if wiki_context:
            result_buffer.append("**🔗 Related Wiki Pages:**\n\n")
            for link_line in wiki_context:
                result_buffer.append(f"- {link_line}\n")
            result_buffer.append("\n")

        # --- Stage 5: Gap analysis (GBrain pattern — no LLM) ---
        # Detect what factual dimensions the query asks about but we have
        # no/weak evidence for.  Pattern from GBrain synthesis layer.
        gaps = _analyze_gaps(query, merged_facts, insights)
        if gaps:
            result_buffer.append("**⚠️ Gaps (what we don't know yet):**\n\n")
            for gap in gaps:
                result_buffer.append(f"- {gap}\n")
            result_buffer.append("\n")

        if not insights and not merged_facts:
            result_buffer.append("No relevant information found in the knowledge base.")

        return "".join(result_buffer)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Knowledge query failed: %s", e, exc_info=True)
        return f"❌ Knowledge query failed [{type(e).__name__}]: {str(e)}"


@mcp.tool()
async def explore_connections(
    ctx: Context, entity: str, depth: int = 2, connection_types: list[str] | None = None
) -> str:
    """
    Explore connections and relationships around a specific entity in the knowledge graph.

    This tool implements the graph traversal capabilities for discovering
    non-obvious connections and patterns.

    Args:
        entity: Entity name to explore from
        depth: How many relationship hops to explore (1-5)
        connection_types: Specific relationship types to follow

    Returns:
        Visual representation of connections and discovered patterns
    """
    try:
        await ctx.info(f"Exploring connections for entity: {entity}")

        synapse = ctx.request_context.lifespan_context["synapse"]
        synapse.set_context(ctx)

        connections = await synapse.knowledge_graph.explore_entity_connections(
            entity=entity,
            depth=min(depth, 5),  # Limit depth for performance
            connection_types=connection_types,
        )

        if not connections:
            return f"No connections found for entity: {entity}"

        result_buffer = [f"🕸️ **Connection Map for '{entity}'**\n\n"]

        # Group by depth level
        by_depth: dict[int, list[dict[str, Any]]] = {}
        for conn in connections:
            level = conn["depth"]
            if level not in by_depth:
                by_depth[level] = []
            by_depth[level].append(conn)

        for level in sorted(by_depth.keys()):
            result_buffer.append(f"**Level {level} Connections:**\n")
            for conn in by_depth[level]:
                result_buffer.append(
                    f"  • {conn['target_entity']} ({conn['relationship_type']})\n"
                )
                if conn.get("strength"):
                    result_buffer.append(f"    Strength: {conn['strength']:.2f}\n")
            result_buffer.append("\n")

        # Highlight unexpected connections
        unexpected = [c for c in connections if c.get("unexpected", False)]
        if unexpected:
            result_buffer.append("🔍 **Unexpected Connections Discovered:**\n")
            for conn in unexpected:
                result_buffer.append(
                    f"  • {entity} → {conn['target_entity']} via {conn['path']}\n"
                )

        return "".join(result_buffer)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Connection exploration failed: %s", e, exc_info=True)
        return f"❌ Connection exploration failed [{type(e).__name__}]: {str(e)}"


@mcp.tool()
async def analyze_semantic_structure(
    ctx: Context, text: str, include_logical_form: bool = False
) -> str:
    """
    Analyze the semantic structure of text using Montague Grammar parsing.

    This tool provides insight into the formal semantic analysis capabilities
    and shows the logical form translations.

    Args:
        text: Text to analyze semantically
        include_logical_form: Whether to include the formal logical representation

    Returns:
        Semantic analysis with entities, relations, and optional logical forms
    """
    try:
        await ctx.info("Performing semantic structure analysis")

        synapse = ctx.request_context.lifespan_context["synapse"]
        synapse.set_context(ctx)

        analysis = await synapse.montague_parser.parse_text(text)

        result_buffer = [
            "🧮 **Semantic Structure Analysis**\n\n",
            f"**Input Text:** {text}\n\n",
        ]

        if analysis.get("entities"):
            result_buffer.append("**Entities Identified:**\n")
            for entity in analysis["entities"]:
                result_buffer.append(
                    f"  • {entity['text']} ({entity['type']}) - "
                    f"Confidence: {entity['confidence']:.2f}\n"
                )
            result_buffer.append("\n")

        if analysis.get("relations"):
            result_buffer.append("**Relations Identified:**\n")
            for relation in analysis["relations"]:
                result_buffer.append(
                    f"  • {relation['subject']} {relation['predicate']} "
                    f"{relation['object']}\n"
                )
            result_buffer.append("\n")

        if include_logical_form and analysis.get("logical_form"):
            result_buffer.append("**Logical Form (Montague Grammar):**\n")
            result_buffer.append(f"```\n{analysis['logical_form']}\n```\n\n")

        if analysis.get("semantic_features"):
            result_buffer.append("**Semantic Features:**\n")
            for feature, value in analysis["semantic_features"].items():
                result_buffer.append(f"  • {feature}: {value}\n")

        return "".join(result_buffer)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Semantic analysis failed: %s", e)
        return f"Error during semantic analysis: {str(e)}"


# =============================================================================
# MCP TOOLS - Wiki (LLM-WIKI) Bridge
# =============================================================================


@mcp.tool()
async def wiki_list_pages(ctx: Context, subdir: str = "wiki") -> str:
    """
    List all markdown pages in the wiki vault.

    Args:
        subdir: Subdirectory to list ('wiki' or 'raw').
    """
    try:
        synapse = ctx.request_context.lifespan_context["synapse"]
        synapse.set_context(ctx)
        if not synapse.wiki_adapter:
            return "Wiki adapter not configured. Set WIKI_VAULT_PATH."
        pages = await synapse.wiki_adapter.list_pages(subdir)
        if not pages:
            return f"No pages found in {subdir}/"
        lines = [f"📂 **{subdir}/** — {len(pages)} pages\n"]
        for pg in pages:
            lines.append(f"- `{pg['path']}` — {pg.get('summary', pg['name'])}")
        return "\n".join(lines)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("wiki_list_pages failed: %s", e)
        return f"Error: {e}"


@mcp.tool()
async def wiki_read_page(ctx: Context, path: str) -> str:
    """
    Read a wiki page by relative path (e.g. 'wiki/concepts/rag.md').

    Args:
        path: Relative path from vault root.
    """
    try:
        synapse = ctx.request_context.lifespan_context["synapse"]
        synapse.set_context(ctx)
        if not synapse.wiki_adapter:
            return "Wiki adapter not configured."
        data = await synapse.wiki_adapter.read_page(path)
        meta: dict[str, Any] = data.get("metadata", {})
        body: str = data.get("body", "")
        header = "\n".join(f"  {k}: {v}" for k, v in meta.items())
        return f"**Metadata:**\n{header}\n\n**Content:**\n{body}"
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("wiki_read_page failed: %s", e)
        return f"❌ Error: {e}"


@mcp.tool()
async def wiki_write_page(
    ctx: Context,
    path: str,
    body: str,
    summary: str = "",
    tags: str = "",
) -> str:
    """
    Write or update a wiki page with frontmatter.

    Args:
        path: Relative path (e.g. 'wiki/entities/neo4j.md').
        body: Markdown body content.
        summary: One-line summary for the index.
        tags: Comma-separated tags.
    """
    try:
        synapse = ctx.request_context.lifespan_context["synapse"]
        synapse.set_context(ctx)
        if not synapse.wiki_adapter:
            return "Wiki adapter not configured."
        meta: dict[str, Any] = {}
        if summary:
            meta["summary"] = summary
        if tags:
            meta["tags"] = [t.strip() for t in tags.split(",")]
        result = await synapse.wiki_adapter.write_page(path, body, meta)
        await synapse.wiki_adapter.append_log("write", f"Updated page: {path}")
        return str(result)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("wiki_write_page failed: %s", e)
        return f"Error: {e}"


@mcp.tool()
async def wiki_search(ctx: Context, query: str) -> str:
    """Search wiki pages by keyword.

    Args:
        query: Space-separated search terms.
    """
    try:
        synapse = ctx.request_context.lifespan_context["synapse"]
        synapse.set_context(ctx)
        if not synapse.wiki_adapter:
            return "Wiki adapter not configured."
        results = await synapse.wiki_adapter.search_pages(query)
        if not results:
            return f"No pages matched: {query}"
        lines = [f"🔍 {len(results)} results for '{query}'\n"]
        for r in results:
            lines.append(f"- `{r['path']}`")
        return "\n".join(lines)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Wiki search failed: %s", e, exc_info=True)
        return f"❌ Wiki search failed [{type(e).__name__}]: {str(e)}"


@mcp.tool()
async def wiki_lint(ctx: Context) -> str:
    """Run a health check on the wiki vault.

    Detects orphan pages, broken wikilinks, and missing frontmatter.
    """
    try:
        synapse = ctx.request_context.lifespan_context["synapse"]
        synapse.set_context(ctx)
        if not synapse.wiki_adapter:
            return "Wiki adapter not configured."
        report = await synapse.wiki_adapter.lint()
        lines = [f"🩺 **Wiki Health Check** — {report['total_pages']} pages\n"]
        if report["orphan_pages"]:
            lines.append(f"**Orphans** ({len(report['orphan_pages'])}):")
            for o in report["orphan_pages"]:
                lines.append(f"  - {o}")
        if report["broken_links"]:
            lines.append(f"**Broken links** ({len(report['broken_links'])}):")
            for bl in report["broken_links"]:
                lines.append(f"  - {bl['source']} → [[{bl['target']}]]")
        if report["missing_frontmatter"]:
            lines.append(
                f"**Missing frontmatter** ({len(report['missing_frontmatter'])}):"
            )
            for mf in report["missing_frontmatter"]:
                lines.append(f"  - {mf}")
        if report.get("non_reciprocal_links"):
            lines.append(
                f"**Non-reciprocal links** ({len(report['non_reciprocal_links'])}) "
                f"— A links to B but B doesn't link back:"
            )
            for nr in report["non_reciprocal_links"]:
                lines.append(
                    f"  - [[{nr['source']}]] → [[{nr['missing_back_link']}]] "
                    "(no return link)"
                )
        if report.get("non_preferred_tags"):
            lines.append(
                f"**Non-preferred tags** ({len(report['non_preferred_tags'])}) "
                f"— use controlled vocabulary:"
            )
            for np_ in report["non_preferred_tags"]:
                lines.append(
                    f"  - {np_['page']}: `{np_['tag']}` → use `{np_['use_instead']}`"
                )
        if not any(
            [
                report["orphan_pages"],
                report["broken_links"],
                report["missing_frontmatter"],
                report.get("non_reciprocal_links"),
                report.get("non_preferred_tags"),
            ]
        ):
            lines.append("✅ All clear — no issues found.")
        await synapse.wiki_adapter.append_log("lint", "\n".join(lines))
        return "\n".join(lines)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Wiki lint failed: %s", e, exc_info=True)
        return f"❌ Wiki lint failed [{type(e).__name__}]: {str(e)}"


@mcp.tool()
async def wiki_hits_analysis(ctx: Context) -> str:
    """Compute HITS hub and authority scores on the wiki wikilink graph.

    Authorities = pages cited by many others — load-bearing knowledge nodes.
    Hubs = pages that link to many good authorities — navigation layers.

    Use to identify which pages need deepening (high authority) and which
    need comprehensive link coverage (high hub).
    """
    try:
        synapse = ctx.request_context.lifespan_context["synapse"]
        synapse.set_context(ctx)
        if not synapse.wiki_adapter:
            return "Wiki adapter not configured."
        scores = await synapse.wiki_adapter.compute_wikilink_hits()
        if not scores:
            return "Not enough pages to compute HITS scores."

        # Sort by authority desc for the top authorities table
        by_auth = sorted(scores.items(), key=lambda x: x[1]["authority"], reverse=True)
        by_hub = sorted(scores.items(), key=lambda x: x[1]["hub"], reverse=True)

        lines = ["📊 **HITS Analysis — Wiki Wikilink Graph**\n"]
        lines.append(
            "**Top Authorities** (pages that should have the richest content):\n"
        )
        for slug, s in by_auth[:8]:
            lines.append(f"  {s['authority']:.4f}  [[{slug}]]")
        lines.append("\n**Top Hubs** (pages that should have comprehensive links):\n")
        for slug, s in by_hub[:8]:
            lines.append(f"  {s['hub']:.4f}  [[{slug}]]")
        return "\n".join(lines)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Wiki HITS analysis failed: %s", e, exc_info=True)
        return f"❌ Wiki HITS analysis failed [{type(e).__name__}]: {str(e)}"


@mcp.tool()
async def wiki_cluster_pages(ctx: Context, n_clusters: int | None = None) -> str:
    """Cluster wiki pages by semantic similarity using GAAC (TF-IDF).

    Identifies:
    - Natural topic clusters — pages that belong together
    - Missing links — same-cluster pages with no wikilink between them
    - Merge candidates — pages so similar they may be redundant (sim > 0.7)

    Args:
        n_clusters: Number of clusters (auto = sqrt of page count if omitted).
    """
    try:
        synapse = ctx.request_context.lifespan_context["synapse"]
        synapse.set_context(ctx)
        if not synapse.wiki_adapter:
            return "Wiki adapter not configured."
        result = await synapse.wiki_adapter.cluster_wiki_pages(n_clusters)

        lines = ["🗂️ **Wiki Page Clusters (GAAC)**\n"]
        for cluster in result["clusters"]:
            cluster_pages = ", ".join(f"[[{p}]]" for p in cluster["pages"])
            lines.append(f"**Cluster {cluster['id']}:** {cluster_pages}")
            if cluster["missing_links"]:
                for a, b in cluster["missing_links"]:
                    lines.append(f"  ⚠️  Missing link: [[{a}]] ↔ [[{b}]]")
        if result["merge_candidates"]:
            lines.append("\n**Merge candidates** (similarity > 0.7):\n")
            for a, b, sim in result["merge_candidates"]:
                lines.append(f"  {sim:.3f}  [[{a}]] ↔ [[{b}]]")
        else:
            lines.append("\n✅ No high-similarity merge candidates found.")
        return "\n".join(lines)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Wiki clustering failed: %s", e, exc_info=True)
        return f"❌ Wiki clustering failed [{type(e).__name__}]: {str(e)}"


@mcp.tool()
async def wiki_update_index(ctx: Context, deep: bool = False) -> str:
    """Rebuild the wiki index from all wiki pages.

    Args:
        deep: If True, performs a disk-level verification of all indexed files.
    """
    try:
        synapse = ctx.request_context.lifespan_context["synapse"]
        synapse.set_context(ctx)
        if not synapse.wiki_adapter:
            return "Wiki adapter not configured."
        result = await synapse.wiki_adapter.update_index(deep=deep)
        await synapse.wiki_adapter.append_log("index", result)
        return str(result)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Wiki index update failed: %s", e, exc_info=True)
        return f"❌ Wiki index update failed [{type(e).__name__}]: {str(e)}"


@mcp.tool()
async def wiki_ingest_raw(
    ctx: Context,
    filename: str,
) -> str:
    """
    Read a raw source file and ingest it into both the knowledge graph and wiki.

    Reads from raw/, runs it through the Synapse semantic pipeline,
    stores in Neo4j, and creates a summary page in wiki/sources/.

    Args:
        filename: Filename inside the raw/ directory.
    """
    try:
        synapse = ctx.request_context.lifespan_context["synapse"]
        synapse.set_context(ctx)
        if not synapse.wiki_adapter:
            return "Wiki adapter not configured."

        # Read raw file
        raw_data = await synapse.wiki_adapter.read_page(f"raw/{filename}")
        if "error" in raw_data:
            return str(raw_data["error"])
        body = raw_data.get("body", "")
        if not body.strip():
            return f"Empty file: raw/{filename}"

        # Ingest into knowledge graph
        kg_result = None
        if synapse.semantic_integrator and synapse.knowledge_graph:
            processed = await synapse.semantic_integrator.process_text_with_semantics(
                body, f"raw/{filename}", raw_data.get("metadata", {})
            )
            kg_result = await synapse.knowledge_graph.store_processed_data(processed)

        # Log
        summary = body[:200].replace("\n", " ") + "..."
        await synapse.wiki_adapter.append_log(
            f"ingest | {filename}",
            f"Ingested raw/{filename} into knowledge graph.\n\nPreview: {summary}",
        )

        parts = [f"✅ Ingested `raw/{filename}`"]
        if kg_result:
            parts.append(
                f"  Graph: {kg_result['new_nodes']} nodes, "
                f"{kg_result['new_edges']} edges added"
            )

        # Auto-move to typed Clippings/ archive
        source_url = raw_data.get("metadata", {}).get("source", "")
        move_result = await synapse.wiki_adapter.move_to_clippings(filename, source_url)
        parts.append(f"  📁 {move_result}")

        parts.append(
            "\nNext: Use `wiki_write_page` to create a summary page in "
            "`wiki/sources/` and update relevant entity/concept pages."
        )
        return "\n".join(parts)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Wiki ingest raw failed: %s", e, exc_info=True)
        return f"❌ Wiki ingest raw failed: {e}"


def _find_node() -> str | None:
    """Find the node executable, checking nvm directories if not on PATH."""
    # Try system PATH first
    node = shutil.which("node") or shutil.which("nodejs")
    if node:
        return node
    # Search nvm versions (pick the most recent)
    nvm_dir = Path(os.path.expanduser("~/.nvm/versions/node"))
    if nvm_dir.exists():
        versions = sorted(nvm_dir.iterdir(), reverse=True)
        for v in versions:
            candidate = v / "bin" / "node"
            if candidate.exists():
                return str(candidate)
    return None


def _find_defuddle() -> str | None:
    """Find the defuddle executable, checking nvm bin directories."""
    defuddle = shutil.which("defuddle")
    if defuddle:
        return defuddle
    nvm_dir = Path(os.path.expanduser("~/.nvm/versions/node"))
    if nvm_dir.exists():
        versions = sorted(nvm_dir.iterdir(), reverse=True)
        for v in versions:
            candidate = v / "bin" / "defuddle"
            if candidate.exists():
                return str(candidate)
    return None


@mcp.tool()
async def wiki_fetch_url(
    ctx: Context,
    url: str,
    ingest: bool = True,
) -> str:
    """
    Fetch a URL with defuddle (clean markdown extraction), save to raw/,
    ingest into the knowledge graph, and archive to Clippings/.

    Use this when researching the web — it strips navigation and clutter,
    leaving only the article content. Much cleaner than raw web_fetch.

    Args:
        url: The URL to fetch and process.
        ingest: If True (default), immediately ingest into Neo4j after saving.
                Set False to save to raw/ only for manual review first.
    """
    try:
        synapse = ctx.request_context.lifespan_context["synapse"]
        synapse.set_context(ctx)
        if not synapse.wiki_adapter:
            return "Wiki adapter not configured."

        # Locate defuddle
        defuddle_bin = _find_defuddle()
        if not defuddle_bin:
            node_bin = _find_node()
            if not node_bin:
                return (
                    "defuddle not found. Install with: npm install -g defuddle\n"
                    "(node also not found — install via nvm first)"
                )
            return (
                "defuddle not found. Install with:\n"
                f"  {node_bin.replace('/bin/node', '/bin/npm')} install -g defuddle"
            )

        # Run defuddle
        result = subprocess.run(  # nosec
            [defuddle_bin, "parse", url, "--md"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return f"defuddle error: {result.stderr.strip()}"

        content = result.stdout.strip()
        if not content:
            return f"defuddle returned empty content for: {url}"

        # Also try to grab title for the filename slug
        title_result = subprocess.run(  # nosec
            [defuddle_bin, "parse", url, "-p", "title"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        raw_title = title_result.stdout.strip() if title_result.returncode == 0 else ""

        # Build slug from title or URL
        if raw_title:
            slug = re.sub(r"[^\w\s-]", "", raw_title.lower())
            slug = re.sub(r"[\s_]+", "-", slug).strip("-")[:60]
        else:
            slug = re.sub(r"[^\w-]", "-", url.split("//")[-1])[:60].strip("-")

        filename = f"{slug}.md"

        # Build frontmatter + content
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        frontmatter = (
            f"---\n"
            f"title: {raw_title or slug}\n"
            f"source: {url}\n"
            f"created: {now}\n"
            f"tags: [clippings]\n"
            f"---\n\n"
        )
        full_content = frontmatter + content

        # Write to raw/
        raw_path = synapse.wiki_adapter.raw_dir / filename
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(full_content, encoding="utf-8")

        if not ingest:
            return (
                f"✅ Saved to raw/{filename}\n"
                f"   Source: {url}\n"
                f"   Run wiki_ingest_raw('{filename}') when ready to process."
            )

        # Ingest immediately (same pipeline as wiki_ingest_raw)
        raw_data = await synapse.wiki_adapter.read_page(f"raw/{filename}")
        body = raw_data.get("body", "")

        kg_result = None
        if synapse.semantic_integrator and synapse.knowledge_graph:
            processed = await synapse.semantic_integrator.process_text_with_semantics(
                body, f"raw/{filename}", raw_data.get("metadata", {})
            )
            kg_result = await synapse.knowledge_graph.store_processed_data(processed)

        summary = body[:200].replace("\n", " ") + "..."
        await synapse.wiki_adapter.append_log(
            f"fetch | {filename}",
            f"Fetched {url} via defuddle → ingested.\n\nPreview: {summary}",
        )

        move_result = await synapse.wiki_adapter.move_to_clippings(filename, url)

        parts = [f"✅ Fetched and ingested `{url}`"]
        parts.append(f"   Saved as: {filename}")
        if kg_result:
            parts.append(
                f"  Graph: {kg_result['new_nodes']} nodes, "
                f"{kg_result['new_edges']} edges added"
            )
        parts.append(f"  📁 {move_result}")
        parts.append(
            "\nNext: Use `wiki_write_page` to create a summary page in `wiki/sources/`."
        )
        return "\n".join(parts)

    except subprocess.TimeoutExpired:
        return f"Timeout fetching {url} — site may be slow or blocking."
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Wiki fetch URL failed: %s", e, exc_info=True)
        return f"❌ Wiki fetch URL failed [{type(e).__name__}]: {str(e)}"


# =============================================================================
# MCP RESOURCES - Knowledge Base Access
# =============================================================================


@mcp.resource("synapse://knowledge_stats")
async def knowledge_statistics() -> str:
    """
    Provides current statistics about the knowledge graph and synthesis engine.

    Returns real-time metrics about:
    - Total entities and relationships
    - Generated insights count
    - Processing statistics
    - System health metrics
    """
    try:
        synapse = synapse_server

        if not synapse.knowledge_graph:
            return "Knowledge graph not initialized"

        stats = await synapse.knowledge_graph.get_statistics()
        if synapse.insight_engine is not None:
            insight_stats = await synapse.insight_engine.get_statistics()
        else:
            insight_stats = {
                "total_insights": 0,
                "active_patterns": 0,
                "avg_confidence": 0.0,
                "last_processing": "N/A",
            }

        return f"""📊 **Project Synapse Knowledge Statistics**

**Knowledge Graph:**
- Total Entities: {stats['entity_count']:,}
- Total Relationships: {stats['relationship_count']:,}
- Total Facts: {stats['fact_count']:,}
- Graph Density: {stats['density']:.4f}

**Zettelkasten Engine:**
- Generated Insights: {insight_stats['total_insights']:,}
- Active Patterns: {insight_stats['active_patterns']:,}
- Avg Confidence: {insight_stats['avg_confidence']:.3f}
- Last Processing: {insight_stats['last_processing']}

**Processing Metrics:**
- Texts Processed: {stats['texts_processed']:,}
- Processing Rate: {stats['processing_rate']:.2f} texts/hour
- Error Rate: {stats['error_rate']:.2%}

**System Health:**
- Database Status: {stats['db_status']}
- Memory Usage: {stats['memory_usage']:.1f}%
- Active Connections: {stats['active_connections']}
"""

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to get knowledge statistics: %s", e, exc_info=True)
        return f"❌ Failed to get knowledge statistics [{type(e).__name__}]: {str(e)}"


@mcp.resource("synapse://insights/{topic}")
async def topic_insights(topic: str) -> str:
    """
    Retrieves all insights related to a specific topic.

    Args:
        topic: The topic to retrieve insights for

    Returns:
        All insights related to the specified topic with evidence trails
    """
    try:
        synapse = synapse_server

        if not synapse.insight_engine:
            return "Insight engine not initialized"

        insights = await synapse.insight_engine.get_insights_by_topic(topic)

        if not insights:
            return f"No insights found for topic: {topic}"

        result = f"💡 **Insights for Topic: {topic}**\n\n"

        for insight in insights:
            result += f"""**{insight['title']}**
Confidence: {insight['confidence']:.3f} | Created: {insight['created_at']}

{insight['content']}

*Evidence Trail:*
"""
            for evidence in insight["evidence"][:3]:  # Show top 3 evidence items
                result += (
                    f"  • {evidence['statement']} (Source: {evidence['source']})\n"
                )

            if len(insight["evidence"]) > 3:
                result += (
                    f"  ... and {len(insight['evidence']) - 3} more evidence items\n"
                )

            result += f"\n*Pattern Type:* {insight['pattern_type']}\n"
            result += f"*Zettel ID:* {insight['zettel_id']}\n\n---\n\n"

        return result

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to retrieve topic insights: %s", e, exc_info=True)
        return f"❌ Failed to retrieve topic insights [{type(e).__name__}]: {str(e)}"


# =============================================================================
# TEMPORAL MEMORY TOOLS — episodic memory across sessions
# =============================================================================
#
# These tools give Claude (and the user) a place to put time-stamped facts.
# Different from the document-extraction graph in two ways:
#
#  - Every fact carries valid_from (when it became true in the world) and
#    observed_at (when we learned it). Either end can be null when unknown.
#  - Writes are first-class user actions, not extraction byproducts. The
#    intent is that Claude can say "remember X" or "what did Ty say about Y
#    last week" and have it actually work.
#
# Design note on date handling: dates come in as ISO 8601 strings from MCP
# clients (no native datetime in JSON). We parse permissively — bare dates
# like "2026-05-14" get treated as midnight UTC, full datetimes are passed
# through. Empty / None / "" → unknown.


def _parse_iso(s: str | None) -> datetime | None:
    """Permissive ISO 8601 parsing for MCP-tool date arguments.

    Accepts bare dates (2026-05-14), full datetimes with or without timezone,
    and treats empty / whitespace / None as 'unknown'. Bare dates become
    midnight UTC so they have a stable point on the timeline.
    """
    if not s or not s.strip():
        return None
    text = s.strip()
    # Bare YYYY-MM-DD → midnight UTC
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        text = text + "T00:00:00+00:00"
    # Python's fromisoformat handles 'Z' suffix from 3.11+
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(f"Could not parse ISO date '{s}': {e}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@mcp.tool()
async def synapse_remember(
    subject: str,
    predicate: str,
    object: str,  # noqa: A002  # pylint: disable=redefined-builtin
    valid_from: str | None = None,
    valid_to: str | None = None,
    confidence: float = 1.0,
    source: str = "agent:claude",
    note: str = "",
) -> str:
    """Record a time-stamped fact in Synapse's episodic memory.

    Use this whenever something is worth remembering across sessions:
    decisions, observations, "Ty said X on date Y", health/diet/symptom
    log entries, project milestones.

    Args:
        subject: Who or what the fact is about. Free-form name.
        predicate: The relationship verb. snake_case preferred
            (e.g. "started_taking", "moved_to", "decided_to_use").
        object: The other side of the relation.
        valid_from: ISO date or datetime when the fact became true. If
            omitted, "now" is used. Bare dates → midnight UTC.
        valid_to: ISO date or datetime when the fact stopped being true.
            Omit for still-current facts.
        confidence: 0–1. Default 1.0 for explicit user statements; lower
            when the agent is inferring.
        source: Where this fact came from. Defaults to "agent:claude" for
            things Claude is recording. Use "user" or a filename for facts
            from explicit user statements or document ingestion.
        note: Free-form context. Stored in metadata for later recall.

    Returns:
        The fact id (stable content hash — safe to call twice).
    """
    if synapse_server.temporal_facts is None:
        return "❌ Temporal-fact store not initialised."
    try:
        vf = _parse_iso(valid_from) or datetime.now(timezone.utc)
        vt = _parse_iso(valid_to)
        fact = TemporalFact(
            subject=subject,
            predicate=predicate,
            object=object,
            valid_from=vf,
            valid_to=vt,
            confidence=confidence,
            source=source,
            metadata={"note": note} if note else {},
        )
        fid = await synapse_server.temporal_facts.add(fact)
        return (
            f"✅ Remembered: ({subject}) -[{predicate}]-> ({object}) "
            f"as of {vf.isoformat()}\nfact_id={fid}"
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("synapse_remember failed: %s", e, exc_info=True)
        return f"❌ Failed to record fact [{type(e).__name__}]: {e}"


@mcp.tool()
async def synapse_recall(
    entity: str,
    as_of: str | None = None,
    direction: str = "both",
) -> str:
    """Look up time-stamped facts about an entity.

    Args:
        entity: Name to look up. Matches subject, object, or both depending
            on `direction`.
        as_of: ISO date/datetime — if given, only facts valid at this point
            in time are returned. Omit for "currently true" facts.
        direction: "outgoing" (entity is the subject), "incoming" (entity
            is the object), or "both" (default).

    Returns:
        Newline-separated list of facts with timestamps. Empty if none found.
    """
    if synapse_server.temporal_facts is None:
        return "❌ Temporal-fact store not initialised."
    try:
        ao = _parse_iso(as_of)
        rows = await synapse_server.temporal_facts.query_entity(
            entity, as_of=ao, direction=direction
        )
        if not rows:
            return f"No temporal facts found for '{entity}'."
        lines = [f"# Facts about {entity}" + (f" as of {ao.date()}" if ao else "")]
        for r in rows:
            vf = r["valid_from"]
            vt = r["valid_to"]
            when = f"{vf}" + (f" → {vt}" if vt else " (still true)")
            line = (
                f"- ({r['subject']}) -[{r['predicate']}]-> ({r['object']})\n"
                f"  {when}  source={r['source']}  conf={r['confidence']:.2f}"
            )
            if r.get("metadata"):
                line += f"\n  note={r['metadata']}"
            lines.append(line)
        return "\n".join(lines)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("synapse_recall failed: %s", e, exc_info=True)
        return f"❌ Failed to recall [{type(e).__name__}]: {e}"


@mcp.tool()
async def synapse_timeline(
    entity: str | None = None,
    limit: int = 50,
) -> str:
    """Chronological view of remembered facts.

    Args:
        entity: Scope to one entity, or None for the global timeline.
        limit: Max number of rows. Default 50.

    Returns:
        Time-ordered fact list, oldest first.
    """
    if synapse_server.temporal_facts is None:
        return "❌ Temporal-fact store not initialised."
    try:
        rows = await synapse_server.temporal_facts.timeline(entity, limit=limit)
        if not rows:
            return f"No timeline available{' for ' + entity if entity else ''}."
        scope = f" — {entity}" if entity else ""
        lines = [f"# Timeline{scope}"]
        for r in rows:
            vf = r["valid_from"]
            vt = r["valid_to"]
            arrow = f" → {vt}" if vt else ""
            lines.append(
                f"- {vf}{arrow}  ({r['subject']}) -[{r['predicate']}]-> "
                f"({r['object']})  src={r['source']}"
            )
        return "\n".join(lines)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("synapse_timeline failed: %s", e, exc_info=True)
        return f"❌ Failed to retrieve timeline [{type(e).__name__}]: {e}"


@mcp.tool()
async def synapse_invalidate(
    subject: str,
    predicate: str,
    object: str,  # noqa: A002  # pylint: disable=redefined-builtin
    ended: str | None = None,
) -> str:
    """Mark a previously-recorded fact as no longer true.

    Sets ``valid_to`` rather than deleting — the historical record stays
    intact, but the fact is no longer "currently true" for default queries.

    Args:
        subject/predicate/object: The triple to invalidate.
        ended: ISO date/datetime when the fact stopped being true.
            Defaults to now if omitted.

    Returns:
        Number of facts affected.
    """
    if synapse_server.temporal_facts is None:
        return "❌ Temporal-fact store not initialised."
    try:
        end_time = _parse_iso(ended) or datetime.now(timezone.utc)
        n = await synapse_server.temporal_facts.invalidate(
            subject, predicate, object, ended=end_time
        )
        if n == 0:
            return (
                f"No still-true facts matched ({subject})-[{predicate}]->"
                f"({object})."
            )
        return (
            f"✅ Invalidated {n} fact(s): ({subject})-[{predicate}]->"
            f"({object}) ended at {end_time.isoformat()}"
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("synapse_invalidate failed: %s", e, exc_info=True)
        return f"❌ Failed to invalidate [{type(e).__name__}]: {e}"


@mcp.tool()
async def synapse_causal_window(
    effect_entity: str,
    before: str,
    within_days: int = 30,
) -> str:
    """Find candidate causes by temporal correlation.

    Surfaces facts whose ``valid_from`` falls in the window
    ``[before - within_days, before]`` and that share at least one
    entity with facts about ``effect_entity``.

    This is exactly the "track everything you ate to find what caused
    the headaches" pattern — you record symptom onset, you record meals
    and medications, then this tool surfaces co-occurring events as
    candidates. The tool returns correlation; the human (or a downstream
    reasoning step) decides what caused what.

    Args:
        effect_entity: The thing whose causes you're hunting (e.g. "headache",
            "rash", "build failure").
        before: ISO date/datetime — when the effect was observed.
        within_days: How far back to search. Default 30.

    Returns:
        Ranked list of candidate cause-effect pairings with day deltas.
    """
    if synapse_server.temporal_facts is None:
        return "❌ Temporal-fact store not initialised."
    try:
        before_dt = _parse_iso(before)
        if before_dt is None:
            return "❌ `before` is required for causal-window search."
        rows = await synapse_server.temporal_facts.causal_chain(
            effect_entity, before=before_dt, within_days=within_days
        )
        if not rows:
            return (
                f"No co-occurring facts found for '{effect_entity}' within "
                f"{within_days} days before {before_dt.date()}."
            )
        lines = [
            f"# Candidate causes for {effect_entity} before "
            f"{before_dt.date()} (window={within_days}d)",
            "(Correlation only — these are candidates, not proven causes.)",
        ]
        for r in rows:
            lines.append(
                f"- {r['days_before']}d before: "
                f"({r['cause_subject']})-[{r['cause_predicate']}]->"
                f"({r['cause_object']})\n"
                f"  effect: ({r['effect_subject']})-[{r['effect_predicate']}]->"
                f"({r['effect_object']})  "
                f"shared_entities={r['shared_entities']}"
            )
        return "\n".join(lines)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("synapse_causal_window failed: %s", e, exc_info=True)
        return f"❌ Failed causal-window search [{type(e).__name__}]: {e}"


@mcp.tool()
async def synapse_memory_stats() -> str:
    """Quick stats: how many temporal facts are stored, time span covered."""
    if synapse_server.temporal_facts is None:
        return "❌ Temporal-fact store not initialised."
    try:
        s = await synapse_server.temporal_facts.stats()
        return (
            f"Temporal facts: {s['total']:,} total, "
            f"{s['still_true']:,} still true\n"
            f"Earliest: {s['earliest']}\n"
            f"Latest:   {s['latest']}"
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("synapse_memory_stats failed: %s", e, exc_info=True)
        return f"❌ Failed stats query [{type(e).__name__}]: {e}"


# =============================================================================
# MCP PROMPTS - AI Guidance Templates
# =============================================================================


@mcp.prompt()
def knowledge_synthesis_prompt(topic: str, context: str = "") -> str:
    """
    Generate a prompt for synthesizing knowledge about a specific topic.

    Args:
        topic: The topic to synthesize knowledge about
        context: Additional context or constraints

    Returns:
        Structured prompt for knowledge synthesis
    """
    return f"""You are Project Synapse, an autonomous knowledge synthesis engine.

Analyze the following topic using formal semantic reasoning and Zettelkasten methodology:

**Topic:** {topic}
{"**Context:** " + context if context else ""}

Please provide:

1. **Semantic Decomposition**: Break down the topic into its core semantic components
2. **Knowledge Connections**: Identify how this topic connects to existing knowledge
3. **Pattern Recognition**: Look for underlying patterns or structures
4. **Insight Generation**: Generate novel insights based on the analysis
5. **Evidence Requirements**: What evidence would strengthen these insights?

Focus on creating atomic, linkable insights that can be integrated into a broader knowledge network. Each insight should be:
- Self-contained and precisely articulated
- Backed by clear reasoning
- Connected to related concepts
- Assigned a confidence level

Format your response to facilitate both human understanding and automated processing."""


@mcp.prompt()
def semantic_analysis_prompt(text: str) -> list[base.Message]:
    """
    Generate a structured prompt for deep semantic analysis using Montague Grammar principles.

    Args:
        text: Text to analyze

    Returns:
        Multi-turn conversation for comprehensive semantic analysis
    """
    return [
        base.UserMessage(
            content=(
                f"Perform a comprehensive semantic analysis of the following "
                f'text using Montague Grammar principles:\n\n"{text}"\n\n'
                "Please analyze:\n"
                "1. Logical structure and compositional semantics\n"
                "2. Entity-relationship extraction\n"
                "3. Semantic ambiguities and resolutions\n"
                "4. Truth-conditional meaning\n"
                "5. Presuppositions and implications"
            )
        ),
        base.AssistantMessage(
            content=(
                "I'll analyze this text using formal semantic methods. "
                "Let me break down the logical structure and identify the "
                "key semantic components systematically."
            )
        ),
        base.UserMessage(
            content=(
                "Focus particularly on how the semantic components can be "
                "represented as atomic facts in a knowledge graph, and identify "
                "any patterns that might indicate deeper insights."
            )
        ),
    ]


@mcp.prompt()
def insight_validation_prompt(insight: str, evidence: str) -> str:
    """
    Generate a prompt for validating AI-generated insights against evidence.

    Args:
        insight: The insight to validate
        evidence: Supporting evidence

    Returns:
        Validation prompt for insight assessment
    """
    return f"""As Project Synapse's validation system, critically assess this AI-generated insight:

**Insight:** {insight}

**Supporting Evidence:**
{evidence}

Evaluation criteria:
1. **Logical Consistency**: Is the insight logically sound?
2. **Evidence Sufficiency**: Does the evidence adequately support the claim?
3. **Novelty Assessment**: Does this provide new understanding?
4. **Confidence Rating**: What confidence level (0.0-1.0) would you assign?
5. **Potential Biases**: What biases might affect this insight?
6. **Falsifiability**: How could this insight be tested or challenged?

Provide a structured assessment with a recommended confidence score and suggestions
for strengthening the insight if needed."""


# =============================================================================
# PROCESS CLEANUP & SERVER MANAGEMENT
# =============================================================================


def cleanup_processes() -> None:
    """Clean up all processes and background tasks on shutdown."""
    logger.info("Performing cleanup on server shutdown")

    # Run async cleanup
    try:
        asyncio.run(synapse_server.cleanup())
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error during cleanup: %s", e)


def signal_handler(signum: int, _frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    logger.info("Received signal %s, shutting down gracefully", signum)
    cleanup_processes()
    sys.exit(0)


# Register cleanup handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup_processes)


def main() -> None:
    """Main entry point for the MCP server."""
    try:
        logger.info("Starting Project Synapse MCP Server")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Server error: %s", e)
        raise
    finally:
        cleanup_processes()


if __name__ == "__main__":
    main()
