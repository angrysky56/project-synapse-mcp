"""
Project Synapse MCP Server

Main server implementation providing autonomous knowledge synthesis capabilities
through the Model Context Protocol (MCP).
"""

import asyncio
import atexit
import signal
import sys
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts import base

# Import our core modules
from .core.knowledge_graph import KnowledgeGraph
from .data_pipeline.semantic_integrator import SemanticIntegrator
from .data_pipeline.text_processor import TextProcessor
from .semantic.montague_parser import MontagueParser
from .utils.logging_config import setup_logging
from .wiki.wiki_adapter import WikiAdapter
from .zettelkasten.insight_engine import InsightEngine

# Load environment variables
load_dotenv()

# Configure logging for MCP (stderr only)
logger = setup_logging(__name__)


class SynapseServer:
    """Main Project Synapse server class managing all components."""

    def __init__(self) -> None:
        self.knowledge_graph: KnowledgeGraph | None = None
        self.montague_parser: MontagueParser | None = None
        self.insight_engine: InsightEngine | None = None
        self.text_processor: TextProcessor | None = None
        self.semantic_integrator: SemanticIntegrator | None = None
        self.wiki_adapter: WikiAdapter | None = None
        self.background_tasks: set = set()

    async def initialize(self) -> None:
        """Initialize all server components."""
        logger.info("Initializing Project Synapse server components")

        try:
            # Initialize knowledge graph
            self.knowledge_graph = KnowledgeGraph()
            await self.knowledge_graph.connect()

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

            logger.info("All components initialized successfully")

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to initialize server components: %s", e)
            raise

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
async def lifespan_context(_: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Manage server lifecycle with proper cleanup."""
    logger.info("Starting Project Synapse MCP server")

    try:
        # Initialize server
        await synapse_server.initialize()

        # Start background insight generation if insight_engine is initialized
        if synapse_server.insight_engine is not None:
            insight_task = asyncio.create_task(
                synapse_server.insight_engine.start_autonomous_processing()
            )
            synapse_server.background_tasks.add(insight_task)

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
async def debug_test(_: Context) -> str:
    """Simple test tool to check if MCP server is working."""
    try:
        logger.info("Debug test tool called")
        return "✅ MCP server is working correctly!"
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Debug test failed: %s", e)
        return f"❌ Debug test failed: {str(e)}"


@mcp.tool()
async def ingest_text(
    ctx: Context,
    text: str,
    source: str = "user_input",
    metadata: dict[str, Any] | None = None
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
        await ctx.info("Ingesting text from source: %s", source)
        logger.info("Starting text ingestion: %s...", text[:50])

        synapse = ctx.request_context.lifespan_context["synapse"]
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
        logger.error("Text ingestion failed: %s", e)
        return f"Error during text ingestion: {str(e)}"


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

        await ctx.info("Generated %d new insights", len(insights))
        return "".join(result_buffer)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Insight generation failed: %s", e)
        return f"Error during insight generation: {str(e)}"


@mcp.tool()
async def query_knowledge(
    ctx: Context,
    query: str,
    include_insights: bool = True,
    max_results: int = 10
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
        await ctx.info("Processing knowledge query: %s...", query[:50])

        synapse = ctx.request_context.lifespan_context["synapse"]

        # First, check for relevant insights (Zettelkasten-first approach)
        insights: list[dict[str, Any]] = []
        if include_insights:
            insights = await synapse.insight_engine.search_insights(
                query, max_results=max_results // 2
            )

        # Then query for factual information
        facts = await synapse.knowledge_graph.query_semantic(
            query, max_results=max_results
        )

        result_buffer = ["🔍 **Knowledge Query Results**\n\n"]

        if insights:
            result_buffer.append("**💡 Relevant Insights:**\n\n")
            for insight in insights:
                result_buffer.append(
                    f"- **{insight['title']}** (Confidence: {insight['confidence']:.2f})\n"
                    f"  {insight['content']}\n"
                    f"  *Evidence:* {len(insight['evidence'])} supporting facts\n\n"
                )

        if facts:
            result_buffer.append("**📊 Factual Information:**\n\n")
            for fact in facts:
                result_buffer.append(
                    f"- {fact['statement']}\n"
                    f"  *Source:* {fact['source']} | *Confidence:* {fact['confidence']:.2f}\n\n"
                )

        if not insights and not facts:
            result_buffer.append("No relevant information found in the knowledge base.")

        return "".join(result_buffer)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Knowledge query failed: %s", e)
        return f"Error during knowledge query: {str(e)}"


@mcp.tool()
async def explore_connections(
    ctx: Context,
    entity: str,
    depth: int = 2,
    connection_types: list[str] | None = None
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
        await ctx.info("Exploring connections for entity: %s", entity)

        synapse = ctx.request_context.lifespan_context["synapse"]

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
                result_buffer.append(f"  • {conn['target_entity']} ({conn['relationship_type']})\n")
                if conn.get("strength"):
                    result_buffer.append(f"    Strength: {conn['strength']:.2f}\n")
            result_buffer.append("\n")

        # Highlight unexpected connections
        unexpected = [c for c in connections if c.get("unexpected", False)]
        if unexpected:
            result_buffer.append("🔍 **Unexpected Connections Discovered:**\n")
            for conn in unexpected:
                result_buffer.append(f"  • {entity} → {conn['target_entity']} via {conn['path']}\n")

        return "".join(result_buffer)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Connection exploration failed: %s", e)
        return f"Error exploring connections: {str(e)}"


@mcp.tool()
async def analyze_semantic_structure(
    ctx: Context,
    text: str,
    include_logical_form: bool = False
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

        analysis = await synapse.montague_parser.parse_text(text)

        result_buffer = [
            "🧮 **Semantic Structure Analysis**\n\n",
            f"**Input Text:** {text}\n\n"
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
        if not synapse.wiki_adapter:
            return "Wiki adapter not configured."
        data = await synapse.wiki_adapter.read_page(path)
        if "error" in data:
            return str(data["error"])
        meta: dict[str, Any] = data.get("metadata", {})
        body: str = data.get("body", "")
        header = "\n".join(f"  {k}: {v}" for k, v in meta.items())
        return f"**Metadata:**\n{header}\n\n**Content:**\n{body}"
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("wiki_read_page failed: %s", e)
        return f"Error: {e}"


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
        logger.error("wiki_search failed: %s", e)
        return f"Error: {e}"


@mcp.tool()
async def wiki_lint(ctx: Context) -> str:
    """Run a health check on the wiki vault.

    Detects orphan pages, broken wikilinks, and missing frontmatter.
    """
    try:
        synapse = ctx.request_context.lifespan_context["synapse"]
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
        if not any(
            [
                report["orphan_pages"],
                report["broken_links"],
                report["missing_frontmatter"],
            ]
        ):
            lines.append("✅ All clear — no issues found.")
        await synapse.wiki_adapter.append_log("lint", "\n".join(lines))
        return "\n".join(lines)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("wiki_lint failed: %s", e)
        return f"Error: {e}"


@mcp.tool()
async def wiki_update_index(ctx: Context) -> str:
    """Rebuild the wiki index from all wiki pages."""
    try:
        synapse = ctx.request_context.lifespan_context["synapse"]
        if not synapse.wiki_adapter:
            return "Wiki adapter not configured."
        result = await synapse.wiki_adapter.update_index()
        await synapse.wiki_adapter.append_log("index", result)
        return str(result)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("wiki_update_index failed: %s", e)
        return f"Error: {e}"


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
        parts.append(
            "\nNext: Use `wiki_write_page` to create a summary page in "
            "`wiki/sources/` and update relevant entity/concept pages."
        )
        return "\n".join(parts)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("wiki_ingest_raw failed: %s", e)
        return f"Error: {e}"


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
        logger.error("Failed to get knowledge statistics: %s", e)
        return f"Error retrieving statistics: {str(e)}"


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

        return str(result)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to retrieve topic insights: %s", e)
        return f"Error retrieving insights for topic '{topic}': {str(e)}"


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
                f"text using Montague Grammar principles:\n\n\"{text}\"\n\n"
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

Provide a structured assessment with a recommended confidence score and suggestions for strengthening the insight if needed."""


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


def signal_handler(signum: int, _: Any) -> None:
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
