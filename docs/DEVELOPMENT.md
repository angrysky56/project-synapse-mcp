# Development Guide

This guide is for developers who want to contribute to Project Synapse, add new tools, or modify the core semantic pipeline.

## 1. Development Environment

Synapse uses [uv](https://github.com/astral-sh/uv) for dependency management and virtual environments.

```bash
# Initialize development environment
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## 2. Core Architecture for Developers

The server logic is encapsulated in the `SynapseServer` class (located in `src/synapse_mcp/server.py`). It manages the lifecycle of several key components:

- **`KnowledgeGraph`**: Interfaces with Neo4j.
- **`MontagueParser`**: Handles formal semantic parsing.
- **`SemanticIntegrator`**: The high-level pipeline for ingestion.
- **`InsightEngine`**: Autonomous pattern recognition and Zettel generation.
- **`WikiAdapter`**: Manages the filesystem-based Obsidian vault.

## 3. Adding a New MCP Tool

Synapse uses the `FastMCP` framework. To add a new tool:

1.  Open `src/synapse_mcp/server.py`.
2.  Use the `@mcp.tool()` decorator on an `async` function.
3.  Access the server state via `ctx.request_context.lifespan_context["synapse"]`.

**Example:**

```python
@mcp.tool()
async def my_new_tool(ctx: Context, arg1: str) -> str:
    """Description of the tool for the LLM."""
    synapse = ctx.request_context.lifespan_context["synapse"]
    synapse.set_context(ctx) # Propagate context for logging
    
    # Use internal components
    stats = await synapse.knowledge_graph.get_statistics()
    return f"Result based on {arg1} and {stats['entity_count']} entities."
```

## 4. Code Quality & Standards

We maintain high code quality standards using the following tools:

### 4.1. Linting and Formatting
We use `ruff` for linting and `black` for formatting.

```bash
# Run linter
uv run ruff check .

# Fix linting issues automatically
uv run ruff check . --fix

# Format code
uv run black .
```

### 4.2. Type Checking
We use `mypy` for static type analysis. All functions must have type hints.

```bash
uv run mypy src
```

## 5. Working with Neo4j

When modifying the graph schema, update the `_initialize_schema` method in `src/synapse_mcp/core/knowledge_graph.py`. We follow a **MERGE-first** approach to ensure idempotency during ingestion.

## 6. Logs and Debugging

Synapse logs to `stderr` to avoid interfering with MCP's `stdout` communication channel. You can adjust the verbosity via the `LOG_LEVEL` environment variable.

```bash
LOG_LEVEL=DEBUG uv run synapse-mcp
```
