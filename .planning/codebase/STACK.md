# Tech Stack

## Core Language
- **Python**: >=3.10
- **Type Hinting**: Strict (`disallow_untyped_defs = true` in mypy)

## Project Management
- **uv**: Package management and environment isolation
- **hatchling**: Build backend
- **ruff**: Linting and formatting
- **black**: Formatting (legacy support, ruff preferred)
- **mypy**: Static type checking

## Knowledge & Data
- **Neo4j**: Primary knowledge graph database (v6.1.0+)
- **DuckDB**: Analytics and local structured data (v1.3.2+)
- **NetworkX**: Graph analysis and algorithms
- **python-louvain**: Community detection in graphs

## NLP & Semantics
- **spaCy**: Entity extraction and linguistic features
- **NLTK**: Text processing utilities
- **Sentence-Transformers**: Vector embeddings for hybrid search
- **Montague Grammar**: Formal semantic parsing (custom implementation)

## Infrastructure & Communication
- **MCP (Model Context Protocol)**: CLI and server communication
- **FastMCP**: High-level framework for MCP server implementation
- **aiohttp**: Async HTTP client for web fetching
- **asyncio-mqtt**: MQTT support (likely for event-driven integration)
- **aiofiles**: Async file I/O

## Environment & Utilities
- **python-dotenv**: Environment variable management
- **pydantic**: Data validation and settings
- **jinja2**: Templating for wiki/docs
- **icecream**: Developer debugging
