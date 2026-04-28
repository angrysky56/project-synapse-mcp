# Coding Conventions

## Core Principles
- **Async-First**: All I/O operations (DB, File, Network) must be asynchronous.
- **Strict Typing**: Mypy is configured for strict checking. All function signatures must have type hints.
- **MCP Standards**: Tools must use the `@mcp.tool()` decorator and provide clear docstrings for discovery.

## Documentation
- **Docstrings**: Required for all public classes and methods. Follow Google or Sphinx style.
- **Markdown**: All project-level documentation and wiki content uses GitHub Flavored Markdown.

## Error Handling
- Use structured logging (`logger.error`, `logger.info`).
- In MCP tools, wrap logic in try/except blocks to return user-friendly error strings instead of crashing the server.
- Log full tracebacks to stderr for developer debugging.

## Formatting & Linting
- **Ruff**: Primary linter and formatter.
- **Line Length**: 88 characters (consistent with Black).
- **Import Sorting**: Handled by Ruff (`I` selector).

## Patterns
- **Lifespan Context**: Use `@asynccontextmanager` for server setup and teardown.
- **Dependency Injection**: Pass initialized components (like `KnowledgeGraph`) to engines rather than creating them globally.
- **Pydantic Models**: Use for data validation and tool argument parsing where appropriate.
