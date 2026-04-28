# Testing Strategy

## Framework
- **Pytest**: Primary test runner.
- **pytest-asyncio**: Required for testing async code.

## Test Categories

### Unit Tests
- Location: `tests/test_basic.py`, etc.
- Focus: Individual component logic (e.g., `MontagueParser` parsing, `TextProcessor` extraction).
- Mocks: Use `MockKnowledgeGraph` for testing logic without a live Neo4j instance.

### Integration Tests
- Location: `tests/test_kg_storage.py`.
- Focus: Verified interaction with live dependencies (Neo4j, DuckDB).
- Requirement: Requires a running Neo4j instance (typically configured via `.env`).

### System / E2E Tests
- Location: `tests/debug_test.py`.
- Focus: Verification of the MCP server tools and overall synthesis pipeline.

## Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_kg_storage.py

# Run with output
uv run pytest -s
```

## Quality Gates
- **Static Analysis**: `uv run mypy src` must pass before commits.
- **Linting**: `uv run ruff check .` must pass.
- **Formatting**: `uv run ruff format .` should be run regularly.
