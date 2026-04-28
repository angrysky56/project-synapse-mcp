# Testing Guide

Project Synapse uses `pytest` for its test suite. Given the complex nature of the system (Neo4j, Local Embeddings, Filesystem Wiki), testing is divided into unit and integration tests.

## 1. Running Tests

To run the entire test suite:

```bash
uv run pytest
```

To run a specific test file:

```bash
uv run pytest tests/test_health.py
```

## 2. Test Categories

### 2.1. Health & Connection Tests (`test_health.py`, `test_setup.py`)
These verify that the environment is correctly configured and that the server can connect to both Neo4j and the Obsidian vault.

### 2.2. Knowledge Graph Storage (`test_kg_storage.py`)
Verifies that entities, facts, and relationships are correctly merged into Neo4j and that vector embeddings are generated and stored without error.

### 2.3. Wiki Resilience (`test_wiki_resilience.py`)
Ensures the `WikiAdapter` can handle edge cases like missing directories, read/write permissions, and complex Markdown frontmatter parsing.

### 2.4. Unit Tests (`tests/unit/`)
Isolated tests for logic-heavy components like the `MontagueParser` and `TextProcessor` that do not require an active database.

## 3. Mocking & Dependencies

We aim to minimize external dependencies in tests. However, since Synapse relies heavily on Neo4j 2026.x features (Vector Search), an active Neo4j instance is often required for integration tests.

> [!TIP]
> Use a dedicated "test" database in Neo4j to avoid clobbering your production knowledge graph. You can specify this via the `NEO4J_DATABASE` environment variable when running tests.

```bash
NEO4J_DATABASE=synapse_test uv run pytest
```

## 4. Writing New Tests

1.  **Use Async**: Most core logic is asynchronous. Use the `@pytest.mark.asyncio` decorator.
2.  **Clean up**: Ensure any files created in the wiki vault during tests are cleaned up (use `tmp_path` fixtures where possible).
3.  **Assertions**: Use descriptive assertions to make debugging easier.

**Example:**

```python
import pytest
from synapse_mcp.wiki.wiki_adapter import WikiAdapter

@pytest.mark.asyncio
async def test_wiki_write_read(tmp_path):
    # Use a temporary directory for the vault
    adapter = WikiAdapter(vault_path=str(tmp_path))
    await adapter.initialize()
    
    path = "wiki/test_page.md"
    await adapter.write_page(path, "Hello World", {"summary": "test"})
    
    data = await adapter.read_page(path)
    assert data["body"] == "Hello World"
    assert data["metadata"]["summary"] == "test"
```
