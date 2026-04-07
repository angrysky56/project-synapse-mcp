# 🧠 Project Synapse MCP Server

**Autonomous Knowledge Synthesis Engine with LLM-WIKI Integration**

Project Synapse is an MCP (Model Context Protocol) server that combines a Neo4j 2026.x graph database with an Obsidian Markdown wiki to create a persistent, compounding knowledge base. Raw text is processed through a semantic pipeline into interconnected graph nodes with vector embeddings, while a human-readable wiki layer provides browsable, interlinked Markdown pages.

## Architecture

```
Raw Sources (Obsidian vault / Clippings)
         │
         ▼
┌──────────────────────┐     ┌─────────────────────┐
│  Semantic Pipeline   │────▶│  Neo4j Knowledge    │
│  (Montague Grammar,  │     │  Graph (entities,   │
│   NLP, embeddings)   │     │  facts, vectors)    │
└──────────────────────┘     └────────┬────────────┘
                                      │
                              ┌───────┴───────┐
                              │ Wiki Adapter  │
                              └───────┬───────┘
                                      │
                              ┌───────▼───────┐
                              │ Obsidian Vault│
                              │ (Markdown,    │
                              │  Git-synced)  │
                              └───────────────┘
```

## Key Features

### Knowledge Graph (Neo4j 2026.x)
- Native **VECTOR** type with ANN semantic search
- Fulltext BM25 indexes for keyword search
- Hybrid search (vector + BM25 score fusion)
- Graph traversal for discovering hidden relationships
- Montague Grammar parser for formal semantic analysis
- Zettelkasten engine for autonomous insight generation

### LLM-WIKI Integration
- Bridges Obsidian Markdown vault with the Neo4j graph
- Full page CRUD with YAML frontmatter
- Automatic index generation and append-only log
- Health checks: orphan detection, broken wikilinks, missing frontmatter
- Delta-sync manifest (content hashing) for efficient graph sync
- Based on [Andrej Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)

### Local-Only Embeddings (No Paid APIs)
- **sentence-transformers** (default) — runs on GPU
- **Ollama** (optional) — any local embedding model
- All vectors stored natively in Neo4j via `db.create.setNodeVectorProperty()`

## Quick Start

### Prerequisites
- Python 3.12+
- Neo4j 2026.x
- uv package manager
- Obsidian (for wiki browsing)

### Installation

```bash
cd /home/ty/Repositories/ai_workspace
git clone <repository-url> project-synapse-mcp
cd project-synapse-mcp
uv venv --python 3.12 --seed
source .venv/bin/activate
uv add -e .
uv run python -m spacy download en_core_web_sm
cp .env.example .env  # edit with your Neo4j password and vault path
```

### Configuration

Edit `.env`:
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Embedding — local only, no paid APIs
EMBEDDING_PROVIDER=sentence-transformers  # or "ollama"
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIMENSION=768

# Wiki vault
WIKI_VAULT_PATH=/path/to/your/obsidian-vault
WIKI_GITHUB_REPO=https://github.com/user/wiki-repo
```

### Claude Desktop / MCP Integration

Add to your MCP config:
```json
{
  "mcpServers": {
    "project-synapse": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/project-synapse-mcp",
        "run", "python", "-m", "synapse_mcp.server"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your_password",
        "NEO4J_DATABASE": "neo4j",
        "WIKI_VAULT_PATH": "/path/to/obsidian-vault",
        "WIKI_GITHUB_REPO": "https://github.com/user/wiki-repo"
      }
    }
  }
}
```

## MCP Tools

### Knowledge Graph
| Tool | Description |
|---|---|
| `ingest_text` | Process text through semantic pipeline → Neo4j |
| `query_knowledge` | Vector semantic search with insight-first results |
| `explore_connections` | Graph traversal for hidden relationships |
| `generate_insights` | Autonomous Zettelkasten pattern detection |
| `analyze_semantic_structure` | Montague Grammar semantic analysis |

### Wiki (LLM-WIKI)
| Tool | Description |
|---|---|
| `wiki_ingest_raw` | Read raw source → Neo4j + wiki summary page |
| `wiki_write_page` | Create/update wiki page with frontmatter |
| `wiki_read_page` | Read a wiki page by path |
| `wiki_search` | Keyword search across wiki pages |
| `wiki_list_pages` | List all pages in a subdirectory |
| `wiki_update_index` | Rebuild the wiki index |
| `wiki_lint` | Health check: orphans, broken links, missing frontmatter |

## Wiki Vault Structure

```
LLM-WIKI/
├── CLAUDE.md           # Schema doc — wiki conventions and workflows
├── raw/                # Immutable source documents
├── wiki/
│   ├── index.md        # Auto-generated page catalogue
│   ├── log.md          # Append-only activity log
│   ├── entities/       # People, tools, projects
│   ├── concepts/       # Ideas, theories, patterns
│   └── sources/        # Summaries of ingested raw documents
└── Clippings/          # Obsidian Web Clipper output
```

### Workflow

1. **Ingest**: Drop source into `raw/`, call `wiki_ingest_raw` → Neo4j + wiki pages
2. **Query**: `query_knowledge` (graph) or `wiki_search` (files) → synthesize answer
3. **Lint**: `wiki_lint` → fix orphans, broken links, stale claims
4. **Rollback**: Git handles version control via Obsidian Git plugin

## Theoretical Foundation

- **Montague Grammar**: Formal compositional semantics for meaning extraction
- **Zettelkasten Method**: Atomic linked notes with emergent structure
- **Graph Theory**: Community detection, centrality, path analysis
- **Karpathy LLM-WIKI**: Persistent knowledge compilation vs stateless RAG
- **Vannevar Bush's Memex**: Private associative knowledge with maintained trails

## License

MIT — see [LICENSE](LICENSE).

---

*Project Synapse: From reactive RAG to persistent, compounding knowledge.*
