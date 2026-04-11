# 🧠 Project Synapse MCP Server

**Autonomous Knowledge Synthesis Engine with LLM-WIKI Integration**

Project Synapse is an MCP (Model Context Protocol) server that combines a Neo4j 2026.x graph database with an Obsidian Markdown wiki to create a persistent, compounding knowledge base. Raw text is processed through a semantic pipeline into interconnected graph nodes with vector embeddings, while a human-readable wiki layer provides browsable, interlinked Markdown pages.

## What This Is (and Isn't)

**This is a knowledge system, not a code editor.** It's for the thinking, research, and writing that surrounds projects — architecture decisions, domain research, design rationale, reference material, meeting notes.

Code lives in its repo. Knowledge *about* the code lives here.

**Use cases:**
- Research deep-dives that accumulate over weeks/months
- Project knowledge bases (why decisions were made, not just what)
- Personal knowledge management (articles, books, podcast notes)
- Collaborative brainstorming with AI as the wiki maintainer

**Per-project setup:** Create a separate Obsidian vault + GitHub repo for each project. Point the `WIKI_VAULT_PATH` env var at it. One Neo4j instance can serve multiple projects (graphs coexist).

## Architecture

```
Web / Raw Sources
         │
    [defuddle]          ← cleans web content before ingestion
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

### Web Content Ingestion (defuddle)
- `wiki_fetch_url` fetches any URL, strips navigation/ads/clutter via defuddle, ingests into Neo4j, and archives to `Clippings/` — one call, fully automated
- `wiki_ingest_raw` auto-moves processed files from `raw/` to `Clippings/` — inbox stays clean
- `raw/` is a true inbox: empty after every session

### Local-Only Embeddings (No Paid APIs)
- **sentence-transformers** (default) — runs on GPU
- **Ollama** (optional) — any local embedding model
- All vectors stored natively in Neo4j via `db.create.setNodeVectorProperty()`

## Quick Start

### Prerequisites

- Python 3.12+
- [Neo4j 2026.x](https://neo4j.com/deployment-center/) (Community or Enterprise)
- uv package manager (`pip install uv`)
- [Obsidian](https://obsidian.md/) with the [Git community plugin](https://publish.obsidian.md/git-doc/Getting+Started)
- A GitHub repo for the wiki vault (can be private)
- Node.js + defuddle (for web content fetching — see below)

### Neo4j Setup
```bash
# Ubuntu/Debian — see neo4j.com for other platforms
sudo apt install neo4j
sudo systemctl start neo4j
sudo systemctl enable neo4j
# Set password (default user: neo4j)
sudo neo4j-admin set-initial-password your_password
```

### defuddle Setup

defuddle extracts clean markdown from web pages, stripping navigation, ads, and boilerplate before ingestion. Required for `wiki_fetch_url`.

```bash
# Install Node.js if not present (via nvm recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install --lts
nvm use --lts

# Install defuddle globally
npm install -g defuddle

# Verify
defuddle --version
```

> **Note:** Synapse finds defuddle automatically via nvm paths even if it's not on your shell's PATH. If `wiki_fetch_url` reports defuddle not found, ensure it's installed in an nvm-managed Node version.

### Obsidian Vault Setup
1. Create a new vault in Obsidian (or clone your wiki repo)
2. Install the **Git** community plugin (Settings → Community Plugins → Browse → "Git")
3. Configure Git plugin with your GitHub credentials
4. The vault structure (`raw/`, `wiki/`, `Clippings/`, `AGENTS.md`) is created automatically by Synapse on first run

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
| `generate_insights` | Autonomous Zettelkatten pattern detection |
| `analyze_semantic_structure` | Montague Grammar semantic analysis |

### Wiki (LLM-WIKI)
| Tool | Description |
|---|---|
| `wiki_fetch_url` | Fetch URL → defuddle clean → ingest → archive to Clippings/ |
| `wiki_ingest_raw` | Ingest file from raw/ → Neo4j + auto-move to Clippings/ |
| `wiki_write_page` | Create/update wiki page with frontmatter |
| `wiki_read_page` | Read a wiki page by path |
| `wiki_search` | Keyword search across wiki pages |
| `wiki_list_pages` | List all pages in a subdirectory |
| `wiki_update_index` | Rebuild the wiki index |
| `wiki_lint` | Health check: orphans, broken links, missing frontmatter |

## Wiki Vault Structure

```
LLM-WIKI/
├── AGENTS.md           # Agent schema doc — conventions and workflows
├── raw/                # INBOX ONLY — unprocessed files; empty after each session
├── raw-inbox.base      # Obsidian Base view of pending raw/ queue
├── Clippings/          # Permanent archive — all processed sources land here
├── wiki/
│   ├── index.md        # Auto-generated page catalogue
│   ├── log.md          # Append-only activity log
│   ├── entities/       # People, tools, projects
│   ├── concepts/       # Ideas, theories, patterns
│   └── sources/        # Summaries of ingested sources
```

### Content Lifecycle

```
You clip/save → raw/          # your inbox
     or
Agent fetches → wiki_fetch_url # web research
                    │
              [defuddle clean]
                    │
              [semantic pipeline] → Neo4j
                    │
              wiki_write_page → wiki/sources/
                    │
              auto-move → Clippings/   # permanent archive
```

`raw/` is always empty after a session. `Clippings/` is the permanent record of everything that's been processed. Source pages in `wiki/sources/` reference the original URL, not the file path.

### Workflow

1. **Web research**: `wiki_fetch_url(url)` → fetches, cleans, ingests, archives in one call
2. **Manual clip**: Drop into `raw/`, call `wiki_ingest_raw(filename)` → auto-archives after ingest
3. **Query**: `query_knowledge` (graph) or `wiki_search` (files) → synthesize answer
4. **Lint**: `wiki_lint` → fix orphans, broken links, stale claims
5. **Rollback**: Git handles version control via Obsidian Git plugin

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
