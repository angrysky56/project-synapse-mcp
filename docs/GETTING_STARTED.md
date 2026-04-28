# Getting Started with Project Synapse

Follow this guide to set up Project Synapse on your local machine and connect it to your favorite AI agent.

## Prerequisites

- **Python 3.10+**: Ensure you have a modern Python version installed.
- **uv**: We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.
- **Neo4j 2026.x**: A running Neo4j instance. You can use Neo4j Desktop or a Docker container.
- **Obsidian**: (Optional but recommended) To view and interact with your knowledge vault.

## 1. Installation

Clone the repository and install the dependencies using `uv`:

```bash
git clone https://github.com/angrysky56/project-synapse-mcp.git
cd project-synapse-mcp

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
uv pip install -e .
```

## 2. Configuration

Set up your environment variables by creating a `.env` file in the project root:

```bash
# Neo4j Connectivity
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Path to your Obsidian vault
WIKI_VAULT_PATH=/path/to/your/obsidian/vault
```

For a full list of configuration options, see the [Configuration Guide](CONFIGURATION.md).

## 3. First Run (Server Mode)

Start the Synapse MCP server. By default, it communicates over `stdio`:

```bash
uv run synapse-mcp
```

## 4. Connecting to an AI Agent

### Claude Desktop
To use Synapse with Claude Desktop, add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "synapse": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/project-synapse-mcp",
        "run",
        "synapse-mcp"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your_password",
        "WIKI_VAULT_PATH": "/path/to/your/vault"
      }
    }
  }
}
```

## 5. Verification

Once connected, you can verify the setup by asking your agent to list the available tools or to ingest a simple fact:

> "Ingest the fact that Project Synapse is an autonomous knowledge engine."

The agent should respond with a summary of the ingestion process, and you should see a new fact appear in your Neo4j browser and a corresponding source page in your Obsidian vault.

## Next Steps

- Explore the [Architecture](ARCHITECTURE.md) to understand how the pipeline works.
- Learn about [Testing](TESTING.md) to ensure your environment is stable.
- Check the [Development Guide](DEVELOPMENT.md) if you want to contribute.
