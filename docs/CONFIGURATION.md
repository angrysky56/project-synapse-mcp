# Configuration Guide

Project Synapse is configured primarily via environment variables. This guide outlines the available settings for the Neo4j knowledge graph, the Wiki adapter, and the semantic processing pipeline.

## 1. Core Database (Neo4j)

Synapse requires a Neo4j 2026.x instance with the **Vector Search** and **Fulltext** features enabled.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `NEO4J_URI` | The bolt/neo4j connection string. | `bolt://localhost:7687` |
| `NEO4J_USER` | Username for authentication. | `neo4j` |
| `NEO4J_PASSWORD` | Password for authentication. | `synapse_password` |
| `NEO4J_DATABASE` | The name of the database to use. | `neo4j` |

## 2. Knowledge Vault (Obsidian/Wiki)

Synapse acts as a bridge to a physical Markdown vault, following the "LLM-WIKI" architectural pattern.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `WIKI_VAULT_PATH` | **(Required)** Absolute path to your Obsidian vault. | *None* |
| `WIKI_GITHUB_REPO` | Optional URL to the vault's GitHub repository for sync. | *None* |

## 3. Semantic Processing & Embeddings

Synapse uses local embeddings to ensure privacy and low latency. It supports two primary providers.

### 3.1. General Settings
| Variable | Description | Default |
| :--- | :--- | :--- |
| `EMBEDDING_PROVIDER` | Choice of `sentence-transformers` or `ollama`. | `sentence-transformers` |
| `EMBEDDING_DIMENSION` | The dimension of the vector embedding. | `768` |

### 3.2. Sentence-Transformers (Default)
Runs locally using Python's `sentence-transformers` library (uses GPU if available).

| Variable | Description | Default |
| :--- | :--- | :--- |
| `EMBEDDING_MODEL` | The specific model identifier from HuggingFace. | `sentence-transformers/all-mpnet-base-v2` |

### 3.3. Ollama (Alternative)
Requires a running Ollama instance on your local machine.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | The URL of the Ollama API server. | `http://localhost:11434` |
| `OLLAMA_EMBED_MODEL` | The model name in Ollama used for embeddings. | `mxbai-embed-large` |

## 4. Logging and Debugging

| Variable | Description | Default |
| :--- | :--- | :--- |
| `LOG_LEVEL` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). | `INFO` |

## 5. Development Example (`.env`)

Create a `.env` file in your project root to manage these settings locally:

```bash
# Neo4j Config
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password
NEO4J_DATABASE=synapse

# Wiki Config
WIKI_VAULT_PATH=/home/user/Documents/MyKnowledgeVault

# Embedding Config (Ollama Example)
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBED_MODEL=mxbai-embed-large
EMBEDDING_DIMENSION=1024
```

> [!NOTE]
> If using `sentence-transformers` (the default), the system will download the model weights (approx. 400MB) on first run if they are not already present in your HuggingFace cache.
