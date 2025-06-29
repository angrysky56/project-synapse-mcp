# 🧠 Project Synapse MCP Server

**Autonomous Knowledge Synthesis and Inference Engine**

Project Synapse is a revolutionary MCP (Model Context Protocol) server that transforms raw text into interconnected knowledge graphs and autonomously generates insights through advanced pattern detection. It combines formal semantic analysis (Montague Grammar) with Zettelkasten methodology to create a true cognitive partnership with AI.

## 🌟 Key Features

### 🔬 **Semantic Blueprint (Montague Grammar)**
- Formal semantic analysis for precise meaning extraction
- Compositional semantics with lambda calculus
- Logical form generation from natural language
- Ambiguity resolution through rule-based frameworks

### 🕸️ **Knowledge Cortex (Neo4j Graph Database)**
- Interconnected storage of entities, relationships, and facts
- High-performance graph traversal and pattern detection
- Scalable architecture supporting complex queries
- Provenance tracking for all knowledge elements

### 🧮 **Autonomous Zettelkasten Engine**
- Pattern detection using graph algorithms and ML
- Autonomous insight generation with confidence scoring
- Auditable reasoning trails for all hypotheses
- Continuous learning and knowledge synthesis

### 🔄 **MCP Integration**
- Full MCP protocol compliance for LLM integration
- Rich tool set for knowledge manipulation
- Real-time resources for knowledge statistics
- Guided prompts for semantic analysis workflows

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Neo4j Database
- uv package manager (recommended)

### Installation

1. **Clone and setup project:**
```bash
cd /home/ty/Repositories/ai_workspace
git clone <repository-url> project-synapse-mcp
cd project-synapse-mcp

# Create virtual environment
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install dependencies
uv add -e .
```

2. **Setup Neo4j Database:**
```bash
# Install Neo4j (Ubuntu/Debian)
sudo apt update
sudo apt install neo4j

# Start Neo4j service
sudo systemctl start neo4j
sudo systemctl enable neo4j

# Set password (default user: neo4j)
sudo neo4j-admin set-initial-password synapse_password
```

3. **Download spaCy model:**
```bash
uv run python -m spacy download en_core_web_sm
```

4. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Claude Desktop Integration

Add the following to your Claude Desktop configuration file (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "project-synapse": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/ty/Repositories/ai_workspace/project-synapse-mcp",
        "run",
        "python",
        "-m",
        "synapse_mcp.server"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j", 
        "NEO4J_PASSWORD": "synapse_password",
        "ANTHROPIC_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

## 🛠️ Core Tools

### `ingest_text`
Process and analyze text using the full semantic pipeline:
```
Ingest raw text → Montague Grammar parsing → Entity extraction → 
Knowledge graph storage → Automatic insight generation
```

### `generate_insights`
Trigger autonomous insight generation:
- Pattern detection using graph algorithms
- Community detection and centrality analysis
- Semantic clustering and path analysis
- Confidence-scored hypothesis generation

### `query_knowledge`
Natural language querying with insight-first responses:
- Prioritizes synthesized insights over raw facts
- Provides complete reasoning trails
- Supports complex semantic queries

### `explore_connections`
Graph traversal for discovering hidden relationships:
- Multi-hop connection exploration
- Unexpected pathway identification
- Relationship strength analysis

### `analyze_semantic_structure`
Deep semantic analysis using Montague Grammar:
- Logical form generation
- Entity-relationship extraction
- Truth-conditional semantics
- Compositional meaning analysis

## 📊 Resources

### `synapse://knowledge_stats`
Real-time knowledge graph statistics:
- Entity and relationship counts
- Insight generation metrics
- Processing performance data
- System health indicators

### `synapse://insights/{topic}`
Topic-specific insight retrieval:
- All insights related to a topic
- Evidence trails and confidence scores
- Pattern type classification
- Chronological insight development

## 🎯 Prompts

### Knowledge Synthesis Prompt
Structured prompt for comprehensive topic analysis using formal semantic reasoning and Zettelkasten methodology.

### Semantic Analysis Prompt
Multi-turn conversation template for deep Montague Grammar-based semantic analysis.

### Insight Validation Prompt
Systematic validation of AI-generated insights against evidence and logical consistency.

## 🧭 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Text Input    │───▶│  Montague Parser │───▶│ Knowledge Graph │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   MCP Client    │◀───│  Insight Engine  │◀────────────┘
│   (Claude AI)   │    │  (Zettelkasten)  │
└─────────────────┘    └──────────────────┘
```

### Components

1. **Semantic Blueprint**: Montague Grammar parser for formal meaning analysis
2. **Knowledge Cortex**: Neo4j graph database for interconnected storage
3. **Zettelkasten Engine**: Autonomous pattern detection and insight synthesis
4. **MCP Interface**: Protocol-compliant integration with LLM applications

## 🔧 Configuration

### Environment Variables
See `.env.example` for complete configuration options:

- **Database**: Neo4j connection settings
- **AI Models**: API keys for various providers  
- **Processing**: Batch sizes and thresholds
- **Insight Generation**: Confidence levels and intervals

### Performance Tuning
- Adjust `SEMANTIC_BATCH_SIZE` for processing throughput
- Configure `PATTERN_DETECTION_INTERVAL` for insight frequency
- Set `INSIGHT_CONFIDENCE_THRESHOLD` for quality control

## 🧪 Development

### Running Tests
```bash
uv run pytest tests/
```

### Development Server
```bash
# Run server directly for development
uv run python -m synapse_mcp.server

# Or use MCP development tools
mcp dev src/synapse_mcp/server.py
```

### Code Quality
```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Type checking
uv run mypy src/
```

## 📚 Theoretical Foundation

### Montague Grammar
- Formal compositional semantics
- Lambda calculus for meaning representation
- Model-theoretic truth conditions
- Systematic syntax-semantics correspondence

### Zettelkasten Method
- Atomic knowledge units with unique identifiers
- Explicit linking for knowledge networks
- Emergent structure through bottom-up organization
- Continuous expansion and connection building

### Graph Theory
- Community detection for knowledge clustering
- Centrality analysis for importance ranking
- Path analysis for connection discovery
- Network topology for insight generation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 and use type hints
- Write comprehensive docstrings
- Include tests for new functionality
- Update documentation for changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Montague Grammar foundational work by Richard Montague
- Zettelkasten methodology inspired by Niklas Luhmann
- MCP protocol by Anthropic for LLM integration
- Neo4j for graph database excellence

## 🔮 Roadmap

- [ ] Multi-modal processing (images, documents)
- [ ] Real-time collaborative knowledge building
- [ ] Advanced NLP beyond Montague Grammar
- [ ] Integration with external knowledge bases
- [ ] Mobile and web interfaces
- [ ] Enterprise security features

---

*Project Synapse: Transforming AI from reactive information retrieval to proactive cognitive partnership.*
