# Development Concerns

## Technical Debt
- **Mocking Strategy**: `MockKnowledgeGraph` needs to stay in sync with `KnowledgeGraph`. As graph schemas evolve, the mock might lag behind, leading to false positives in tests.
- **Error Granularity**: Some MCP tools have broad try/except blocks. More granular error handling for specific failure modes (e.g., Neo4j connection vs. query syntax) would improve resilience.
- **Wiki Syncing**: The current mechanism for syncing graph data with wiki files relies on content hashing but may not robustly handle manual file renames or deletions in the Obsidian vault.

## Performance Risks
- **Local Embeddings**: Generating embeddings on-the-fly via `sentence-transformers` can be CPU-intensive if a GPU is unavailable, potentially timing out MCP requests.
- **Graph Traversal Depth**: Unrestricted depth in `explore_connections` could lead to expensive Neo4j queries. Currently capped at 5, but complex graphs may still see latency.
- **Montague Grammar Parsing**: The formal semantic pipeline is significantly more complex than standard NER and may become a bottleneck for large document ingestion.

## Dependencies
- **Neo4j 2026.x**: The project targets a very specific and modern version of Neo4j. Compatibility with older versions or alternative graph DBs is not guaranteed.
- **defuddle**: Web fetching depends on a global installation of `defuddle` (Node.js). If Node.js environments change (e.g., nvm versions), this integration can break.
- **spaCy Models**: Dependency on `en_core_web_sm` requires a manual download step after installation, which can be easily missed in new environments.

## Future Considerations
- **Multi-Graph Support**: While the README mentions one Neo4j instance can serve multiple projects, the current `KnowledgeGraph` class implementation assumes a single target database/graph.
- **Scalability**: The autonomous synthesis engine (`InsightEngine`) runs as a background task. As the graph grows, the synthesis frequency and resource consumption will need to be throttled or optimized.
