# Context: Phase 3 - Semantic Retrieval & Graph Cleaning

## Research Findings

### 1. Ingestion Noise Patterns
During the ingestion of the "Eidetic Learning" paper, the following noise patterns were identified:
- **HTML Spans**: `<span id="A6.T8.pic1.1.1...">` which are likely leftovers from PDF-to-Markdown conversion.
- **LaTeX Math**: Expressions like `$t_{1}$` are being treated as text but sometimes break the semantic parser's ability to identify entities properly.
- **CID Strings**: `(cid:123)` sequences occasionally appear in malformed PDFs.

### 2. Entity Type Misidentification
The `MontagueParser` relies on spaCy's `ORG` label for many things. Technical concepts like "Eidetic Learning" or "Backpropagation" are often tagged as `ORG` because they are proper-noun-like but don't fit `PERSON` or `GPE`. 
- **Solution**: We need to expand the schema to include `Concept` and `Method` and implement a "Technical Term" dictionary or heuristic (e.g., if it contains "Learning", "Algorithm", "Network" but not "Inc", "Corp").

### 3. RRF Optimization
Reciprocal Rank Fusion (RRF) is currently implemented with a simple $1/(k + rank)$ formula.
- **Standard k**: 60 is the industry standard for RRF.
- **Entity Boost**: Results coming from direct graph traversal (`query_by_entities`) should probably be treated as high-precision signals and given a boost or a fixed high rank in the fusion.

## Requirements Mapping
- **SEM-01**: Improved noise filtering (Regex-based).
- **SEM-02**: Entity type normalization (Heuristics).
- **SEM-03**: Optimized hybrid retrieval (RRF tuning).
