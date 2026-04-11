"""
Wiki Adapter for Project Synapse.

Bridges the Obsidian Markdown vault (LLM-WIKI) with the Neo4j knowledge graph.
Provides tools for:
  - Reading/writing wiki pages with frontmatter
  - Syncing wiki content to/from the graph
  - Linting the wiki for health checks
  - Listing and searching wiki pages via the index
"""

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles  # type: ignore

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Frontmatter helpers
# ---------------------------------------------------------------------------

def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Returns:
        Tuple of (metadata dict, body text without frontmatter).
    """
    if not content.startswith("---"):
        return {}, content

    end = content.find("---", 3)
    if end == -1:
        return {}, content

    raw = content[3:end].strip()
    meta: dict[str, Any] = {}
    for line in raw.split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            # Use a separate variable to avoid type shadowing/reassignment error
            cleaned_val: str | list[str] = val.strip().strip('"').strip("'")
            if (
                isinstance(cleaned_val, str)
                and cleaned_val.startswith("[")
                and cleaned_val.endswith("]")
            ):
                cleaned_val = [
                    v.strip().strip('"').strip("'")
                    for v in cleaned_val[1:-1].split(",")
                    if v.strip()
                ]
            meta[key] = cleaned_val
    return meta, content[end + 3:].strip()


def build_frontmatter(meta: dict[str, Any]) -> str:
    """Serialize a metadata dict into YAML frontmatter."""
    lines = ["---"]
    for k, v in meta.items():
        if isinstance(v, list):
            lines.append(f"{k}: [{', '.join(str(i) for i in v)}]")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WikiAdapter
# ---------------------------------------------------------------------------

class WikiAdapter:
    """Manages read/write access to an Obsidian vault for the LLM-WIKI pattern."""

    def __init__(self, vault_path: str | None = None, repo_url: str | None = None):
        self.vault_path = Path(str(vault_path or os.getenv("WIKI_VAULT_PATH", "")))
        self.repo_url = repo_url or os.getenv("WIKI_GITHUB_REPO", "")
        # Sub-directories following the Karpathy 3-layer architecture
        self.raw_dir = self.vault_path / "raw"
        self.wiki_dir = self.vault_path / "wiki"
        self.schema_path = self.vault_path / "CLAUDE.md"
        self.index_path = self.wiki_dir / "index.md"
        self.log_path = self.wiki_dir / "log.md"

    async def initialize(self) -> None:
        """Ensure vault directories exist."""
        if not self.vault_path or not self.vault_path.exists():
            logger.warning("Wiki vault path not found: %s", self.vault_path)
            return
        for d in [self.raw_dir, self.wiki_dir]:
            d.mkdir(parents=True, exist_ok=True)
        logger.info("Wiki adapter initialised – vault: %s", self.vault_path)

    # ------------------------------------------------------------------
    # Page CRUD
    # ------------------------------------------------------------------

    async def list_pages(self, subdir: str = "wiki") -> list[dict[str, Any]]:
        """List all .md pages in a vault subdirectory with frontmatter metadata."""
        target = self.vault_path / subdir
        if not target.exists():
            return []
        pages: list[dict[str, Any]] = []
        for p in sorted(target.rglob("*.md")):
            rel = p.relative_to(self.vault_path)
            async with aiofiles.open(p, encoding="utf-8") as f:
                content = await f.read()
            meta, _ = parse_frontmatter(content)
            pages.append({"path": str(rel), "name": p.stem, **meta})
        return pages

    async def read_page(self, rel_path: str) -> dict[str, Any]:
        """Read a wiki page and return metadata + body."""
        full = self.vault_path / rel_path
        if not full.exists():
            return {"error": f"Page not found: {rel_path}"}
        async with aiofiles.open(full, encoding="utf-8") as f:
            content = await f.read()
        meta, body = parse_frontmatter(content)
        return {"path": rel_path, "metadata": meta, "body": body}

    async def write_page(
        self,
        rel_path: str,
        body: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Write or update a wiki page with frontmatter."""
        full = self.vault_path / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        meta = metadata or {}
        meta.setdefault("updated", now)
        if not full.exists():
            meta.setdefault("created", now)

        content = build_frontmatter(meta) + "\n\n" + body.strip() + "\n"
        async with aiofiles.open(full, "w", encoding="utf-8") as f:
            await f.write(content)
        logger.info("Wrote wiki page: %s", rel_path)
        return f"Wrote {rel_path}"

    async def delete_page(self, rel_path: str) -> str:
        """Delete a wiki page."""
        full = self.vault_path / rel_path
        if full.exists():
            full.unlink()
            return f"Deleted {rel_path}"
        return f"Not found: {rel_path}"

    # ------------------------------------------------------------------
    # Search & Index
    # ------------------------------------------------------------------

    async def search_pages(self, query: str, subdir: str = "wiki") -> list[dict[str, Any]]:
        """Simple keyword search across wiki pages (case-insensitive)."""
        results: list[dict[str, Any]] = []
        target = self.vault_path / subdir
        if not target.exists():
            return results
        terms = query.lower().split()
        for p in target.rglob("*.md"):
            async with aiofiles.open(p, encoding="utf-8") as f:
                content = (await f.read()).lower()
            if all(t in content for t in terms):
                rel = str(p.relative_to(self.vault_path))
                results.append({"path": rel, "name": p.stem})
        return results

    async def update_index(self) -> str:
        """Rebuild wiki/index.md from all wiki pages."""
        pages = await self.list_pages("wiki")
        lines = [
            "---",
            f"updated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
            "type: index",
            "---",
            "",
            "# Wiki Index",
            "",
        ]
        for pg in pages:
            if pg["name"] == "index" or pg["name"] == "log":
                continue
            summary = pg.get("summary", "")
            lines.append(f"- [[{pg['name']}]] — {summary}")
        lines.append("")
        content = "\n".join(lines)
        async with aiofiles.open(self.index_path, "w", encoding="utf-8") as f:
            await f.write(content)
        return f"Index updated with {len(pages)} pages"

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    async def append_log(self, action: str, details: str) -> str:
        """Append an entry to wiki/log.md."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        entry = f"\n## [{now}] {action}\n\n{details}\n"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            header = "---\ntype: log\n---\n\n# Wiki Log\n"
            async with aiofiles.open(self.log_path, "w", encoding="utf-8") as f:
                await f.write(header)
        async with aiofiles.open(self.log_path, "a", encoding="utf-8") as f:
            await f.write(entry)
        return f"Logged: {action}"

    # ------------------------------------------------------------------
    # Lint / health check
    # ------------------------------------------------------------------

    async def lint(self) -> dict[str, Any]:
        """Run a health check on the wiki vault.

        Checks for:
        - Orphan pages (no inbound wikilinks)
        - Broken wikilinks (target page missing)
        - Pages missing frontmatter
        - Missing index entries
        """
        pages = await self.list_pages("wiki")
        page_names = {pg["name"] for pg in pages}

        # Build normalized lookup: lowercase, spaces↔hyphens, stripped
        def _normalize(name: str) -> str:
            return name.lower().replace(" ", "-").replace("_", "-").strip()

        normalized_lookup: set[str] = {_normalize(n) for n in page_names}

        # Collect all outbound wikilinks and inbound counts
        link_re = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
        heading_re = re.compile(r"^#{1,3}\s+(.+)", re.MULTILINE)
        inbound: dict[str, int] = {name: 0 for name in page_names}
        broken_links: list[dict[str, Any]] = []
        missing_frontmatter: list[str] = []
        # outbound: slug → {(target_slug, in_connections_section)}
        outbound: dict[str, set[str]] = {}

        for pg in pages:
            data = await self.read_page(pg["path"])
            body = data.get("body", "")
            meta = data.get("metadata", {})
            slug = pg["name"]
            if not meta:
                missing_frontmatter.append(pg["path"])

            # Find Connections section span for reciprocal scoping
            connections_spans: list[tuple[int, int]] = []
            headings = list(heading_re.finditer(body))
            for i, m in enumerate(headings):
                if "connection" in m.group(1).lower():
                    end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
                    connections_spans.append((m.start(), end))

            def _in_connections(pos: int) -> bool:
                return any(s <= pos < e for s, e in connections_spans)

            outbound[slug] = set()
            for match in link_re.finditer(body):
                target = match.group(1).strip()
                norm_target = _normalize(target)
                matched = False
                for pname in page_names:
                    if _normalize(pname) == norm_target:
                        inbound[pname] += 1
                        outbound[slug].add(_normalize(pname))
                        matched = True
                        break
                if not matched and norm_target not in normalized_lookup:
                    broken_links.append({"source": pg["path"], "target": target})

        orphans = [name for name, count in inbound.items()
                   if count == 0 and name not in ("index", "log")]

        # --- Reciprocal link check (concept/entity pages, Connections section) ---
        concept_entity_slugs = {
            pg["name"] for pg in pages
            if pg["path"].startswith(("wiki/concepts/", "wiki/entities/"))
            and pg["name"] not in ("index", "log", "tag-taxonomy")
        }
        non_reciprocal: list[dict[str, str]] = []
        for slug_a in concept_entity_slugs:
            for slug_b in (outbound.get(slug_a, set()) & concept_entity_slugs):
                if slug_a not in outbound.get(slug_b, set()):
                    non_reciprocal.append({"source": slug_a, "missing_back_link": slug_b})

        # --- Tag consistency check ---
        non_preferred_tags: list[dict[str, Any]] = []
        use_map = await self._load_tag_taxonomy()
        if use_map:
            for pg in pages:
                data = await self.read_page(pg["path"])
                meta = data.get("metadata", {})
                raw_tags = meta.get("tags", [])
                if isinstance(raw_tags, str):
                    raw_tags = [t.strip() for t in raw_tags.strip("[]").split(",")]
                for tag in raw_tags:
                    tag = tag.strip()
                    if tag in use_map:
                        non_preferred_tags.append({
                            "page": pg["path"],
                            "tag": tag,
                            "use_instead": use_map[tag],
                        })

        return {
            "total_pages": len(pages),
            "orphan_pages": orphans,
            "broken_links": broken_links,
            "missing_frontmatter": missing_frontmatter,
            "non_reciprocal_links": non_reciprocal,
            "non_preferred_tags": non_preferred_tags,
        }

    async def get_wikilink_neighbors(
        self, page_slugs: list[str]
    ) -> dict[str, list[tuple[str, float]]]:
        """Return weighted wikilink neighbours of a set of pages.

        Links found in a '## Connections' section carry weight 1.0
        (deliberate associative pointer).  Links in body prose carry
        weight 0.5 (contextual reference — may be a mere mention).

        Returns:
            Dict mapping slug → list of (linked_slug, weight) tuples.
        """
        link_re = re.compile(r"\[\[([^\]|#]+)")
        # Heading detector — capture the heading text
        heading_re = re.compile(r"^#{1,3}\s+(.+)", re.MULTILINE)
        neighbours: dict[str, list[tuple[str, float]]] = {}

        for slug in page_slugs:
            found: Path | None = None
            for subdir in ("entities", "concepts", "sources"):
                candidate = self.wiki_dir / subdir / f"{slug}.md"
                if candidate.exists():
                    found = candidate
                    break

            if not found:
                neighbours[slug] = []
                continue

            async with aiofiles.open(found, encoding="utf-8") as f:
                content = await f.read()

            _, body = parse_frontmatter(content)

            # Split body into sections; tag each span with its heading
            section_weights: list[tuple[int, int, float]] = []  # (start, end, weight)
            headings = list(heading_re.finditer(body))
            for i, m in enumerate(headings):
                end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
                heading_text = m.group(1).strip().lower()
                weight = 1.0 if "connection" in heading_text else 0.5
                section_weights.append((m.start(), end, weight))

            def _weight_for_pos(pos: int) -> float:
                for start, end, w in section_weights:
                    if start <= pos < end:
                        return w
                return 0.5  # pre-first-heading prose

            seen: dict[str, float] = {}
            for match in link_re.finditer(body):
                target = match.group(1).strip().lower().replace(" ", "-")
                if not target or target == slug:
                    continue
                w = _weight_for_pos(match.start())
                # Keep highest weight if seen more than once
                seen[target] = max(seen.get(target, 0.0), w)

            neighbours[slug] = list(seen.items())

        return neighbours

    async def compute_wikilink_hits(self) -> dict[str, dict[str, float]]:
        """Compute HITS hub and authority scores on the wiki wikilink graph.

        Authorities = pages cited by many others (load-bearing knowledge nodes).
        Hubs = pages that link to many good authorities (navigation layers).

        Returns:
            Dict mapping slug → {'hub': float, 'authority': float}
        """
        import networkx as nx

        link_re = re.compile(r"\[\[([^\]|#]+)")
        pages = await self.list_pages("wiki")
        G: nx.DiGraph = nx.DiGraph()

        for pg in pages:
            if pg["name"] in ("index", "log"):
                continue
            data = await self.read_page(pg["path"])
            body = data.get("body", "")
            slug = pg["name"]
            G.add_node(slug)
            for match in link_re.finditer(body):
                target = match.group(1).strip().lower().replace(" ", "-")
                if target and target != slug:
                    G.add_edge(slug, target)

        if G.number_of_nodes() < 2:
            return {}

        try:
            hubs, authorities = nx.hits(G, max_iter=100, normalized=True)
        except nx.PowerIterationFailedConvergence:
            # Fall back to in/out degree normalised
            n = G.number_of_nodes()
            authorities = {v: G.in_degree(v) / max(n - 1, 1) for v in G.nodes()}
            hubs = {v: G.out_degree(v) / max(n - 1, 1) for v in G.nodes()}

        return {
            node: {"hub": round(hubs.get(node, 0.0), 4),
                   "authority": round(authorities.get(node, 0.0), 4)}
            for node in G.nodes()
        }

    async def _load_tag_taxonomy(self) -> dict[str, str]:
        """Parse tag-taxonomy.md and return {non_preferred: preferred} mapping."""
        taxonomy_path = self.wiki_dir / "concepts" / "tag-taxonomy.md"
        if not taxonomy_path.exists():
            return {}
        async with aiofiles.open(taxonomy_path, encoding="utf-8") as f:
            content = await f.read()
        # Parse table rows: | `non-preferred` | `preferred` |
        row_re = re.compile(r"\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|")
        return {
            m.group(1): m.group(2)
            for m in row_re.finditer(content)
            if m.group(1) != "Tag used"  # skip header row
        }

    async def cluster_wiki_pages(
        self, n_clusters: int | None = None
    ) -> dict[str, Any]:
        """Cluster wiki pages by semantic similarity using GAAC (TF-IDF).

        Uses Group-Average Agglomerative Clustering — preferred over k-means
        because it doesn't require pre-specifying K and avoids chaining/outlier
        problems. Auto-selects K as sqrt(n_pages) rounded, or uses n_clusters
        if provided.

        Returns:
            {
              'clusters': [{id, pages, missing_links}],
              'merge_candidates': [(slug_a, slug_b, similarity)],
            }
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        pages = await self.list_pages("wiki")
        pages = [p for p in pages if p["name"] not in ("index", "log")]
        if len(pages) < 3:
            return {"clusters": [], "merge_candidates": []}

        # Build corpus from page bodies
        slugs: list[str] = []
        texts: list[str] = []
        for pg in pages:
            data = await self.read_page(pg["path"])
            body = data.get("body", "")
            if body.strip():
                slugs.append(pg["name"])
                texts.append(body)

        if len(texts) < 3:
            return {"clusters": [], "merge_candidates": []}

        # TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        matrix = vectorizer.fit_transform(texts).toarray()
        sim_matrix = cosine_similarity(matrix)
        # Distance matrix for clustering
        dist_matrix = 1.0 - sim_matrix
        np.clip(dist_matrix, 0.0, None, out=dist_matrix)  # numerical safety

        # Auto K
        k = n_clusters or max(2, round(len(slugs) ** 0.5))
        k = min(k, len(slugs) - 1)

        clustering = AgglomerativeClustering(
            n_clusters=k, linkage="average", metric="precomputed"
        )
        labels = clustering.fit_predict(dist_matrix)

        # Build current wikilink adjacency for missing-link detection
        link_re = re.compile(r"\[\[([^\]|#]+)")
        adjacency: dict[str, set[str]] = {s: set() for s in slugs}
        for pg in pages:
            if pg["name"] not in adjacency:
                continue
            data = await self.read_page(pg["path"])
            body = data.get("body", "")
            for m in link_re.finditer(body):
                t = m.group(1).strip().lower().replace(" ", "-")
                if t and t != pg["name"]:
                    adjacency[pg["name"]].add(t)

        # Group pages by cluster
        cluster_map: dict[int, list[str]] = {}
        for slug, label in zip(slugs, labels):
            cluster_map.setdefault(int(label), []).append(slug)

        clusters = []
        for cid, members in sorted(cluster_map.items()):
            # Find pairs in this cluster with no link in either direction
            missing = []
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    if b not in adjacency.get(a, set()) and a not in adjacency.get(b, set()):
                        missing.append((a, b))
            clusters.append({
                "id": cid,
                "pages": members,
                "missing_links": missing,
            })

        # Find high-similarity pairs as merge candidates (sim > 0.7)
        merge_candidates = []
        for i in range(len(slugs)):
            for j in range(i + 1, len(slugs)):
                if sim_matrix[i, j] > 0.7:
                    merge_candidates.append(
                        (slugs[i], slugs[j], round(float(sim_matrix[i, j]), 3))
                    )
        merge_candidates.sort(key=lambda x: x[2], reverse=True)

        return {"clusters": clusters, "merge_candidates": merge_candidates[:10]}

    # ------------------------------------------------------------------
    # File lifecycle
    # ------------------------------------------------------------------

    async def move_to_clippings(self, filename: str) -> str:
        """Move a processed raw file to Clippings/ archive.

        Called automatically after successful wiki_ingest_raw so that
        raw/ stays clean as an inbox-only directory.

        Args:
            filename: Filename inside raw/ (not a full path).

        Returns:
            Status string.
        """
        src = self.raw_dir / filename
        clippings_dir = self.vault_path / "Clippings"
        clippings_dir.mkdir(parents=True, exist_ok=True)
        dest = clippings_dir / filename

        if not src.exists():
            return f"Source not found: raw/{filename}"

        # Avoid clobbering existing Clippings file
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            dest = clippings_dir / f"{stem}_{ts}{suffix}"

        src.rename(dest)
        logger.info("Moved raw/%s → Clippings/%s", filename, dest.name)
        return f"Moved raw/{filename} → Clippings/{dest.name}"

    # ------------------------------------------------------------------
    # Graph sync helpers
    # ------------------------------------------------------------------

    async def get_sync_manifest(self) -> list[dict[str, Any]]:
        """Return a list of wiki pages with hashes for delta-sync with Neo4j."""
        pages = await self.list_pages("wiki")
        manifest: list[dict[str, Any]] = []
        for pg in pages:
            data = await self.read_page(pg["path"])
            body = data.get("body", "")
            content_hash = hashlib.sha256(body.encode()).hexdigest()[:12]
            manifest.append({
                "path": pg["path"],
                "name": pg["name"],
                "hash": content_hash,
                "metadata": data.get("metadata", {}),
            })
        return manifest
