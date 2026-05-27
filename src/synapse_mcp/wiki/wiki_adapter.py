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

import aiofiles
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.exceptions import (
    WikiAccessError,
    WikiError,
    WikiIndexError,
    WikiPageNotFoundError,
)
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
    return meta, content[end + 3 :].strip()


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
        self.vault_path = Path(vault_path or os.getenv("WIKI_VAULT_PATH", ""))
        self.repo_url = repo_url or os.getenv("WIKI_GITHUB_REPO", "")
        # Sub-directories following the Karpathy 3-layer architecture
        self.raw_dir = self.vault_path / "raw"
        self.wiki_dir = self.vault_path / "wiki"
        self.schema_path = self.vault_path / "CLAUDE.md"
        self.index_path = self.wiki_dir / "index.md"
        self.log_path = self.wiki_dir / "log.md"
        self.logger = logger

    @logger.timer()
    async def initialize(self) -> None:
        """Ensure vault directories exist."""
        if not self.vault_path or not self.vault_path.exists():
            logger.warning("Wiki vault path not found: %s", self.vault_path)
            return
        for d in [self.raw_dir, self.wiki_dir]:
            d.mkdir(parents=True, exist_ok=True)
        await self.check_health()
        logger.info("Wiki adapter initialised – vault: %s", self.vault_path)

    async def check_health(self) -> bool:
        """Verify wiki vault accessibility and write permissions."""
        if not self.vault_path:
            raise RuntimeError("Wiki vault path not configured")
        if not self.vault_path.exists():
            raise RuntimeError(f"Wiki vault path does not exist: {self.vault_path}")
        if not self.vault_path.is_dir():
            raise RuntimeError(f"Wiki vault path is not a directory: {self.vault_path}")

        # Check write permissions by attempting to write a tiny hidden file
        health_file = self.vault_path / ".synapse_health"
        try:
            async with aiofiles.open(health_file, "w") as f:
                await f.write("ok")
            health_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Wiki health check failed (write permission): {e}")
            raise RuntimeError(f"Wiki vault is not writable: {str(e)}") from e

    # ------------------------------------------------------------------
    # Page CRUD
    # ------------------------------------------------------------------

    @logger.timer()
    async def list_pages(self, subdir: str = "wiki") -> list[dict[str, Any]]:
        """List all .md pages in a vault subdirectory with frontmatter metadata."""
        target = self.vault_path / subdir
        if not target.exists():
            return []

        pages: list[dict[str, Any]] = []
        # Use a list to avoid issues if files are deleted during iteration
        file_list = sorted(target.rglob("*.md"))

        for p in file_list:
            try:
                rel = p.relative_to(self.vault_path)
                async with aiofiles.open(p, encoding="utf-8") as f:
                    content = await f.read()
                meta, _ = parse_frontmatter(content)
                pages.append({"path": str(rel), "name": p.stem, **meta})
            except (FileNotFoundError, PermissionError) as e:
                logger.warning(f"Skipping page {p} due to access error: {e}")
                continue
        return pages

    async def read_page(self, rel_path: str) -> dict[str, Any]:
        """Read a wiki page and return metadata + body."""
        full = self.vault_path / rel_path
        if not full.exists():
            raise WikiPageNotFoundError(rel_path)

        try:
            async with aiofiles.open(full, encoding="utf-8") as f:
                content = await f.read()
            meta, body = parse_frontmatter(content)
            return {"path": rel_path, "metadata": meta, "body": body}
        except PermissionError as e:
            raise WikiAccessError(f"Permission denied reading {rel_path}") from e
        except Exception as e:
            raise WikiError(f"Unexpected error reading {rel_path}: {str(e)}") from e

    async def write_page(
        self,
        rel_path: str,
        body: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Write or update a wiki page with frontmatter."""
        full = self.vault_path / rel_path

        # GBrain quality.md citation + backlink convention enforcement (non-blocking)
        convention_warnings = await self._enforce_page_conventions(rel_path, body)
        for warn in convention_warnings:
            logger.warning("Convention violation in %s: %s", rel_path, warn)

        try:
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
        except PermissionError as e:
            raise WikiAccessError(f"Permission denied writing to {rel_path}") from e
        except Exception as e:
            raise WikiError(f"Unexpected error writing to {rel_path}: {str(e)}") from e

    async def _enforce_page_conventions(self, rel_path: str, body: str) -> list[str]:
        """Check page body for GBrain citation + back-link convention violations.

        Non-blocking — returns a list of warning strings, not exceptions.
        Checks:
          1. All [[wikilinks]] resolve to an existing page.
          2. New pages declare at least one related page via wikilink or
             explicit ``## Related`` / ``## Connections`` section.
          3. Fenced fact blocks include a ``source:`` field.

        Returns
        -------
        list[str]
            Warning messages for each convention violation found.
        """
        import re as _re

        warnings: list[str] = []
        slug = rel_path.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".md")

        # --- 1. Broken wikilink detection ---
        link_re = _re.compile(r"\[\[([^\]|#]+)(?:\|[^]]+)?\]\]")
        wikilink_targets = link_re.findall(body)
        for target_raw in wikilink_targets:
            target_slug = target_raw.strip()
            target_path = self.vault_path / "wiki" / f"{target_slug}.md"
            if not target_path.exists():
                warnings.append(
                    f"Broken wikilink [[{target_slug}]] → no page found at "
                    f"wiki/{target_slug}.md"
                )

        # --- 2. Orphan-page detection (new pages only) ---
        full = self.vault_path / rel_path
        if not full.exists():
            has_related_section = bool(
                _re.search(r"^##\s+(?:Related|Connections|Links)", body, _re.M)
            )
            has_outbound_links = bool(wikilink_targets)
            if not has_related_section and not has_outbound_links:
                warnings.append(
                    f"New page wiki/{slug}.md has no ## Related section and no "
                    f"outbound [[wikilinks]] — it will be an orphan. "
                    f"Add at least one [[link]] or a ## Related section."
                )

        # --- 3. Citation enforcement for gbrain-facts blocks ---
        fence_re = _re.compile(r"```gbrain-facts\b(.*?)```", _re.DOTALL)
        for match in fence_re.finditer(body):
            block_content = match.group(1)
            if not _re.search(r"^\s*source\s*:", block_content, _re.M):
                warnings.append(
                    "gbrain-facts block is missing ``source:`` field — "
                    "add ``source: <url or page>`` for citation traceability."
                )

        return warnings

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

    @logger.timer()
    async def search_pages(
        self, query: str, subdir: str = "wiki"
    ) -> list[dict[str, Any]]:
        """Simple keyword search across wiki pages (case-insensitive)."""
        results: list[dict[str, Any]] = []
        target = self.vault_path / subdir
        if not target.exists():
            return results
        terms = query.lower().split()
        for p in target.rglob("*.md"):
            try:
                async with aiofiles.open(p, encoding="utf-8") as f:
                    content = (await f.read()).lower()
                if all(t in content for t in terms):
                    rel = str(p.relative_to(self.vault_path))
                    results.append({"path": rel, "name": p.stem})
            except (FileNotFoundError, PermissionError):
                continue
        return results

    @logger.timer()
    async def update_index(self, deep: bool = False) -> str:
        """Rebuild wiki/index.md from all wiki pages.

        Args:
            deep: If True, performs a disk-level verification of all indexed files
                  and runs a linting pass to identify other inconsistencies.
        """
        if deep:
            logger.info("Performing deep index refresh and health check")
            report = await self.lint()
            logger.info(
                f"Health check complete: {len(report['broken_links'])} broken links, "
                f"{len(report['orphan_pages'])} orphan pages found."
            )

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

        indexed_count = 0
        for pg in pages:
            if pg["name"] in ("index", "log"):
                continue

            # Verify file exists on disk if deep mode is on (though list_pages already did this)
            if deep:
                full_path = self.vault_path / pg["path"]
                if not full_path.exists():
                    logger.warning(
                        f"Indexed page {pg['path']} missing from disk, skipping"
                    )
                    continue

            summary = pg.get("summary", "")
            lines.append(f"- [[{pg['name']}]] — {summary}")
            indexed_count += 1

        lines.append("")
        content = "\n".join(lines)

        try:
            async with aiofiles.open(self.index_path, "w", encoding="utf-8") as f:
                await f.write(content)

            msg = f"Index updated with {indexed_count} pages"
            if deep:
                msg += " (Deep refresh completed)"
            return msg
        except Exception as e:
            raise WikiIndexError(f"Failed to write wiki index: {str(e)}") from e

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

    @logger.timer()
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
        # Strip code fences and inline code before scanning for links
        code_fence_re = re.compile(r"```.*?```", re.DOTALL)
        inline_code_re = re.compile(r"`[^`]+`")
        inbound: dict[str, int] = {name: 0 for name in page_names}
        broken_links: list[dict[str, Any]] = []
        missing_frontmatter: list[str] = []
        # outbound: slug → {(target_slug, in_connections_section)}
        outbound: dict[str, set[str]] = {}

        for pg in pages:
            # Exclude log.md — it contains raw historical entries with wikilink
            # text that are not real links, always producing false positives.
            if pg["name"] in ("log", "index"):
                continue
            data = await self.read_page(pg["path"])
            body = data.get("body", "")
            # Strip code fences and inline code so examples don't register as links
            body_clean = code_fence_re.sub("", body)
            body_clean = inline_code_re.sub("", body_clean)
            meta = data.get("metadata", {})
            slug = pg["name"]
            if not meta:
                missing_frontmatter.append(pg["path"])

            # Link properties
            outbound[slug] = set()
            for match in link_re.finditer(body_clean):
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

        orphans = [
            name
            for name, count in inbound.items()
            if count == 0 and name not in ("index", "log")
        ]

        # --- Reciprocal link check (concept/entity pages, Connections section) ---
        concept_entity_slugs = {
            pg["name"]
            for pg in pages
            if pg["path"].startswith(("wiki/concepts/", "wiki/entities/"))
            and pg["name"] not in ("index", "log", "tag-taxonomy")
        }
        non_reciprocal: list[dict[str, str]] = []
        for slug_a in concept_entity_slugs:
            for slug_b in outbound.get(slug_a, set()) & concept_entity_slugs:
                if slug_a not in outbound.get(slug_b, set()):
                    non_reciprocal.append(
                        {"source": slug_a, "missing_back_link": slug_b}
                    )

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
                        non_preferred_tags.append(
                            {
                                "page": pg["path"],
                                "tag": tag,
                                "use_instead": use_map[tag],
                            }
                        )

        return {
            "total_pages": len(pages),
            "orphan_pages": orphans,
            "broken_links": broken_links,
            "missing_frontmatter": missing_frontmatter,
            "non_reciprocal_links": non_reciprocal,
            "non_preferred_tags": non_preferred_tags,
        }

    @logger.timer()
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

            def _weight_for_pos(
                pos: int, weights: list[tuple[int, int, float]]
            ) -> float:
                for start, end, w in weights:
                    if start <= pos < end:
                        return w
                return 0.5  # pre-first-heading prose

            seen: dict[str, float] = {}
            for match in link_re.finditer(body):
                target = match.group(1).strip().lower().replace(" ", "-")
                if not target or target == slug:
                    continue
                w = _weight_for_pos(match.start(), section_weights)
                # Keep highest weight if seen more than once
                seen[target] = max(seen.get(target, 0.0), w)

            neighbours[slug] = list(seen.items())

        return neighbours

    @logger.timer()
    async def compute_wikilink_hits(self) -> dict[str, dict[str, float]]:
        """Compute HITS hub and authority scores on the wiki wikilink graph.

        Authorities = pages cited by many others (load-bearing knowledge nodes).
        Hubs = pages that link to many good authorities (navigation layers).

        Returns:
            Dict mapping slug → {'hub': float, 'authority': float}
        """
        link_re = re.compile(r"\[\[([^\]|#]+)")
        pages = await self.list_pages("wiki")
        graph: nx.DiGraph = nx.DiGraph()

        for pg in pages:
            if pg["name"] in ("index", "log"):
                continue
            data = await self.read_page(pg["path"])
            body = data.get("body", "")
            slug = pg["name"]
            graph.add_node(slug)
            for match in link_re.finditer(body):
                target = match.group(1).strip().lower().replace(" ", "-")
                if target and target != slug:
                    graph.add_edge(slug, target)

        if graph.number_of_nodes() < 2:
            return {}

        try:
            hubs, authorities = nx.hits(graph, max_iter=100, normalized=True)
        except nx.PowerIterationFailedConvergence:
            # Fall back to in/out degree normalised
            n = graph.number_of_nodes()
            authorities = {v: graph.in_degree(v) / max(n - 1, 1) for v in graph.nodes()}
            hubs = {v: graph.out_degree(v) / max(n - 1, 1) for v in graph.nodes()}

        return {
            node: {
                "hub": round(hubs.get(node, 0.0), 4),
                "authority": round(authorities.get(node, 0.0), 4),
            }
            for node in graph.nodes()
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

    @logger.timer()
    async def cluster_wiki_pages(self, n_clusters: int | None = None) -> dict[str, Any]:
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
        sparse_matrix = vectorizer.fit_transform(texts)
        matrix = csr_matrix(sparse_matrix).toarray()
        sim_matrix = cosine_similarity(matrix)
        # Distance matrix for clustering
        dist_matrix = 1.0 - sim_matrix
        dist_matrix.clip(min=0.0, out=dist_matrix)  # numerical safety

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
                for b in members[i + 1 :]:
                    if b not in adjacency.get(a, set()) and a not in adjacency.get(
                        b, set()
                    ):
                        missing.append((a, b))
            clusters.append(
                {
                    "id": cid,
                    "pages": members,
                    "missing_links": missing,
                }
            )

        # Find high-similarity pairs as merge candidates (sim > 0.7)
        merge_candidates = []
        for i, slug_a in enumerate(slugs):
            for j in range(i + 1, len(slugs)):
                if sim_matrix[i, j] > 0.7:
                    merge_candidates.append(
                        (slug_a, slugs[j], round(float(sim_matrix[i, j]), 3))
                    )
        merge_candidates.sort(key=lambda x: x[2], reverse=True)

        return {"clusters": clusters, "merge_candidates": merge_candidates[:10]}

    # ------------------------------------------------------------------
    # File lifecycle
    # ------------------------------------------------------------------

    def _infer_content_type(self, filename: str = "", source_url: str = "") -> str:
        """Classify a source file into one of four archive types.

        Routing logic (checked in order — first match wins):
          papers       → academic publishers, preprint servers, DOIs
          repositories → code hosting platforms
          documentation→ official docs, specs, skills, READMEs
          articles     → everything else (default)
        """
        url = source_url.lower()
        name = filename.lower()

        paper_signals = [
            "arxiv.org",
            "doi.org",
            "pubmed",
            "sciencedirect.com",
            "nature.com",
            "science.org",
            "acm.org",
            "ieee.org",
            "springer.com",
            "wiley.com",
            "plos",
            "biorxiv",
            "medrxiv",
            "ncbi.nlm.nih.gov",
            "semanticscholar.org",
            "sciadv",
        ]
        repo_signals = [
            "github.com",
            "gitlab.com",
            "pypi.org",
            "npmjs.com",
            "crates.io",
            "huggingface.co/",
            "bitbucket.org",
        ]
        doc_signals = [
            "docs.",
            "/docs/",
            "help.",
            "/documentation/",
            "developer.",
            "spec.",
            "standard",
            "niso",
            "ansi",
            "skill.md",
            "readme",
            "publish.obsidian",
            "overleaf.com",
            "latex",
            "manual",
        ]

        if any(s in url for s in paper_signals) or name.endswith(".pdf"):
            return "papers"
        if any(s in url for s in repo_signals):
            return "repositories"
        if any(s in url or s in name for s in doc_signals):
            return "documentation"
        return "articles"

    @logger.timer()
    async def move_to_clippings(self, filename: str, source_url: str = "") -> str:
        """Move a processed raw file into the typed Clippings archive.

        Routes to  Clippings/<type>/<YYYY>/<filename>  so the archive
        stays navigable as it grows. Type is inferred from source URL or
        filename; year is always the current year.

        Args:
            filename:   Filename inside raw/ (not a full path).
            source_url: Original URL of the source, used for type routing.
        """
        src = self.raw_dir / filename
        if not src.exists():
            return f"Source not found: raw/{filename}"

        content_type = self._infer_content_type(filename, source_url)
        year = datetime.now(timezone.utc).strftime("%Y")
        dest_dir = self.vault_path / "Clippings" / content_type / year
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename

        # Avoid clobbering an existing file
        if dest.exists():
            ts = datetime.now(timezone.utc).strftime("%H%M%S")
            dest = dest_dir / f"{dest.stem}_{ts}{dest.suffix}"

        src.rename(dest)
        rel = dest.relative_to(self.vault_path)
        logger.info("Archived raw/%s → %s", filename, rel)
        return f"Archived to Clippings/{content_type}/{year}/{dest.name}"

    # ------------------------------------------------------------------
    # Graph sync helpers
    # ------------------------------------------------------------------

    @logger.timer()
    async def get_sync_manifest(self) -> list[dict[str, Any]]:
        """Return a list of wiki pages with hashes for delta-sync with Neo4j."""
        pages = await self.list_pages("wiki")
        manifest: list[dict[str, Any]] = []
        for pg in pages:
            data = await self.read_page(pg["path"])
            body = data.get("body", "")
            content_hash = hashlib.sha256(body.encode()).hexdigest()[:12]
            manifest.append(
                {
                    "path": pg["path"],
                    "name": pg["name"],
                    "hash": content_hash,
                    "metadata": data.get("metadata", {}),
                }
            )
        return manifest
