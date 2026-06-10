"""
Wiki Adapter for Project Synapse.

Bridges the Obsidian Markdown vault (LLM-WIKI) with the Neo4j knowledge graph.
Provides tools for:
  - Reading/writing wiki pages with frontmatter
  - Syncing wiki content to/from the graph
  - Linting the wiki for health checks
  - Listing and searching wiki pages via the index
"""

import json
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
from .parser import build_frontmatter, parse_frontmatter
from .vault_index import VaultIndex

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Frontmatter helpers are imported from .parser
# ---------------------------------------------------------------------------


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
        self.vault_index = VaultIndex(self.vault_path)
        self.logger = logger

    @logger.timer()
    async def initialize(self) -> None:
        """Ensure vault directories exist."""
        if not self.vault_path or not self.vault_path.exists():
            logger.warning("Wiki vault path not found: %s", self.vault_path)
            return
        for d in [self.raw_dir, self.wiki_dir]:
            d.mkdir(parents=True, exist_ok=True)
        await self.vault_index.initialize()
        # NOTE: no vault_index.sync() here — a full scan of a large vault
        # takes 25s+ and blocks MCP startup past the client's handshake
        # timeout. Wiki tools sync lazily on first use, and the server warms
        # the index in a background task right after startup.
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
    async def list_pages(
        self,
        subdir: str = "wiki",
        limit: int = 10000,
        offset: int = 0,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all .md pages in a vault subdirectory via the index."""
        await self.vault_index.sync()
        res = await self.vault_index.list_pages(
            subdir=subdir, limit=limit, offset=offset, tag=tag
        )
        return res["pages"]

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

            if full.exists():
                try:
                    async with aiofiles.open(full, encoding="utf-8") as f:
                        old_content = await f.read()
                    old_meta, _ = parse_frontmatter(old_content)
                    merged_meta = dict(old_meta)
                    merged_meta.update(meta)
                    meta = merged_meta
                except Exception as e:
                    logger.warning(
                        "Could not read existing metadata for %s: %s", rel_path, e
                    )

            meta["updated"] = now
            if not full.exists() or "created" not in meta:
                meta.setdefault("created", now)

            content = build_frontmatter(meta) + "\n\n" + body.strip() + "\n"

            tmp_full = full.with_suffix(full.suffix + ".tmp")
            async with aiofiles.open(tmp_full, "w", encoding="utf-8") as f:
                await f.write(content)
            os.replace(tmp_full, full)
            logger.info("Wrote wiki page: %s", rel_path)
            await self.vault_index.upsert_page(rel_path)
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

        # --- 4. Duplicate frontmatter detection ---
        if _re.match(r"^\s*---\n[\s\S]*?\n---\n", body):
            raise WikiError(
                "Body contains a frontmatter block (starts with `---`). "
                "This will result in duplicate frontmatter because write_page "
                "automatically injects frontmatter."
            )

        return warnings

    async def delete_page(self, rel_path: str) -> str:
        """Delete a wiki page."""
        full = self.vault_path / rel_path
        if full.exists():
            full.unlink()
            await self.vault_index.remove_page(rel_path)
            return f"Deleted {rel_path}"
        return f"Not found: {rel_path}"

    # ------------------------------------------------------------------
    # Search & Index
    # ------------------------------------------------------------------

    @logger.timer()
    async def search_pages(
        self, query: str, subdir: str = "wiki", limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search wiki pages using the index."""
        return await self.vault_index.search(query, subdir=subdir, limit=limit)

    @logger.timer()
    async def update_index(self, deep: bool = False) -> str:
        """Rebuild wiki/index.md from all indexed pages."""
        await self.vault_index.sync()

        if deep:
            logger.info("Performing deep index refresh and health check")
            report = await self.lint()
            logger.info(
                f"Health check complete: {len(report['broken_links'])} broken links, "
                f"{len(report['orphan_pages'])} orphan pages found."
            )

        pages = await self.vault_index._query_dicts(
            "SELECT name, summary FROM pages WHERE is_operational = FALSE"
        )

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
            summary = pg.get("summary") or ""
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
        """Append an entry to wiki/log.md.

        Defensive size guard: log.md is append-only and read by Obsidian, which
        chokes on very large files. Any single entry's details are capped so a
        caller accidentally passing a huge body (e.g. a full lint report) can no
        longer bloat the log. The full data should be the tool's return value,
        not the log entry.
        """
        max_detail_chars = 2000
        if details and len(details) > max_detail_chars:
            details = (
                details[:max_detail_chars]
                + f"\n…(truncated {len(details) - max_detail_chars} chars; "
                "full output is the tool's return value, not logged)"
            )
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
        """Run a health check on the wiki vault using the index."""
        await self.vault_index.sync()
        return await self.vault_index.lint()

    @logger.timer()
    async def get_wikilink_neighbors(
        self, page_slugs: list[str]
    ) -> dict[str, list[tuple[str, float]]]:
        """Return weighted wikilink neighbours of a set of pages."""
        await self.vault_index.sync()
        neighbors = {}
        slugs_lower = [s.lower() for s in page_slugs]
        if not slugs_lower:
            return {}
        placeholders = ", ".join("?" for _ in slugs_lower)

        rows = await self.vault_index._query_dicts(
            f"SELECT name, wikilinks FROM pages WHERE LOWER(name) IN ({placeholders})",
            slugs_lower,
        )

        row_map = {r["name"].lower(): r["wikilinks"] for r in rows}

        for slug in page_slugs:
            slug_lower = slug.lower()
            if slug_lower in row_map:
                try:
                    links = json.loads(row_map[slug_lower])
                    neighbors[slug] = [
                        (target, float(weight)) for target, weight in links
                    ]
                except Exception:
                    neighbors[slug] = []
            else:
                neighbors[slug] = []

        return neighbors

    @logger.timer()
    async def compute_wikilink_hits(self) -> dict[str, dict[str, float]]:
        """Compute HITS hub and authority scores on the wiki wikilink graph."""
        await self.vault_index.sync()
        rows = await self.vault_index._query_dicts(
            "SELECT name, wikilinks FROM pages WHERE is_operational = FALSE"
        )
        graph: nx.DiGraph = nx.DiGraph()
        for row in rows:
            slug = row["name"]
            graph.add_node(slug)
            try:
                links = json.loads(row["wikilinks"])
                for target_slug, _ in links:
                    target_slug_norm = target_slug.lower().replace(" ", "-")
                    if target_slug_norm and target_slug_norm != slug.lower().replace(
                        " ", "-"
                    ):
                        graph.add_edge(slug, target_slug)
            except Exception:
                continue

        if graph.number_of_nodes() < 2:
            return {}

        try:
            hubs, authorities = nx.hits(graph, max_iter=100, normalized=True)
        except nx.PowerIterationFailedConvergence:
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
        """Cluster wiki pages by semantic similarity using GAAC (TF-IDF)."""
        await self.vault_index.sync()
        rows = await self.vault_index._query_dicts(
            "SELECT name, body, wikilinks FROM pages WHERE is_operational = FALSE"
        )

        slugs: list[str] = []
        texts: list[str] = []
        adjacency: dict[str, set[str]] = {}

        for row in rows:
            name = row["name"]
            body = row["body"]
            if body and body.strip():
                slugs.append(name)
                texts.append(body)

                try:
                    links = json.loads(row["wikilinks"])
                    adjacency[name] = {
                        target.lower().replace(" ", "-") for target, _ in links
                    }
                except Exception:
                    adjacency[name] = set()

        if len(texts) < 3:
            return {"clusters": [], "merge_candidates": []}

        # TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        sparse_matrix = vectorizer.fit_transform(texts)
        matrix = csr_matrix(sparse_matrix).toarray()
        sim_matrix = cosine_similarity(matrix)
        dist_matrix = 1.0 - sim_matrix
        dist_matrix.clip(min=0.0, out=dist_matrix)  # numerical safety

        k = n_clusters or max(2, round(len(slugs) ** 0.5))
        k = min(k, len(slugs) - 1)

        clustering = AgglomerativeClustering(
            n_clusters=k, linkage="average", metric="precomputed"
        )
        labels = clustering.fit_predict(dist_matrix)

        cluster_map: dict[int, list[str]] = {}
        for slug, label in zip(slugs, labels):
            cluster_map.setdefault(int(label), []).append(slug)

        clusters = []
        for cid, members in sorted(cluster_map.items()):
            missing = []
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    if b.lower().replace(" ", "-") not in adjacency.get(
                        a, set()
                    ) and a.lower().replace(" ", "-") not in adjacency.get(b, set()):
                        missing.append((a, b))
            clusters.append(
                {
                    "id": cid,
                    "pages": members,
                    "missing_links": missing[:15],
                    "total_missing_links": len(missing),
                }
            )

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
        await self.vault_index.sync()
        rows = await self.vault_index._query_dicts(
            "SELECT path, name, content_hash AS hash, frontmatter FROM pages WHERE subdir = 'wiki'"
        )
        for row in rows:
            try:
                row["metadata"] = json.loads(row.pop("frontmatter", "{}"))
            except Exception:
                row["metadata"] = {}
        return rows
