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
        inbound: dict[str, int] = {name: 0 for name in page_names}
        broken_links: list[dict[str, Any]] = []
        missing_frontmatter: list[str] = []

        for pg in pages:
            data = await self.read_page(pg["path"])
            body = data.get("body", "")
            meta = data.get("metadata", {})
            if not meta:
                missing_frontmatter.append(pg["path"])
            for match in link_re.finditer(body):
                target = match.group(1).strip()
                norm_target = _normalize(target)
                # Check if any page matches (case-insensitive, hyphen/space agnostic)
                matched = False
                for pname in page_names:
                    if _normalize(pname) == norm_target:
                        inbound[pname] += 1
                        matched = True
                        break
                if not matched and norm_target not in normalized_lookup:
                    broken_links.append({"source": pg["path"], "target": target})

        orphans = [name for name, count in inbound.items()
                    if count == 0 and name not in ("index", "log")]

        return {
            "total_pages": len(pages),
            "orphan_pages": orphans,
            "broken_links": broken_links,
            "missing_frontmatter": missing_frontmatter,
        }

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
