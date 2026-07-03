"""
Persistent page index for Project Synapse.

Backed by SQLite in WAL mode, providing incremental sync, fast metadata/search
queries, and fluid multi-process sharing: WAL lets any number of Synapse
instances read concurrently while writers queue briefly (busy_timeout), so a
second instance no longer falls back to a throwaway in-memory index the way
the old DuckDB backend (single-process exclusive lock) forced it to.
"""

import asyncio
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..utils.logging_config import get_logger
from .parser import extract_outbound_links, parse_frontmatter

logger = get_logger(__name__)


@dataclass
class SyncResult:
    """Result of an incremental index sync operation."""

    added: int
    updated: int
    deleted: int
    total: int


class VaultIndex:
    """Persistent wiki page index backed by SQLite (WAL mode, multi-process safe)."""

    def __init__(self, vault_path: Path):
        self.vault_path = Path(vault_path)
        self.db_path = self.vault_path / ".synapse" / "vault_index.sqlite3"
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        # Kept for API compatibility; SQLite WAL never needs the in-memory
        # fallback the DuckDB backend used when another process held the lock.
        self._is_fallback = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get or open the SQLite connection (WAL mode, autocommit)."""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False: calls run via asyncio.to_thread (pool
            # threads vary); the asyncio lock already serialises access
            # within this process. isolation_level=None = autocommit, which
            # matches the previous DuckDB behaviour.
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                isolation_level=None,
                timeout=10.0,
            )
            # WAL: concurrent readers across processes, writers queue briefly.
            conn.execute("PRAGMA journal_mode=WAL")
            # Wait up to 5s on a contended write lock instead of erroring.
            conn.execute("PRAGMA busy_timeout=5000")
            # NORMAL is durable enough for a rebuildable derived cache and
            # much faster than FULL during bulk syncs.
            conn.execute("PRAGMA synchronous=NORMAL")
            self._conn = conn

            # Setup tables if they don't exist
            self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS pages (
                path          TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                subdir        TEXT NOT NULL,
                summary       TEXT DEFAULT '',
                tags          TEXT DEFAULT '[]',
                created       TEXT,
                updated       TEXT,
                word_count    INTEGER DEFAULT 0,
                content_hash  TEXT NOT NULL,
                frontmatter   TEXT DEFAULT '{}',
                wikilinks     TEXT DEFAULT '[]',
                is_operational BOOLEAN DEFAULT FALSE,
                mtime         DOUBLE NOT NULL,
                yaml_error    TEXT DEFAULT NULL,
                body          TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS link_graph (
                source_slug   TEXT NOT NULL,
                target_slug   TEXT NOT NULL,
                weight        REAL DEFAULT 0.5,
                PRIMARY KEY (source_slug, target_slug)
            );

            DROP INDEX IF EXISTS idx_pages_name;
            DROP INDEX IF EXISTS idx_pages_subdir;
            """)
        return self._conn

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as e:
                logger.warning(f"Error closing SQLite index: {e}")
            self._conn = None

    async def _execute(self, query: str, params: list[Any] | None = None) -> list[Any]:
        """Execute a query and return raw results."""
        async with self._lock:

            def _run() -> list[Any]:
                conn = self._get_connection()
                if params:
                    res = conn.execute(query, params)
                else:
                    res = conn.execute(query)
                try:
                    return res.fetchall()
                except Exception:
                    return []

            return await asyncio.to_thread(_run)

    async def _query_dicts(
        self, query: str, params: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as dictionaries."""
        async with self._lock:

            def _run() -> list[dict[str, Any]]:
                conn = self._get_connection()
                if params:
                    res = conn.execute(query, params)
                else:
                    res = conn.execute(query)
                if res.description:
                    cols = [c[0] for c in res.description]
                    return [dict(zip(cols, row)) for row in res.fetchall()]
                return []

            return await asyncio.to_thread(_run)

    def _is_operational(self, path: str, name: str) -> bool:
        """Check if a page is operational rather than knowledge."""
        if name in ("index", "log", "TEMPLATE"):
            return True
        op_prefixes = (
            "wiki/scratchpad/",
            "wiki/agents/",
            "wiki/audits/",
            "wiki/jobs/",
            "wiki/discovery/",
            "wiki/headlines/",
            "wiki/overseer/",
        )
        if path.startswith(op_prefixes):
            return True
        op_substrings = (
            "carryover",
            "batch-progress",
            "vault.md",
            "/audit-",
            "/ingest-",
            "agent-sheet",
        )
        return any(s in path for s in op_substrings)

    def _read_and_parse_file(self, rel_path: str) -> dict[str, Any]:
        """Read file from disk and parse all metadata."""
        full_path = self.vault_path / rel_path
        if not full_path.exists():
            raise FileNotFoundError(rel_path)

        mtime = full_path.stat().st_mtime
        content = full_path.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]

        meta, body = parse_frontmatter(content)

        # Validate frontmatter YAML
        yaml_error = None
        if content.startswith("---"):
            fm_end = content.find("\n---", 3)
            if fm_end != -1:
                try:
                    yaml.safe_load(content[3:fm_end])
                except yaml.YAMLError as ye:
                    yaml_error = str(ye).splitlines()[0] if str(ye) else "invalid YAML"

        word_count = len(body.split())
        slug = full_path.stem
        outbound = extract_outbound_links(body, slug)

        parts = rel_path.replace("\\", "/").split("/")
        subdir = parts[0] if parts else ""

        # Normalize tags
        tags = meta.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.strip("[]").split(",") if t.strip()]
        elif not isinstance(tags, list):
            tags = []

        is_operational = self._is_operational(rel_path, slug)

        # Format created/updated dates properly
        created = meta.get("created")
        if created:
            if hasattr(created, "isoformat"):
                created = created.isoformat()
            else:
                created = str(created)

        updated = meta.get("updated")
        if updated:
            if hasattr(updated, "isoformat"):
                updated = updated.isoformat()
            else:
                updated = str(updated)

        return {
            "path": rel_path,
            "name": slug,
            "subdir": subdir,
            "summary": meta.get("summary", ""),
            "tags": json.dumps(tags),
            "created": created,
            "updated": updated,
            "word_count": word_count,
            "content_hash": content_hash,
            "frontmatter": json.dumps(meta),
            "wikilinks": json.dumps(outbound),
            "is_operational": is_operational,
            "mtime": mtime,
            "yaml_error": yaml_error,
            "body": body,
        }

    async def initialize(self) -> None:
        """Initialize the database tables and write .gitignore config."""
        # Create .synapse directory if not exists
        syn_dir = self.vault_path / ".synapse"
        syn_dir.mkdir(parents=True, exist_ok=True)

        # Add to .gitignore
        gitignore = self.vault_path / ".gitignore"
        try:
            content = ""
            if gitignore.exists():
                content = gitignore.read_text(encoding="utf-8")
            if ".synapse/" not in content:
                with open(gitignore, "a", encoding="utf-8") as f:
                    f.write("\n# Synapse index\n.synapse/\n")
        except Exception as e:
            logger.warning(f"Could not update vault .gitignore: {e}")

        # Initialize SQLite connection to create tables
        async with self._lock:
            self._get_connection()

    async def sync(self) -> SyncResult:
        """Incremental filesystem scan to update index."""
        async with self._lock:
            self._get_connection()

        # 1. Walk filesystem to find all .md files and get their mtime
        fs_files = {}

        def _walk() -> None:
            for sdir in ("wiki", "raw"):
                target = self.vault_path / sdir
                if not target.exists():
                    continue
                for p in target.rglob("*.md"):
                    rel = p.relative_to(self.vault_path)
                    try:
                        fs_files[str(rel)] = p.stat().st_mtime
                    except Exception:
                        continue

        await asyncio.to_thread(_walk)

        # 2. Get all indexed files from DB
        db_files = {}
        rows = await self._query_dicts("SELECT path, mtime, content_hash FROM pages")
        for row in rows:
            db_files[row["path"]] = (row["mtime"], row["content_hash"])

        # 3. Identify added, updated, deleted
        to_add_update = []
        to_delete = []

        for rel_path, mtime in fs_files.items():
            if rel_path not in db_files:
                to_add_update.append(rel_path)
            else:
                db_mtime, _ = db_files[rel_path]
                # 1µs tolerance: SQLite stores the float64 mtime exactly
                # (the old 10ms slack was a DuckDB round-trip artifact and
                # made edits within 10ms of the last sync invisible).
                if abs(mtime - db_mtime) > 1e-6:
                    to_add_update.append(rel_path)

        for rel_path in db_files:
            if rel_path not in fs_files:
                to_delete.append(rel_path)

        added_count = 0
        updated_count = 0
        deleted_count = 0

        # 4. Read and parse changed/new files
        for rel_path in to_add_update:
            try:
                parsed = await asyncio.to_thread(self._read_and_parse_file, rel_path)
                is_update = rel_path in db_files

                query = """
                INSERT OR REPLACE INTO pages (
                    path, name, subdir, summary, tags, created, updated,
                    word_count, content_hash, frontmatter, wikilinks,
                    is_operational, mtime, yaml_error, body
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                params = [
                    parsed["path"],
                    parsed["name"],
                    parsed["subdir"],
                    parsed["summary"],
                    parsed["tags"],
                    parsed["created"],
                    parsed["updated"],
                    parsed["word_count"],
                    parsed["content_hash"],
                    parsed["frontmatter"],
                    parsed["wikilinks"],
                    parsed["is_operational"],
                    parsed["mtime"],
                    parsed["yaml_error"],
                    parsed["body"],
                ]
                await self._execute(query, params)

                if is_update:
                    updated_count += 1
                else:
                    added_count += 1
            except Exception as e:
                logger.error(f"Error parsing file during sync {rel_path}: {e}")

        # 5. Handle deleted files
        for rel_path in to_delete:
            await self._execute("DELETE FROM pages WHERE path = ?", [rel_path])
            deleted_count += 1

        # 6. Rebuild link_graph table
        if to_add_update or to_delete:
            await self._execute("DELETE FROM link_graph")
            # Include all pages except log/index templates so operational links count towards non-orphan status
            rows = await self._query_dicts(
                "SELECT name, wikilinks FROM pages WHERE name NOT IN ('index', 'log', 'TEMPLATE')"
            )

            insert_params = []
            for row in rows:
                source_slug = row["name"].lower().replace(" ", "-")
                try:
                    links = json.loads(row["wikilinks"])
                    for target_slug, weight in links:
                        target_slug_norm = target_slug.lower().replace(" ", "-")
                        insert_params.append((source_slug, target_slug_norm, weight))
                except Exception:
                    continue

            if insert_params:
                async with self._lock:

                    def _bulk_insert() -> None:
                        conn = self._get_connection()
                        deduped: dict[tuple[str, str], float] = {}
                        for s, t, w in insert_params:
                            deduped[(s, t)] = max(deduped.get((s, t), 0.0), w)

                        stmt_params = [(s, t, w) for (s, t), w in deduped.items()]
                        conn.executemany(
                            "INSERT OR REPLACE INTO link_graph (source_slug, target_slug, weight) VALUES (?, ?, ?)",
                            stmt_params,
                        )

                    await asyncio.to_thread(_bulk_insert)

        total_rows = await self._query_dicts("SELECT COUNT(*) as count FROM pages")
        total_count = total_rows[0]["count"] if total_rows else 0

        return SyncResult(
            added=added_count,
            updated=updated_count,
            deleted=deleted_count,
            total=total_count,
        )

    async def list_pages(
        self,
        subdir: str = "wiki",
        limit: int = 50,
        offset: int = 0,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Query the page index. Never touches the filesystem."""
        prefix = f"{subdir}/"

        # Build query
        query = "FROM pages WHERE (path = ? OR path LIKE ?)"
        params: list[Any] = [subdir, prefix + "%"]

        if tag:
            query += " AND tags LIKE ?"
            params.append(f'%"{tag}"%')

        # Get total count first
        count_rows = await self._query_dicts(
            f"SELECT COUNT(*) as count {query}", params
        )
        total = count_rows[0]["count"] if count_rows else 0

        # Order and paginate
        select_query = f"SELECT path, name, summary, tags, created, updated, word_count, is_operational, frontmatter {query} ORDER BY name ASC LIMIT ? OFFSET ?"
        params_with_limits = list(params)
        params_with_limits.extend([limit, offset])

        pages = await self._query_dicts(select_query, params_with_limits)

        # Unpack frontmatter keys to root level for backwards compatibility
        for pg in pages:
            fm_str = pg.pop("frontmatter", "{}")
            try:
                fm = json.loads(fm_str)
                if isinstance(fm, dict):
                    pg.update(fm)
            except Exception:
                pass
            # Parse tags
            if "tags" in pg and isinstance(pg["tags"], str):
                try:
                    pg["tags"] = json.loads(pg["tags"])
                except Exception:
                    pg["tags"] = []

        has_more = (offset + len(pages)) < total

        return {"pages": pages, "total": total, "has_more": has_more}

    async def search(
        self,
        query: str,
        subdir: str = "wiki",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search pages by query terms."""
        terms = query.lower().split()
        if not terms:
            return []

        sql = "SELECT path, name, summary, body FROM pages WHERE (path = ? OR path LIKE ?)"
        params: list[Any] = [subdir, f"{subdir}/%"]

        for term in terms:
            sql += " AND (LOWER(name) LIKE ? OR LOWER(summary) LIKE ? OR LOWER(body) LIKE ?)"
            like_term = f"%{term}%"
            params.extend([like_term, like_term, like_term])

        sql += " LIMIT ?"
        params.append(limit)

        rows = await self._query_dicts(sql, params)

        results = []
        for row in rows:
            body = row["body"] or ""
            summary = row["summary"] or ""

            excerpt = ""
            if body:
                body_lower = body.lower()
                first_pos = -1
                for term in terms:
                    pos = body_lower.find(term)
                    if pos != -1:
                        first_pos = pos
                        break

                if first_pos != -1:
                    start = max(0, first_pos - 100)
                    end = min(len(body), first_pos + 150)
                    snippet = body[start:end].replace("\n", " ").strip()
                    prefix = "..." if start > 0 else ""
                    suffix = "..." if end < len(body) else ""
                    excerpt = f"{prefix}{snippet}{suffix}"
                else:
                    excerpt = body[:200] + "..." if len(body) > 200 else body
            else:
                excerpt = summary

            results.append(
                {
                    "path": row["path"],
                    "name": row["name"],
                    "summary": summary,
                    "excerpt": excerpt,
                }
            )

        return results

    async def get_page_meta(self, path: str) -> dict[str, Any] | None:
        """Get metadata for a single page from the index."""
        rows = await self._query_dicts(
            "SELECT path, name, summary, tags, created, updated, word_count, is_operational, frontmatter, yaml_error FROM pages WHERE path = ?",
            [path],
        )
        if not rows:
            return None

        pg = rows[0]
        fm_str = pg.pop("frontmatter", "{}")
        try:
            fm = json.loads(fm_str)
            if isinstance(fm, dict):
                pg.update(fm)
        except Exception:
            pass

        if "tags" in pg and isinstance(pg["tags"], str):
            try:
                pg["tags"] = json.loads(pg["tags"])
            except Exception:
                pg["tags"] = []

        return pg

    async def get_page_hash(self, path: str) -> str | None:
        """Return content hash for delta checking."""
        rows = await self._query_dicts(
            "SELECT content_hash FROM pages WHERE path = ?", [path]
        )
        if not rows:
            return None
        return str(rows[0]["content_hash"])

    async def upsert_page(self, rel_path: str) -> None:
        """Upsert a single page metadata into the index."""
        async with self._lock:
            self._get_connection()
        try:
            parsed = await asyncio.to_thread(self._read_and_parse_file, rel_path)

            query = """
            INSERT OR REPLACE INTO pages (
                path, name, subdir, summary, tags, created, updated,
                word_count, content_hash, frontmatter, wikilinks,
                is_operational, mtime, yaml_error, body
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = [
                parsed["path"],
                parsed["name"],
                parsed["subdir"],
                parsed["summary"],
                parsed["tags"],
                parsed["created"],
                parsed["updated"],
                parsed["word_count"],
                parsed["content_hash"],
                parsed["frontmatter"],
                parsed["wikilinks"],
                parsed["is_operational"],
                parsed["mtime"],
                parsed["yaml_error"],
                parsed["body"],
            ]
            await self._execute(query, params)

            # Rebuild link graph for this single page
            # 1. Delete outbound links from this page
            source_slug = parsed["name"].lower().replace(" ", "-")
            await self._execute(
                "DELETE FROM link_graph WHERE source_slug = ?", [source_slug]
            )

            # 2. Insert new outbound links
            if not parsed["is_operational"]:
                links = json.loads(parsed["wikilinks"])
                insert_params = []
                for target_slug, weight in links:
                    target_slug_norm = target_slug.lower().replace(" ", "-")
                    insert_params.append((source_slug, target_slug_norm, weight))

                if insert_params:
                    async with self._lock:

                        def _bulk_insert() -> None:
                            conn = self._get_connection()
                            deduped: dict[tuple[str, str], float] = {}
                            for s, t, w in insert_params:
                                deduped[(s, t)] = max(deduped.get((s, t), 0.0), w)

                            stmt_params = [(s, t, w) for (s, t), w in deduped.items()]
                            conn.executemany(
                                "INSERT OR REPLACE INTO link_graph (source_slug, target_slug, weight) VALUES (?, ?, ?)",
                                stmt_params,
                            )

                        await asyncio.to_thread(_bulk_insert)
        except Exception as e:
            logger.error(f"Error upserting page {rel_path}: {e}")

    async def remove_page(self, rel_path: str) -> None:
        """Remove a single page from the index."""
        async with self._lock:
            self._get_connection()
        # Find page slug first
        rows = await self._query_dicts(
            "SELECT name FROM pages WHERE path = ?", [rel_path]
        )
        if not rows:
            return

        name = rows[0]["name"]
        source_slug = name.lower().replace(" ", "-")

        await self._execute("DELETE FROM pages WHERE path = ?", [rel_path])
        await self._execute(
            "DELETE FROM link_graph WHERE source_slug = ?", [source_slug]
        )

    async def _load_tag_taxonomy(self) -> dict[str, str]:
        """Parse tag-taxonomy.md and return {non_preferred: preferred} mapping."""
        taxonomy_path = self.vault_path / "wiki" / "concepts" / "tag-taxonomy.md"
        if not taxonomy_path.exists():
            return {}
        try:
            content = await asyncio.to_thread(taxonomy_path.read_text, encoding="utf-8")
            # Parse table rows: | `non-preferred` | `preferred` |
            row_re = re.compile(r"\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|")
            return {
                m.group(1): m.group(2)
                for m in row_re.finditer(content)
                if m.group(1) != "Tag used"  # skip header row
            }
        except Exception as e:
            logger.warning(f"Error loading tag taxonomy: {e}")
            return {}

    async def lint(self) -> dict[str, Any]:
        """Run all lint checks entirely from the index (plus tag-taxonomy file)."""
        # Fetch totals
        total_rows = await self._query_dicts("SELECT COUNT(*) as count FROM pages")
        total_pages = total_rows[0]["count"] if total_rows else 0

        know_rows = await self._query_dicts(
            "SELECT COUNT(*) as count FROM pages WHERE is_operational = FALSE"
        )
        knowledge_pages = know_rows[0]["count"] if know_rows else 0

        op_rows = await self._query_dicts(
            "SELECT COUNT(*) as count FROM pages WHERE is_operational = TRUE"
        )
        operational_excluded = op_rows[0]["count"] if op_rows else 0

        # 1. Orphans: Knowledge pages with no inbound links in link_graph
        orphan_rows = await self._query_dicts("""
            SELECT name FROM pages
            WHERE is_operational = FALSE
              AND LOWER(REPLACE(name, ' ', '-')) NOT IN (SELECT DISTINCT target_slug FROM link_graph)
        """)
        orphans = [r["name"] for r in orphan_rows]

        # 2. Broken links: Outbound links in link_graph pointing to non-existent knowledge pages
        broken_rows = await self._query_dicts("""
            SELECT lg.source_slug, lg.target_slug, p.path AS source_path
            FROM link_graph lg
            JOIN pages p ON p.is_operational = FALSE AND LOWER(REPLACE(p.name, ' ', '-')) = lg.source_slug
            LEFT JOIN pages pt ON pt.is_operational = FALSE AND LOWER(REPLACE(pt.name, ' ', '-')) = lg.target_slug
            WHERE pt.name IS NULL AND lg.target_slug NOT IN ('index', 'log')
        """)
        broken_links = [
            {"source": r["source_path"], "target": r["target_slug"]}
            for r in broken_rows
        ]

        # 3. Missing frontmatter
        missing_fm_rows = await self._query_dicts(
            "SELECT path FROM pages WHERE is_operational = FALSE AND (frontmatter = '{}' OR frontmatter IS NULL)"
        )
        missing_frontmatter = [r["path"] for r in missing_fm_rows]

        # 4. Invalid frontmatter
        invalid_fm_rows = await self._query_dicts(
            "SELECT path, yaml_error FROM pages WHERE is_operational = FALSE AND yaml_error IS NOT NULL"
        )
        invalid_frontmatter = [
            {"page": r["path"], "error": r["yaml_error"]} for r in invalid_fm_rows
        ]

        # 5. Non-reciprocal links
        non_recip_rows = await self._query_dicts("""
            SELECT lg1.source_slug, lg1.target_slug, p1.path AS source_path, p2.name AS target_name
            FROM link_graph lg1
            JOIN pages p1 ON p1.is_operational = FALSE AND LOWER(REPLACE(p1.name, ' ', '-')) = lg1.source_slug
            JOIN pages p2 ON p2.is_operational = FALSE AND LOWER(REPLACE(p2.name, ' ', '-')) = lg1.target_slug
            WHERE (p1.path LIKE 'wiki/concepts/%' OR p1.path LIKE 'wiki/entities/%')
              AND (p2.path LIKE 'wiki/concepts/%' OR p2.path LIKE 'wiki/entities/%')
              AND NOT EXISTS (
                  SELECT 1 FROM link_graph lg2
                  WHERE lg2.source_slug = lg1.target_slug
                    AND lg2.target_slug = lg1.source_slug
              )
        """)
        non_reciprocal = [
            {
                "source": r["source_path"]
                .replace("\\", "/")
                .rsplit("/", 1)[-1]
                .removesuffix(".md"),
                "missing_back_link": r["target_name"],
            }
            for r in non_recip_rows
        ]

        # 6. Tag taxonomy check
        non_preferred_tags = []
        use_map = await self._load_tag_taxonomy()
        if use_map:
            tag_rows = await self._query_dicts(
                "SELECT path, tags FROM pages WHERE is_operational = FALSE"
            )
            for row in tag_rows:
                try:
                    tags = json.loads(row["tags"])
                    for tag in tags:
                        tag = tag.strip()
                        if tag in use_map:
                            non_preferred_tags.append(
                                {
                                    "page": row["path"],
                                    "tag": tag,
                                    "use_instead": use_map[tag],
                                }
                            )
                except Exception:
                    continue

        return {
            "total_pages": total_pages,
            "knowledge_pages": knowledge_pages,
            "operational_excluded": operational_excluded,
            "orphan_pages": orphans,
            "broken_links": broken_links,
            "missing_frontmatter": missing_frontmatter,
            "invalid_frontmatter": invalid_frontmatter,
            "non_reciprocal_links": non_reciprocal,
            "non_preferred_tags": non_preferred_tags,
        }
