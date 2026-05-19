#!/usr/bin/env python3
"""
synapse-ingest — point Synapse at any folder and have it build a knowledge
graph over the text files inside.

Replaces the original ``reingest_all.py`` which was hardcoded to wipe
Neo4j and re-process ``Clippings/`` end-to-end. This generalised version:

* Accepts any folder via ``--path`` (Obsidian vault, notes folder, repo, etc.)
* Defaults to *non-destructive* — append-mode unless ``--wipe`` is passed
* Skips files whose contents haven't changed since the last ingest, tracked
  in a small SQLite ledger at ``<path>/.synapse_ingest.db``
* Configurable glob pattern (default ``**/*.md``) so it can scoop up plain
  text, code-comment exports, or anything else markdown-ish
* ``--dry-run`` to see what would be processed without touching the graph

Usage (after ``uv sync``):

    synapse-ingest --path /path/to/vault
    synapse-ingest --path ~/Documents/LLM-WIKI --wipe
    synapse-ingest --path ~/notes --glob "**/*.{md,txt}" --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sqlite3
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

# Load .env from project root if running from a checked-out copy.
_PKG_DIR = Path(__file__).resolve().parents[3]
load_dotenv(_PKG_DIR / ".env")

from synapse_mcp.core.knowledge_graph import KnowledgeGraph  # noqa: E402
from synapse_mcp.data_pipeline.semantic_integrator import (  # noqa: E402
    SemanticIntegrator,
)

# ---------------------------------------------------------------------------
# Defaults & environment
# ---------------------------------------------------------------------------

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "synapse_password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "synapse")

# Per-folder ledger filename. Lives at the root of each ingested folder so
# different vaults track their own history independently.
LEDGER_FILENAME = ".synapse_ingest.db"

# Files smaller than this are skipped — they're almost always empty stubs
# or single-line redirects that produce nothing useful but cost an LLM call.
MIN_FILE_BYTES = 50


# ---------------------------------------------------------------------------
# Ledger: SQLite file-hash tracking
# ---------------------------------------------------------------------------


class IngestLedger:
    """SQLite-backed record of which files have been ingested at which content
    hash. Used to skip unchanged files on re-ingest, so pointing the CLI at
    the same folder repeatedly is cheap.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ingested (
                relpath TEXT PRIMARY KEY,
                sha256  TEXT NOT NULL,
                bytes   INTEGER NOT NULL,
                ts      REAL NOT NULL
            )
            """)
        self._conn.commit()

    def get(self, relpath: str) -> str | None:
        """Return the recorded hash for ``relpath``, or None if unseen."""
        row = self._conn.execute(
            "SELECT sha256 FROM ingested WHERE relpath = ?", (relpath,)
        ).fetchone()
        return row[0] if row else None

    def record(self, relpath: str, sha256: str, size: int) -> None:
        self._conn.execute(
            """
            INSERT INTO ingested (relpath, sha256, bytes, ts)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(relpath) DO UPDATE SET
                sha256 = excluded.sha256,
                bytes  = excluded.bytes,
                ts     = excluded.ts
            """,
            (relpath, sha256, size, time.time()),
        )
        self._conn.commit()

    def forget_all(self) -> None:
        """Drop every record. Used with ``--wipe`` so next run treats everything
        as new (mirrors the destroyed Neo4j state)."""
        self._conn.execute("DELETE FROM ingested")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Source-type inference
# ---------------------------------------------------------------------------


def detect_type(filepath: Path, root: Path) -> str:
    """Infer source type from folder structure under the ingest root.

    Heuristic: folder names along the path hint at content type. Falls back
    to ``article`` for generic markdown.
    """
    try:
        relparts = filepath.relative_to(root).parts
    except ValueError:
        relparts = filepath.parts

    lower = {p.lower() for p in relparts}
    if "papers" in lower or "research" in lower:
        return "paper"
    if "repositories" in lower or "repos" in lower:
        return "repository"
    if "documentation" in lower or "docs" in lower:
        return "documentation"
    if "clippings" in lower:
        return "article"
    return "article"


# ---------------------------------------------------------------------------
# Per-file ingestion
# ---------------------------------------------------------------------------


async def ingest_file(
    integrator: SemanticIntegrator,
    kg: KnowledgeGraph,
    filepath: Path,
    root: Path,
    idx: int,
    total: int,
    *,
    dry_run: bool,
) -> dict:
    """Run a single file through the Synapse pipeline. Returns a status dict."""
    source_name = filepath.stem
    rel = str(filepath.relative_to(root))
    t0 = time.time()

    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
        # Strip YAML frontmatter so it doesn't pollute the entity extractor.
        if text.startswith("---"):
            end = text.find("---", 3)
            if end >= 0:
                text = text[end + 3 :].strip()

        if len(text) < MIN_FILE_BYTES:
            return {"file": rel, "status": "skipped", "reason": "too short"}

        # In dry-run mode skip both extraction AND storage. Otherwise we'd
        # burn LLM calls on a "preview" run, which defeats the whole point
        # of dry-run for a folder of any size.
        if dry_run:
            elapsed = time.time() - t0
            display = rel if len(rel) <= 55 else "…" + rel[-54:]
            print(
                f"  [{idx:4d}/{total}] {display:55s}  "
                f"{len(text):6d} chars   {elapsed:5.2f}s  (dry)"
            )
            return {
                "file": rel,
                "status": "ok",
                "entities": 0,
                "relationships": 0,
                "facts": 0,
                "seconds": elapsed,
                "dry_run": True,
            }

        processed = await integrator.process_text_with_semantics(
            text,
            source_name,
            {"type": detect_type(filepath, root)},
        )

        await kg.store_processed_data(processed)

        elapsed = time.time() - t0
        n_ent = len(processed.get("entities", []))
        n_rel = len(processed.get("relationships", []))
        n_fact = len(processed.get("facts", []))

        # Truncate the display name only — the rel path is still in the dict
        # for the JSON-y end report.
        display = rel if len(rel) <= 55 else "…" + rel[-54:]
        print(
            f"  [{idx:4d}/{total}] {display:55s}  "
            f"{n_ent:4d} ent  {n_rel:4d} rel  {n_fact:4d} fact  "
            f"{elapsed:5.1f}s"
        )

        return {
            "file": rel,
            "status": "ok",
            "entities": n_ent,
            "relationships": n_rel,
            "facts": n_fact,
            "seconds": elapsed,
        }

    except Exception as e:  # pylint: disable=broad-exception-caught
        elapsed = time.time() - t0
        print(f"  [{idx:4d}/{total}] {rel:55s}  ERROR: {e}")
        return {
            "file": rel,
            "status": "error",
            "error": str(e),
            "seconds": elapsed,
        }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def collect_files(root: Path, glob: str) -> list[Path]:
    """Return matching files, excluding the ledger itself and hidden folders.

    Python's ``Path.rglob`` doesn't support brace expansion, so we accept a
    comma-separated list ("**/*.md,**/*.txt") *or* a single glob and union
    the results — matches what shells normally do with `{a,b}`.
    """
    patterns = [g.strip() for g in glob.split(",") if g.strip()]
    seen: set[Path] = set()
    out: list[Path] = []
    for pat in patterns:
        for p in root.rglob(pat):
            if not p.is_file():
                continue
            # Skip hidden folders (.git, .obsidian, .trash, .synapse_ingest.db)
            if any(part.startswith(".") for part in p.relative_to(root).parts):
                continue
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
    out.sort()
    return out


def filter_unchanged(
    files: Iterable[Path], root: Path, ledger: IngestLedger
) -> tuple[list[Path], int]:
    """Split ``files`` into (to_process, skipped_count) based on ledger state."""
    to_process: list[Path] = []
    skipped = 0
    for f in files:
        rel = str(f.relative_to(root))
        current = file_sha256(f)
        prior = ledger.get(rel)
        if prior == current:
            skipped += 1
            continue
        # Save the new hash now so an interrupted run still records progress
        # on every fully-completed file later (we'll re-record after success).
        to_process.append(f)
    return to_process, skipped


# ---------------------------------------------------------------------------
# Wipe & stats
# ---------------------------------------------------------------------------


async def wipe_graph(driver) -> int:
    async with driver.session(database=NEO4J_DATABASE) as session:
        result = await session.run(
            "MATCH (n) DETACH DELETE n RETURN count(n) AS deleted"
        )
        records = await result.data()
        return records[0]["deleted"] if records else 0


async def read_stats(driver) -> dict:
    stats: dict = {}
    async with driver.session(database=NEO4J_DATABASE) as session:
        result = await session.run(
            "MATCH (e:Entity) RETURN e.type AS type, count(*) AS count "
            "ORDER BY count DESC"
        )
        stats["entity_types"] = await result.data()

        result = await session.run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count "
            "ORDER BY count DESC LIMIT 20"
        )
        stats["rel_types"] = await result.data()

        result = await session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count "
            "ORDER BY count DESC"
        )
        stats["totals"] = await result.data()
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="synapse-ingest",
        description=(
            "Ingest a folder of text files into a Synapse knowledge graph. "
            "Idempotent by content hash; safe to re-run."
        ),
    )
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Folder to ingest (Obsidian vault, notes directory, etc.)",
    )
    parser.add_argument(
        "--glob",
        default="**/*.md",
        help="Glob pattern (default: %(default)s). Comma-separated for "
        "multiple, e.g. '**/*.md,**/*.txt'.",
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Destroy the Neo4j graph and ledger before ingesting. Default "
        "is append-mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be ingested but don't call the pipeline "
        "or touch Neo4j.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Ignore the ledger and re-run every matching file. Use this "
        "after pipeline changes when you want updates without --wipe.",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip the per-type stats summary at the end.",
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    root = args.path.expanduser().resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    print("Synapse ingest")
    print(f"  Path:      {root}")
    print(f"  Glob:      {args.glob}")
    print(f"  Wipe:      {args.wipe}")
    print(f"  Dry run:   {args.dry_run}")
    print(f"  Reprocess: {args.reprocess}")
    print()

    files_all = collect_files(root, args.glob)
    if not files_all:
        print(f"No files match {args.glob} under {root}")
        return 0
    print(f"Discovered {len(files_all)} candidate files")

    ledger = IngestLedger(root / LEDGER_FILENAME)

    # Connect to Neo4j up-front so we fail fast on connectivity errors
    # rather than after the first file finishes its (slow) LLM extraction.
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        await driver.verify_connectivity()
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"error: cannot connect to Neo4j at {NEO4J_URI}: {e}", file=sys.stderr)
        ledger.close()
        return 3
    print(f"Connected to Neo4j at {NEO4J_URI}, database: {NEO4J_DATABASE}")

    integrator = SemanticIntegrator()
    await integrator.initialize()
    kg = KnowledgeGraph()
    await kg.connect()

    try:
        # Wipe step
        if args.wipe and not args.dry_run:
            print("\nWiping existing graph…")
            deleted = await wipe_graph(driver)
            print(f"  Deleted {deleted:,} nodes")
            ledger.forget_all()
            print("  Cleared ingest ledger")
        elif args.wipe and args.dry_run:
            print("\n(dry run — would wipe graph and ledger)")

        # Decide what to actually process
        if args.reprocess or args.wipe:
            files_to_process = files_all
            skipped_unchanged = 0
        else:
            files_to_process, skipped_unchanged = filter_unchanged(
                files_all, root, ledger
            )
        if skipped_unchanged:
            print(
                f"Skipping {skipped_unchanged} unchanged files (use --reprocess to override)"
            )

        if not files_to_process:
            print("Nothing to do.")
            return 0

        print(f"\nProcessing {len(files_to_process)} file(s)…")
        print("-" * 80)

        results: list[dict] = []
        t_total = time.time()
        for idx, filepath in enumerate(files_to_process, 1):
            result = await ingest_file(
                integrator,
                kg,
                filepath,
                root,
                idx,
                len(files_to_process),
                dry_run=args.dry_run,
            )
            results.append(result)

            # Record in ledger only on success — failures will be retried next run
            if result["status"] == "ok" and not args.dry_run:
                rel = str(filepath.relative_to(root))
                ledger.record(rel, file_sha256(filepath), filepath.stat().st_size)
        t_total = time.time() - t_total

        # Summary
        ok = [r for r in results if r["status"] == "ok"]
        errors = [r for r in results if r["status"] == "error"]
        skipped = [r for r in results if r["status"] == "skipped"]
        total_ent = sum(r.get("entities", 0) for r in ok)
        total_rel = sum(r.get("relationships", 0) for r in ok)
        total_fact = sum(r.get("facts", 0) for r in ok)

        print(f"\n{'=' * 60}")
        print("INGEST COMPLETE")
        print(f"{'=' * 60}")
        print(
            f"Files: {len(ok)} ok, {len(errors)} errors, "
            f"{len(skipped)} skipped (this run), "
            f"{skipped_unchanged} unchanged (cached)"
        )
        print(f"Total entities:      {total_ent:,}")
        print(f"Total relationships: {total_rel:,}")
        print(f"Total facts:         {total_fact:,}")
        if files_to_process:
            print(f"Total time:          {t_total:.1f}s ({t_total / 60:.1f}m)")
            print(
                f"Avg per file:        {t_total / max(len(files_to_process), 1):.1f}s"
            )

        # Graph stats (post-ingest)
        if not args.dry_run and not args.no_stats:
            print("\n--- Neo4j Graph Statistics ---")
            stats = await read_stats(driver)
            print("\nNode totals:")
            for row in stats["totals"]:
                print(f"  {row['label']:24s} {row['count']:,}")
            print("\nEntity type distribution:")
            for row in stats["entity_types"]:
                print(f"  {row['type']:24s} {row['count']:,}")
            print("\nRelationship type distribution:")
            for row in stats["rel_types"]:
                rtype = row["type"] or "(unnamed)"
                print(f"  {rtype:24s} {row['count']:,}")

        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"  {e['file']}: {e.get('error', 'unknown')}")

        return 1 if errors else 0
    finally:
        ledger.close()
        await kg.close()
        await driver.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
