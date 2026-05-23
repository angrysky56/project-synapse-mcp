#!/usr/bin/env python3
"""Standalone insight generator — bypasses the MCP server.

Use this from cron or any long-running job where the MCP client's tool-call
timeout (typically 300s) would kill the run before the Zettelkasten engine
finishes. It instantiates the same engine the MCP tool uses, writes results
to ``INSIGHTS_OUTPUT_DIR`` (default ``./data/insights/``), and persists
each insight to Neo4j.

Cron pattern (read latest after a quiet background run)::

    0 6 * * *  cd /path/to/project-synapse-mcp && \\
        uv run python scripts/generate_insights.py --topic general \\
        >> data/insights/cron.log 2>&1

Then a separate reader (chat agent, daily-briefing script, etc.) picks up
``data/insights/latest.md`` whenever it's convenient — no live session
needed.

Exit codes:
    0  success (insights may be empty if confidence threshold not met)
    1  generation failed (Neo4j / config error)
    2  bad CLI arguments
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Allow running directly from a checkout without `uv pip install -e .`.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv  # noqa: E402  (after sys.path tweak)

load_dotenv(_HERE.parent / ".env", override=False)

from synapse_mcp.core.knowledge_graph import KnowledgeGraph  # noqa: E402
from synapse_mcp.semantic.montague_parser import MontagueParser  # noqa: E402
from synapse_mcp.zettelkasten.insight_engine import InsightEngine  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("generate_insights")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Zettelkasten insights without going through MCP."
    )
    p.add_argument(
        "--topic",
        default=None,
        help="Optional topic to focus on. Omit for general insights.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("INSIGHT_CONFIDENCE_THRESHOLD", "0.8")),
        help="Minimum confidence to keep (default from env or 0.8).",
    )
    p.add_argument(
        "--no-persist",
        action="store_true",
        help="Don't store insights as Zettel nodes in Neo4j.",
    )
    p.add_argument(
        "--no-file",
        action="store_true",
        help="Don't write JSON/Markdown to INSIGHTS_OUTPUT_DIR.",
    )
    p.add_argument(
        "--print",
        dest="print_results",
        action="store_true",
        help="Print a short summary of each insight to stdout.",
    )
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    kg = KnowledgeGraph()
    montague = MontagueParser()
    engine = InsightEngine(knowledge_graph=kg, montague_parser=montague)

    try:
        await kg.connect()
        await montague.initialize()
        await engine.initialize()
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error("Failed to initialize Synapse components: %s", e, exc_info=True)
        return 1

    try:
        insights = await engine.generate_insights(
            topic=args.topic,
            confidence_threshold=args.threshold,
            persist=not args.no_persist,
            write_file=not args.no_file,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error("Insight generation failed: %s", e, exc_info=True)
        return 1
    finally:
        try:
            await kg.close()
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.warning("Error closing knowledge graph: %s", e)

    log.info(
        "Generated %d insight(s) above threshold %.2f (topic=%s)",
        len(insights),
        args.threshold,
        args.topic or "general",
    )

    if args.print_results:
        for i, ins in enumerate(insights, 1):
            title = ins.get("title", "(untitled)")
            conf = ins.get("confidence", 0.0)
            print(f"{i:>2}. [{conf:.2f}] {title}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
