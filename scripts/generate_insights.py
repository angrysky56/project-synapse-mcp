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

Shutdown behavior:
    The script ends with ``os._exit(rc)`` after writing output. Cron-fired
    runs were hanging at ``asyncio.run`` teardown waiting for the Neo4j
    async driver's worker pool to drain — meanwhile the cron's session-level
    wall clock would expire and kill the process anyway. We do our own
    cleanup explicitly and then hard-exit, since this is a one-shot batch
    job and graceful Python shutdown buys us nothing.

Exit codes:
    0  success (insights may be empty if confidence threshold not met)
    1  generation failed (Neo4j / config error)
    2  bad CLI arguments
    3  exceeded --max-runtime soft cap (output may still be partial)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import sys
import threading
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
from synapse_mcp.utils.logging_config import quiet_chatty_loggers  # noqa: E402
from synapse_mcp.zettelkasten.insight_engine import InsightEngine  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
# Mute Neo4j's "index already exists, IF NOT EXISTS worked correctly" info
# stream and similar low-signal output — otherwise the cron log fills with
# pages of notification objects per startup.
quiet_chatty_loggers()
log = logging.getLogger("generate_insights")


# ---------------------------------------------------------------------------
# Watchdog: hard SIGKILL via os._exit if the asyncio path can't be cancelled.
#
# ``_detect_patterns`` runs synchronous numpy/networkx code that doesn't yield
# to the event loop. ``asyncio.wait_for`` can't cancel it mid-call — the
# computation has to return on its own. The watchdog gives us a guaranteed
# upper bound regardless of what the engine is doing.
# ---------------------------------------------------------------------------
def _install_hard_watchdog(seconds: int) -> None:
    """Start a daemon timer that hard-exits the process after ``seconds``.

    Uses SIGALRM on Linux (preferred — interrupts blocking syscalls) and a
    daemon Timer thread elsewhere. The thread variant can't interrupt a
    blocking C extension but is the best we can do off-Linux.
    """
    if seconds <= 0:
        return

    def _bomb() -> None:
        log.error(
            "Hard watchdog fired after %ds — process did not exit cleanly. "
            "Output already on disk; killing now.",
            seconds,
        )
        os._exit(3)

    if hasattr(signal, "SIGALRM"):
        signal.signal(
            signal.SIGALRM,
            lambda *_: _bomb(),
        )
        signal.alarm(seconds)
    else:  # pragma: no cover — Windows / non-POSIX fallback
        t = threading.Timer(seconds, _bomb)
        t.daemon = True
        t.start()


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
    p.add_argument(
        "--max-runtime",
        type=int,
        default=int(os.getenv("INSIGHT_CLI_MAX_RUNTIME", "540")),
        help=(
            "Soft cap on engine.generate_insights() in seconds (default 540 — "
            "leaves headroom under a typical 600s cron wall). Set 0 to "
            "disable. A separate hard SIGALRM fires at max_runtime + 30s as a "
            "safety net for synchronous code that ignores asyncio cancellation."
        ),
    )
    p.add_argument(
        "--no-force-exit",
        action="store_true",
        help=(
            "Don't os._exit() after work completes. Useful for tests; "
            "harmful for cron because asyncio teardown may hang on the "
            "Neo4j driver's worker pool."
        ),
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

    insights: list = []
    try:
        coro = engine.generate_insights(
            topic=args.topic,
            confidence_threshold=args.threshold,
            persist=not args.no_persist,
            write_file=not args.no_file,
        )
        if args.max_runtime and args.max_runtime > 0:
            try:
                insights = await asyncio.wait_for(coro, timeout=args.max_runtime)
            except asyncio.TimeoutError:
                log.error(
                    "engine.generate_insights() exceeded --max-runtime %ds. "
                    "Any insights already persisted to Neo4j survive; "
                    "the file-output step may have been skipped.",
                    args.max_runtime,
                )
                return 3
        else:
            insights = await coro
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error("Insight generation failed: %s", e, exc_info=True)
        return 1
    finally:
        # Best-effort cleanup. Errors here are logged but never override the
        # primary return code — the work was either done or not, and that's
        # what cron cares about.
        with contextlib.suppress(Exception):
            await engine.cleanup()
        with contextlib.suppress(Exception):
            await kg.close()

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

    # 30s grace beyond the soft cap so the soft path has a chance to win.
    if args.max_runtime and args.max_runtime > 0:
        _install_hard_watchdog(args.max_runtime + 30)

    rc = asyncio.run(_run(args))

    # Cron-fired runs were hanging here on asyncio teardown — Neo4j's async
    # driver keeps a worker pool that can take a long time to drain. We're a
    # one-shot batch job; once the file is written and Neo4j connections
    # closed, there's nothing left worth waiting for.
    if not args.no_force_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
