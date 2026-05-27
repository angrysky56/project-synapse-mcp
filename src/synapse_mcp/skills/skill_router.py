"""
GBrain RESOLVER-style skill / intent router for Project Synapse MCP.

Maps natural-language trigger phrases to tool methods using a hard-coded
decision table (regex anchors + keyword sets).  This is the same pattern
GBrain uses: fast, predictable, no embedding / ANN lookup required.

Usage
-----
    from synapse_mcp.skills.skill_router import SkillRouter

    router = SkillRouter()
    route = router.route("tell me about alice and her work at Acme")
    print(route)   # → {"tool": "query_knowledge", "confidence": 0.95, "reason": "query phrase"}

A "route" is a dict with at minimum a ``tool`` key (the MCP tool name to call)
and a ``confidence`` float in [0, 1].  Confidence < 0.5 means "no confident match"
— callers should fall back to a general-purpose handler.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# Route result
# ----------------------------------------------------------------------


@dataclass
class Route:
    """Result of a single route lookup."""

    tool: str  # MCP tool name
    confidence: float  # 0.0–1.0
    reason: str = ""  # human-readable match reason
    params: dict[str, str] = field(default_factory=dict)  # extracted query params


# ----------------------------------------------------------------------
# Trigger entry
# ----------------------------------------------------------------------


@dataclass
class TriggerEntry:
    """One row in the routing table."""

    # Anchor regex — must match the beginning of the normalised message.
    # None = always eligible (catch-all row).
    anchor_re: re.Pattern | None
    # Collection of keywords — message must contain ALL of them (intersection).
    # Empty set = no keyword filter.
    keywords: frozenset[str]
    # Collection of stop-phrases — message must NOT contain ANY of them.
    # Only used when required_phrases is empty.
    stop_phrases: frozenset[str]
    # Collection of required-phrases — message MUST contain AT LEAST ONE.
    # Empty set = no requirement (keyword/stop-phrase logic applies instead).
    required_phrases: frozenset[str]
    # Scoring hint: which GBrain pattern does this implement.
    pattern: str
    tool: str
    confidence: float
    reason: str
    # Optional post-hook: (message_text, route) → Route
    # Allows dynamic param extraction after the static match.
    transform: Callable[[str, Route], Route] | None = None


# ----------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------


class SkillRouter:
    """
    Hard-coded decision table for intent routing.

    The table is ordered — first matching entry wins (no score aggregation).
    Anchor regexes are checked first (fast), then keyword intersection,
    then stop-phrase exclusion.
    """

    # ------------------------------------------------------------------
    # Routing table  (no catch-all row — unmatched messages get 0.40 fallback)
    # ------------------------------------------------------------------

    TABLE: list[TriggerEntry] = [
        # ---------- Temporal / episodic memory ----------
        TriggerEntry(
            anchor_re=re.compile(
                r"^(?:remember|record|log|note that|snapshot)\b", re.I
            ),
            keywords=frozenset(),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="brain-ops",
            tool="synapse_remember",
            confidence=0.95,
            reason="remember/record/log/note that/snapshot",
        ),
        TriggerEntry(
            anchor_re=re.compile(
                r"^(?:recall|what do we (?:have|know) about|look up|find.*fact)\b", re.I
            ),
            keywords=frozenset({"remembered", "fact", "recorded", "log entry"}),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="brain-ops",
            tool="synapse_recall",
            confidence=0.92,
            reason="recall/fact lookup",
        ),
        TriggerEntry(
            anchor_re=re.compile(
                r"^(?:timeline|history|chronology|what happened|events? around)\b", re.I
            ),
            keywords=frozenset(),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="brain-ops",
            tool="synapse_timeline",
            confidence=0.93,
            reason="timeline/history/chronology",
        ),
        # ---------- Causal inference ----------
        TriggerEntry(
            anchor_re=re.compile(
                r"^(?:what caused|cause of|caused by|root cause|why did)\b", re.I
            ),
            keywords=frozenset(),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="brain-ops",
            tool="synapse_causal_window",
            confidence=0.93,
            reason="causal query",
        ),
        TriggerEntry(
            anchor_re=re.compile(
                r"^(?:invalidate|retract|correct|undo|invalidat)\b", re.I
            ),
            keywords=frozenset(),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="brain-ops",
            tool="synapse_invalidate",
            confidence=0.93,
            reason="invalidate/retract",
        ),
        # ---------- Graph queries ----------
        TriggerEntry(
            anchor_re=re.compile(
                r"^(?:who (?:knows|is connected|relates)|relationship (?:between|of)|"
                r"connections? (?:between|of)|graph (?:query|explore)|"
                r"explore (?:the )?connections|linked to)",
                re.I,
            ),
            keywords=frozenset(),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="brain-ops",
            tool="explore_connections",
            confidence=0.95,
            reason="graph exploration phrase",
        ),
        # ---------- Knowledge query ----------
        TriggerEntry(
            anchor_re=re.compile(
                r"^(?:what do we know about|tell me about|who is|who was|"
                r"background on|notes on|search for|find (?:me )?(?:info|information))",
                re.I,
            ),
            keywords=frozenset(),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="query",
            tool="query_knowledge",
            confidence=0.95,
            reason="query phrase",
        ),
        # ---------- Entity enrichment ----------
        TriggerEntry(
            anchor_re=re.compile(
                r"^(?:enrich|expand|fill in|improve)(?:\s+\w+)?\s+(?:page|entity|note)",
                re.I,
            ),
            keywords=frozenset(),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="enrich",
            tool="query_knowledge",
            confidence=0.85,
            reason="enrich entity phrase",
        ),
        # ---------- Wiki operations ----------
        TriggerEntry(
            anchor_re=re.compile(r"^(?:wiki|page|write|create|update|edit)\b", re.I),
            keywords=frozenset(),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="wiki-ops",
            tool="wiki_write_page",
            confidence=0.88,
            reason="wiki write/edit phrase",
            transform=lambda msg, route: (
                route
                if any(
                    k in msg.lower()
                    for k in ("wiki", "write page", "create page", "new page", "wrote")
                )
                else Route(
                    tool="query_knowledge",
                    confidence=0.40,
                    reason="weak wiki signal — fallback",
                )
            ),
        ),
        TriggerEntry(
            anchor_re=re.compile(
                r"^(?:fetch|fetch url|ingest url|save url|grab url|download url)\b",
                re.I,
            ),
            keywords=frozenset(),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="wiki-ops",
            tool="wiki_fetch_url",
            confidence=0.92,
            reason="fetch URL phrase",
        ),
        # Explicit "ingest raw/source/file " → wiki_ingest_raw
        # required_phrases ensures we only fire on actual file-ingest commands
        TriggerEntry(
            anchor_re=re.compile(r"^(?:ingest|process|import)\b", re.I),
            keywords=frozenset(),
            stop_phrases=frozenset(),
            required_phrases=frozenset({"raw", "source", "file"}),
            pattern="wiki-ops",
            tool="wiki_ingest_raw",
            confidence=0.88,
            reason="ingest raw source phrase",
        ),
        TriggerEntry(
            anchor_re=re.compile(r"^(?:cluster|group|organize|topic map)\b", re.I),
            keywords=frozenset({"wiki", "pages", "cluster"}),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="wiki-ops",
            tool="wiki_cluster_pages",
            confidence=0.88,
            reason="wiki cluster phrase",
        ),
        TriggerEntry(
            anchor_re=re.compile(
                r"^(?:hubs?|authorities|important pages|key pages)\b", re.I
            ),
            keywords=frozenset(),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="wiki-ops",
            tool="wiki_hits_analysis",
            confidence=0.88,
            reason="HITS analysis phrase",
        ),
        # ---------- Lint / health ----------
        TriggerEntry(
            anchor_re=re.compile(
                r"^(?:lint|health check|broken links|orphan(?:ed)? pages)\b", re.I
            ),
            keywords=frozenset(),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="maintain",
            tool="wiki_lint",
            confidence=0.92,
            reason="wiki lint phrase",
        ),
        # ---------- Ingest pipeline ----------
        # Generic ingest — fires when anchor matches but no raw/source/file signals present
        # (stop-phrases prevent this firing on literal "ingest raw foo.txt" commands;
        # those jump to the wiki-specific row above)
        TriggerEntry(
            anchor_re=re.compile(r"^(?:ingest|process|import|extract)\b", re.I),
            keywords=frozenset(),
            stop_phrases=frozenset({"raw", "source", "file"}),
            required_phrases=frozenset(),
            pattern="ingest",
            tool="ingest_text",
            confidence=0.88,
            reason="generic ingest phrase",
        ),
        # ---------- Insights ----------
        TriggerEntry(
            anchor_re=re.compile(
                r"^(?:synthesise|synthesize|concept synthesis|trace.*idea|intellectual map)\b",
                re.I,
            ),
            keywords=frozenset({"insights"}),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="insights",
            tool="generate_insights",
            confidence=0.88,
            reason="insight generation phrase",
        ),
        TriggerEntry(
            anchor_re=re.compile(r"^(?:autonomous|auto|turbo)\b", re.I),
            keywords=frozenset({"insight", "generate"}),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="insights",
            tool="generate_insights",
            confidence=0.90,
            reason="autonomous insight phrase",
        ),
        TriggerEntry(
            anchor_re=re.compile(r"^(?:generate|find|discover|surface)\b", re.I),
            keywords=frozenset({"insights"}),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="insights",
            tool="generate_insights",
            confidence=0.88,
            reason="generate insights",
        ),
        # ---------- Index ----------
        TriggerEntry(
            anchor_re=re.compile(
                r"^(?:reindex|re-?index|rebuild index|update index)\b", re.I
            ),
            keywords=frozenset(),
            stop_phrases=frozenset(),
            required_phrases=frozenset(),
            pattern="wiki-ops",
            tool="wiki_update_index",
            confidence=0.92,
            reason="reindex phrase",
        ),
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, message: str) -> Route:
        """
        Match ``message`` against the routing table and return the best Route.

        Matching order:
          1. anchor regex (must match message start after normalisation)
          2. keyword intersection (ALL keywords present)
          3. stop-phrase exclusion (NO stop phrase present)
          4. highest confidence wins (among remaining candidates)

        Parameters
        ----------
        message:
            Raw user message.  Stripped, lowercased, and excess whitespace
            removed before matching.

        Returns
        -------
        Route
            With ``tool`` and ``confidence``.  Confidence = 0.0 indicates
            "no match" (caller should fall back to a general handler).
        """
        normalized = self._normalize(message)

        for entry in self.TABLE:
            # Anchor regex
            if entry.anchor_re is not None:
                if not entry.anchor_re.search(message):
                    continue

            # Required phrases — at least one must be present (if specified)
            if entry.required_phrases:
                if not any(rp in message.lower() for rp in entry.required_phrases):
                    continue

            # Keyword intersection (only when required_phrases is empty)
            if not entry.required_phrases and entry.keywords:
                if not entry.keywords.issubset(normalized.split()):
                    continue

            # Stop phrases (only when required_phrases is empty)
            if not entry.required_phrases and entry.stop_phrases:
                if any(sp.lower() in normalized for sp in entry.stop_phrases):
                    continue

            route = Route(
                tool=entry.tool,
                confidence=entry.confidence,
                reason=entry.reason,
            )

            # Dynamic transform hook
            if entry.transform is not None:
                route = entry.transform(message, route)

            return route

        # No match — return a low-confidence generic query fallback
        return Route(
            tool="query_knowledge",
            confidence=0.40,
            reason="no pattern matched — generic fallback",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Lightweight normalisation: strip, lowercase, collapse whitespace."""
        return " ".join(text.strip().lower().split())


# ----------------------------------------------------------------------
# Convenience singleton
# ----------------------------------------------------------------------

_router: SkillRouter | None = None


def get_router() -> SkillRouter:
    """Thread-safe lazy singleton for the global router."""
    global _router
    if _router is None:
        _router = SkillRouter()
    return _router


def route_message(message: str) -> Route:
    """Thin wrapper around get_router().route() for one-shot calls."""
    return get_router().route(message)
