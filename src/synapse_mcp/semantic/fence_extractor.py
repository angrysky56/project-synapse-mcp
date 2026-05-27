"""
Fence-based typed facts extractor — deterministic, no LLM calls.

Implements GBrain-style fence parsing for structured fact extraction.
Fences are markdown code blocks with structured fact rows that get parsed
directly into typed edges without any LLM call overhead.

Fence format:
```gbrain-facts
metric: mrr
value: 49.1
unit: percent
period: 2024-Q4
---

event: hired_alice
subject: alice
object: engineering
date: 2026-01-15

---

preference: Ty
predicate: prefers
object: concise responses
```

Supports:
- metric/value/unit/period → quantitative facts (trajectory tracking)
- event kinds: meeting, job_change, location_change, hired, fired, etc.
- preference, commitment, belief, fact kinds (matching GBrain's fact taxonomy)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# Regex patterns for fence parsing
FENCE_START = re.compile(r"^```gbrain-facts\s*$", re.MULTILINE)
FENCE_END = re.compile(r"^```\s*$", re.MULTILINE)

# Row patterns (key: value)
ROW_PATTERN = re.compile(r"^(\w+):\s*(.*)$", re.MULTILINE)

# Fact kinds supported
VALID_KINDS = frozenset(
    {
        "metric",
        "event",
        "preference",
        "commitment",
        "belief",
        "fact",
        "goal",
        "habit",
        "skill",
        "relationship",
        "location",
    }
)

# Event type values
VALID_EVENT_TYPES = frozenset(
    {
        "meeting",
        "job_change",
        "location_change",
        "hired",
        "fired",
        "founded",
        "invested_in",
        "acquired",
        "partnered",
        "published",
        "presented",
        "promoted",
        "demoted",
    }
)

# Valid metric units
METRIC_UNITS = frozenset(
    {
        "USD",
        "EUR",
        "GBP",
        "JPY",  # currency
        "percent",
        "percentage",
        "%",  # percentages
        "people",
        "users",
        "customers",
        "employees",  # counts
        "days",
        "weeks",
        "months",
        "years",  # duration
        "requests",
        "queries",
        "operations",  # volume
        "ms",
        "seconds",
        "minutes",
        "hours",  # time duration
    }
)


@dataclass
class ParsedFact:
    """A single parsed fact from a fence block."""

    kind: str  # metric, event, preference, etc.
    raw: dict[str, str]  # original key-value pairs
    valid_from: date | None = None  # when the fact became true
    valid_until: date | None = None  # when the fact stopped being true
    strikethrough: bool = False  # true if marked with ~~ (forget intent)
    source_markdown_slug: str = ""  # which page this came from
    row_num: int = 0  # line number in fence (for provenance)
    # Extracted fields (populated by _extract_fields)
    subject: str = ""
    predicate: str = ""
    object: str = ""
    value: Any = None
    unit: str = ""
    period: str = ""
    event_type: str = ""
    confidence: float = 1.0  # fence facts are high-confidence by default


@dataclass
class FenceExtractionResult:
    """Result of parsing a full fence block."""

    facts: list[ParsedFact] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)  # non-fatal


def parse_gbrain_facts(
    text: str,
    source_slug: str = "",
) -> FenceExtractionResult:
    """
    Parse gbrain-facts fence blocks from markdown text.

    Args:
        text: Full text content that may contain fence blocks
        source_slug: Slug of the page this text came from (for provenance)

    Returns:
        FenceExtractionResult with parsed facts and any non-fatal parse errors

    This function is intentionally pure: no I/O, no LLM calls.
    """
    result = FenceExtractionResult()

    # Find all fence blocks
    fence_blocks = _extract_fence_blocks(text)
    if not fence_blocks:
        return result

    for block_idx, (block_text, block_start_line) in enumerate(fence_blocks):
        facts_in_block = _parse_fence_block(block_text, source_slug, block_start_line)
        for fact in facts_in_block:
            if fact.raw:  # has at least one valid row
                result.facts.append(fact)
            # else: skip empty/parse-error rows (already logged)

    return result


def _extract_fence_blocks(text: str) -> list[tuple[str, int]]:
    """Extract all gbrain-facts fence blocks with their starting line numbers."""
    blocks = []
    start_match = None
    block_start_line = 0

    lines = text.split("\n")
    for i, line in enumerate(lines):
        if FENCE_START.search(line):
            start_match = i
            block_start_line = i + 1  # 1-indexed, like text editors
            continue
        if start_match is not None and FENCE_END.search(line):
            block_text = "\n".join(lines[start_match + 1 : i])
            blocks.append((block_text, block_start_line))
            start_match = None

    return blocks


def _parse_fence_block(
    block_text: str,
    source_slug: str,
    block_start_line: int,
) -> list[ParsedFact]:
    """Parse a single fence block into ParsedFact rows."""
    facts: list[ParsedFact] = []

    # Split on --- separators (horizontal rules within fence)
    sections = re.split(r"\n---\n", block_text)

    for section_idx, section in enumerate(sections):
        raw_rows: dict[str, str] = (
            {}
        )  # fresh per-section dict — fixes cross-section bleed
        errors: list[str] = []
        strikethrough = False

        section_lines = section.strip().split("\n")
        current_row_num = block_start_line + sum(
            len(s.split("\n")) + 2 for s in sections[:section_idx]  # +2 for ---\n
        )

        for line in section_lines:
            current_row_num += 1
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check for strikethrough (~~text~~) → forget/forget intent
            if line.startswith("~~") and line.endswith("~~"):
                strikethrough = True
                line = line[2:-2]

            # Parse key: value rows
            match = ROW_PATTERN.match(line)
            if match:
                key, value = match.group(1).strip(), match.group(2).strip()
                key_lower = key.lower()

                # Validate kind
                if key_lower == "kind":
                    if value.lower() not in VALID_KINDS:
                        errors.append(
                            f"Line {current_row_num}: unknown kind '{value}', "
                            f"expected one of {sorted(VALID_KINDS)}"
                        )
                        # Still include it — don't reject the whole section
                    raw_rows["kind"] = value.lower()
                elif key_lower in VALID_KINDS:
                    raw_rows[key_lower] = value
                else:
                    raw_rows[key] = value
            else:
                # Non-row lines (comments, blank noise) — skip silently
                pass

        if not raw_rows:
            # Empty section — not an error, just skip
            continue

        # Derive kind: explicit "kind:" field takes priority, then check
        # for a VALID_KIND key used as a section-header (GBrain convention:
        # "metric:" at the start of a section means the whole section is a
        # metric fact, not a key-value pair where "metric" is the key name).
        explicit_kind = raw_rows.get("kind", "").lower()
        if explicit_kind in VALID_KINDS:
            kind = explicit_kind
        else:
            # Section-header form: e.g. the first line is "metric: mrr"
            # which stored raw_rows["metric"] = "mrr" — use that key as kind
            kind = next(
                (k for k in VALID_KINDS if k in raw_rows and raw_rows[k] != ""),
                "fact",
            )
        fact = ParsedFact(
            kind=kind,
            raw=raw_rows,
            strikethrough=strikethrough,
            source_markdown_slug=source_slug,
            row_num=block_start_line + section_idx * 10,  # approximate
        )

        # Extract common fields — use the kind-specific field names, falling
        # back to generic aliases so both forms work:
        #   subject / who / entity
        #   predicate / relation / verb
        #   object  / target / what / that
        fact.subject = (
            raw_rows.get("subject") or raw_rows.get("who") or raw_rows.get("entity", "")
        )
        fact.predicate = (
            raw_rows.get("predicate")
            or raw_rows.get("relation")
            or raw_rows.get("verb", "")
        )
        fact.object = (
            raw_rows.get("object")
            or raw_rows.get("target")
            or raw_rows.get("what")
            or raw_rows.get("that", "")
        )

        # For section-header form where "metric: mrr" stored the kind key's
        # value into raw_rows[kind], use it to fill only EMPTY fields.
        # This way an explicit "subject:" row is not clobbered.
        if kind in raw_rows and kind != "kind":
            val_for_kind = raw_rows[kind]
            if not fact.subject:
                fact.subject = val_for_kind
            elif not fact.predicate:
                fact.predicate = val_for_kind
            elif not fact.object:
                fact.object = val_for_kind

        # Parse kind-specific fields
        if kind == "metric":
            _parse_metric_fields(fact, raw_rows, errors)
        elif kind == "event":
            # Split underscore-suffixed event_type (e.g. "hired_alice" → "hired", "alice")
            # Must happen BEFORE _parse_event_fields because it sets event_type
            raw_event_type = raw_rows.get(
                "event", raw_rows.get("event_type", "")
            ).lower()
            if "_" in raw_event_type:
                parts = raw_event_type.split("_", 1)
                fact.event_type = parts[0]
                if not fact.subject:
                    fact.subject = parts[1]
                # Override predicate to the action verb
                fact.predicate = parts[0]
            else:
                _parse_event_fields(fact, raw_rows, errors)
                # After parsing, split underscore in the event_type
                if "_" in fact.event_type:
                    parts = fact.event_type.split("_", 1)
                    fact.event_type = parts[0]
                    if not fact.subject:
                        fact.subject = parts[1]
                    fact.predicate = parts[0]
        elif kind in ("preference", "commitment", "belief"):
            _parse_assertion_fields(fact, raw_rows)

        # Parse temporal fields
        fact.valid_from = _parse_date(
            raw_rows.get("valid_from", raw_rows.get("since", ""))
        )
        fact.valid_until = _parse_date(
            raw_rows.get("valid_until", raw_rows.get("until", ""))
        )

        # Log any parse errors (non-fatal — row is still emitted)
        for err in errors:
            logger.debug(
                "Fence parse warning in %s row %d: %s",
                source_slug,
                current_row_num,
                err,
            )

        facts.append(fact)

    return facts


def _parse_metric_fields(
    fact: ParsedFact, raw: dict[str, str], errors: list[str]
) -> None:
    """Parse metric-specific fields: value, unit, period."""
    # Try numeric first
    value_str = raw.get("value", raw.get("amount", ""))
    try:
        # Handle "49.1%" → strip %
        value_str = value_str.rstrip("%")
        if "." in value_str:
            fact.value = float(value_str)
        else:
            fact.value = int(value_str)
    except ValueError:
        fact.value = value_str  # Keep as string if not numeric

    fact.unit = raw.get("unit", "").lower().strip()
    fact.period = raw.get("period", raw.get("when", ""))

    # Normalize unit
    if fact.unit == "percentage":
        fact.unit = "percent"
    if fact.unit == "$":
        fact.unit = "USD"

    # Validate unit
    if fact.unit and fact.unit not in METRIC_UNITS and not fact.unit.startswith("$"):
        logger.debug("Unusual metric unit '%s' — accepted anyway", fact.unit)


def _parse_event_fields(
    fact: ParsedFact, raw: dict[str, str], errors: list[str]
) -> None:
    """Parse event-specific fields: event_type, date."""
    event_type = raw.get("event", raw.get("event_type", "")).lower()
    if event_type not in VALID_EVENT_TYPES:
        if event_type:
            errors.append(
                f"Unknown event type '{event_type}', treating as generic event"
            )
    fact.event_type = event_type or "generic"

    # If no subject, try who
    if not fact.subject:
        fact.subject = raw.get("who", "")

    # If no object, try target
    if not fact.object:
        fact.object = raw.get("target", "")


def _parse_assertion_fields(fact: ParsedFact, raw: dict[str, str]) -> None:
    """Parse preference/commitment/belief fields."""
    # subject already extracted; object is the thing preferred/committed-to/believed
    if not fact.object:
        # Fall back to 'what' or 'that'
        fact.object = raw.get("what", raw.get("that", raw.get("claim", "")))


def _parse_date(date_str: str) -> date | None:
    """Parse ISO date strings or return None."""
    if not date_str:
        return None
    date_str = date_str.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Public interface for the semantic pipeline
# ---------------------------------------------------------------------------


def extract_facts_from_text(
    text: str,
    source_slug: str = "",
) -> FenceExtractionResult:
    """
    Main entry point. Extract gbrain-facts fence blocks from any text.

    Call this BEFORE LLM extraction — fence facts are high-confidence
    and don't need model inference. The LLM extractor can then focus
    on unstructured text only.

    Usage:
        result = extract_facts_from_text(page_content, source_slug="people/alice")
        for fact in result.facts:
            if fact.kind == "metric":
                store_trajectory_point(fact.subject, fact.value, fact.unit, fact.period)
    """
    return parse_gbrain_facts(text, source_slug)
