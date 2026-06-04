"""
Response budget utility for Project Synapse.

Prevents flooding agent context windows by truncating large list-like tool outputs
and optionally writing a full report to a spillover file.
"""

from pathlib import Path

MAX_RESPONSE_CHARS = 4000  # ~1000 tokens


def budget_response(
    header: str,
    items: list[str],
    footer: str = "",
    max_chars: int = MAX_RESPONSE_CHARS,
    spillover_path: Path | None = None,
) -> str:
    """Build a response that fits within a character budget.

    If the full response exceeds max_chars, items are truncated to fit
    and a summary indicator is added. If spillover_path is provided, the
    full itemized content is written to that path on disk first.
    """
    total_non_items_len = len(header) + len(footer) + len("\n") * 2
    if total_non_items_len >= max_chars:
        # Non-item content itself exceeds budget; return a truncated header
        return header[: max_chars - 10] + "\n...(truncated)"

    # If spillover is provided, write the full content to disk
    if spillover_path:
        try:
            spillover_path.parent.mkdir(parents=True, exist_ok=True)
            full_content = header + "\n" + "\n".join(items) + "\n" + footer
            spillover_path.write_text(full_content, encoding="utf-8")
        except Exception:
            # Non-blocking, ignore errors writing spillover
            pass

    # Build the buffered response item-by-item
    result = [header]
    current_len = len(header)
    truncated = False
    items_added = 0

    for item in items:
        # Item length plus separator
        item_len = len(item) + 1
        if current_len + item_len + len(footer) + 50 > max_chars:
            truncated = True
            break
        result.append(item)
        current_len += item_len
        items_added += 1

    if truncated:
        spillover_msg = ""
        if spillover_path:
            spillover_msg = f" Full report saved to: `{spillover_path.name}`"

        remaining = len(items) - items_added
        result.append(f"\n_... {remaining} more items truncated.{spillover_msg}_")

    if footer:
        result.append(footer)

    return "\n".join(result)
