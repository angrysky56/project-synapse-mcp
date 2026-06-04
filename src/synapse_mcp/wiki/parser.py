"""
Markdown parsing helper functions for Project Synapse wiki pages.

Contains functions for frontmatter parsing, serialization, and link extraction.
"""

import re
from typing import Any


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
            # Clean values
            cleaned_val: Any = val.strip().strip('"').strip("'")
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


def extract_outbound_links(body: str, slug: str) -> list[tuple[str, float]]:
    """Extract outbound wikilinks from body content with weights.

    Links found in a section heading with 'connection' in it carry weight 1.0.
    Other links carry weight 0.5.

    Args:
        body: Markdown body text (without frontmatter).
        slug: The page's own slug, to prevent self-linking.

    Returns:
        List of tuples (target_slug, weight).
    """
    link_re = re.compile(r"\[\[([^\]|#]+)")
    heading_re = re.compile(r"^#{1,3}\s+(.+)", re.MULTILINE)

    # Clean code blocks to avoid false link matches
    code_fence_re = re.compile(r"```.*?```", re.DOTALL)
    inline_code_re = re.compile(r"`[^`]+`")
    body_clean = code_fence_re.sub("", body)
    body_clean = inline_code_re.sub("", body_clean)

    # Split body into sections and tag each span with its heading
    section_weights: list[tuple[int, int, float]] = []  # (start, end, weight)
    headings = list(heading_re.finditer(body_clean))
    for i, m in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body_clean)
        heading_text = m.group(1).strip().lower()
        weight = 1.0 if "connection" in heading_text else 0.5
        section_weights.append((m.start(), end, weight))

    def _weight_for_pos(pos: int, weights: list[tuple[int, int, float]]) -> float:
        for start, end, w in weights:
            if start <= pos < end:
                return w
        return 0.5  # pre-first-heading prose

    seen: dict[str, float] = {}
    for match in link_re.finditer(body_clean):
        target = match.group(1).strip().lower().replace(" ", "-")
        if not target or target == slug.lower().replace(" ", "-"):
            continue
        w = _weight_for_pos(match.start(), section_weights)
        seen[target] = max(seen.get(target, 0.0), w)

    return list(seen.items())
