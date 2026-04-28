import asyncio
import re
import string
from typing import Any


async def _clean_text(text: str) -> str:
    """Original cleaning logic."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove problematic characters with a simpler approach
    # Keep alphanumeric, spaces, and basic punctuation
    allowed_chars = (
        string.ascii_letters + string.digits + string.whitespace + ".,!?;:-()[]\"'/&"
    )
    text = "".join(char for char in text if char in allowed_chars)

    # Normalize quotes
    text = re.sub(r"[\u201C\u201D]", '"', text)
    text = re.sub(r"[\u2018\u2019]", "'", text)

    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


async def _clean_text_improved(text: str) -> str:
    """Improved cleaning logic."""
    # 1. Remove HTML spans (often LaTeX artifacts)
    text = re.sub(r"</?span[^>]*>", "", text)

    # 2. Remove CID strings
    text = re.sub(r"\(CID:\d+\)", "", text)

    # 3. Remove common LaTeX commands (simplified)
    # Matches \command{...} or \command
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)

    # 4. Remove display math and inline math
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]+\$", "", text)

    # 5. Remove citations like [1], [1, 2], [1-3]
    text = re.sub(r"\[\d+(?:,\s*\d+|-\d+)*\]", "", text)

    # 6. Remove parenthetical citations like (Author, 2020)
    text = re.sub(r"\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)", "", text)

    # Original cleaning logic for whitespace and allowed chars
    text = re.sub(r"\s+", " ", text)

    allowed_chars = (
        string.ascii_letters + string.digits + string.whitespace + ".,!?;:-()[]\"'/&"
    )
    text = "".join(char for char in text if char in allowed_chars)

    text = re.sub(r"[\u201C\u201D]", '"', text)
    text = re.sub(r"[\u2018\u2019]", "'", text)

    text = text.strip()
    # Remove leading/trailing punctuation that might result from stripping
    text = re.sub(r"^[.,!?;:-]+", "", text)

    if text and text[-1] not in ".!?":
        text += "."

    # Final whitespace squeeze
    text = re.sub(r"\s+", " ", text).strip()

    return text


async def test():
    samples = [
        "The model (CID:123) achieved 95% accuracy [1, 2].",
        "We use <span class='math'>E=mc^2</span> for calculations.",
        "As shown in \\textbf{Figure 1}, the results vary.",
        "The energy equation $$ E = h \\nu $$ is fundamental (Einstein, 1905).",
        "Multiple citations [3-5] and nested tags <span>outer <span>inner</span></span>.",
    ]

    print(f"{'Original':<20} | {'Cleaned (Old)':<30} | {'Cleaned (New)':<30}")
    print("-" * 85)
    for s in samples:
        old = await _clean_text(s)
        new = await _clean_text_improved(s)
        print(f"{s[:20]:<20} | {old[:30]:<30} | {new[:30]:<30}")


if __name__ == "__main__":
    asyncio.run(test())
