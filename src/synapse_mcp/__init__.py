"""
Project Synapse MCP Server

An autonomous knowledge synthesis and inference engine that combines:
- Montague Grammar for precise semantic processing
- Zettelkasten methodology for knowledge synthesis
- Neo4j graph database for knowledge storage
- MCP protocol for LLM integration

This system transforms raw text into interconnected knowledge graphs
and autonomously generates insights through pattern detection.
"""

from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE any submodule import. Several modules (e.g.
# core.knowledge_graph) read env vars at module level, so the .env must be
# in the process environment before they are imported. This makes the
# project .env the single source of truth — no env blocks needed in MCP
# client configs. Existing process env still wins (load_dotenv never
# overrides), so deliberate per-launch overrides keep working.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from .__about__ import __description__, __title__, __version__  # noqa: E402

__all__ = ["__version__", "__title__", "__description__"]
