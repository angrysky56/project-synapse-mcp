#!/usr/bin/env python3
import asyncio
import os
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, "src")
os.chdir("/home/ty/Repositories/ai_workspace/project-synapse-mcp")
load_dotenv(".env", override=False)

from synapse_mcp.core.knowledge_graph import KnowledgeGraph
from synapse_mcp.semantic.montague_parser import MontagueParser
from synapse_mcp.zettelkasten.insight_engine import InsightEngine


async def t():
    log_lines = []

    def log(msg):
        log_lines.append(msg)
        print(msg, flush=True)

    total_start = time.time()
    kg = KnowledgeGraph()
    await kg.connect()
    montague = MontagueParser()
    await montague.initialize()
    engine = InsightEngine(knowledge_graph=kg, montague_parser=montague)
    await engine.initialize()

    t0 = time.time()
    patterns = await engine._detect_communities()
    log(f"Detect: {time.time()-t0:.1f}s, {len(patterns)} communities")

    # Test 10 patterns
    patterns = patterns[:10]

    for i, p in enumerate(patterns):
        t0 = time.time()
        ins = await engine._generate_insight_from_pattern(p)
        elapsed = time.time() - t0
        conf = ins.get("confidence", 0) if ins else 0
        log(f"  {i+1}/10: {elapsed:.1f}s conf={conf:.2f}")

    log(f"Total: {time.time()-total_start:.1f}s")
    await engine.cleanup()
    await kg.close()

    # Write to file
    with open("/tmp/insight_timing_result.txt", "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    asyncio.run(t())
