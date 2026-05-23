import asyncio

from synapse_mcp.data_pipeline.semantic_integrator import SemanticIntegrator
from synapse_mcp.semantic.montague_parser import MontagueParser


async def main():
    p = MontagueParser()
    await p.initialize()
    text = "Amirhossein Kazemnejad is a researcher. He invented Delethink."

    si = SemanticIntegrator(p)
    await si.initialize()
    si_res = await si.process_text_with_semantics(text)
    print("Entities:", [e["id"] for e in si_res["entities"]])
    print("SI Relations:", [(r["source_id"], r["target_id"]) for r in si_res["relationships"]])

asyncio.run(main())
