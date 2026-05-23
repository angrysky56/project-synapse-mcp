import asyncio

from synapse_mcp.data_pipeline.semantic_integrator import SemanticIntegrator
from synapse_mcp.semantic.montague_parser import MontagueParser


async def main():
    p = MontagueParser()
    await p.initialize()
    text = "Amirhossein Kazemnejad is a researcher. He invented Delethink."
    res = await p.parse_text(text)
    print("Entities:", [e["text"] for e in res["entities"]])
    print("Relations:", res["relations"])

    si = SemanticIntegrator(p)
    await si.initialize()
    si_res = await si.process_text_with_semantics(text)
    print("SI Relations:", si_res["relationships"])

asyncio.run(main())
