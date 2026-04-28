
import asyncio
import os
from synapse_mcp.data_pipeline.semantic_integrator import SemanticIntegrator
from synapse_mcp.utils.logging_config import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)

async def test_ingestion():
    integrator = SemanticIntegrator()
    await integrator.initialize()
    
    with open("data/noisy_sample.txt", "r") as f:
        text = f.read()
    
    cleaned = integrator._clean_text(text)
    print("--- CLEANED TEXT ---")
    print(cleaned)
    print("--------------------")
    
    # Process it
    # result = await integrator.process_text(cleaned, source="noisy_sample.txt")
    # print(f"Ingested {len(result['facts'])} facts and {len(result['entities'])} entities")
    
    # await integrator.close()

if __name__ == "__main__":
    asyncio.run(test_ingestion())
