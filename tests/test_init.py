import asyncio
import logging

from src.synapse_mcp.server import SynapseServer

logging.basicConfig(level=logging.DEBUG)


async def main():
    server = SynapseServer()
    print("Initializing server...")
    await server.initialize()
    print("Done!")


asyncio.run(main())
