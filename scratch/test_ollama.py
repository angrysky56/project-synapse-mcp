import asyncio
import aiohttp
import sys

async def main():
    url = "http://localhost:11434/api/embed"
    payload = {"model": "qwen3-embedding:4b", "input": "Hello world"}
    print(f"Querying {url} with model qwen3-embedding:4b...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                print(f"Status: {resp.status}")
                data = await resp.json()
                print("Keys in response:", list(data.keys()))
                if "error" in data:
                    print("Error message:", data["error"])
                elif "embeddings" in data:
                    print("Embeddings length:", len(data["embeddings"]))
                    print("First embedding dimension:", len(data["embeddings"][0]))
                else:
                    print("Full response:", data)
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
