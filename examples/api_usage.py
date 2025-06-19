"""
API usage examples for the LLM Factory.
"""

import asyncio
import httpx


async def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:8000/v1"
    
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/health")
        print("Health check:", response.json())
        
        response = await client.get(f"{base_url}/models")
        print("Available models:", response.json())
        
        chat_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        response = await client.post(f"{base_url}/chat/completions", json=chat_request)
        print("Chat response:", response.json())
        
        chat_request["stream"] = True
        async with client.stream("POST", f"{base_url}/chat/completions", json=chat_request) as response:
            print("Streaming response:")
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        print(data)
        
        response = await client.get(f"{base_url}/providers/status")
        print("Provider status:", response.json())


if __name__ == "__main__":
    print("Testing LLM Factory API...")
    asyncio.run(test_api())
