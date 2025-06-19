"""
Basic usage examples for the LLM Factory.
"""

import asyncio
import os
from dotenv import load_dotenv

from llm_factory import LLMFactory, ModelConfig, ProviderType, ChatMessage

load_dotenv()


async def basic_example():
    """Basic usage example."""
    config = ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_API_BASE"),
        api_version=os.getenv("OPENAI_API_VERSION"),
    )
    
    factory = LLMFactory(config)
    
    response = await factory.chat_async("Hello, how are you?")
    print("Response:", response.choices[0]["message"]["content"])
    
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is the capital of France?"),
    ]
    
    response = await factory.chat_async(messages)
    print("Response:", response.choices[0]["message"]["content"])


async def load_balancing_example():
    """Load balancing example with multiple providers."""
    configs = [
        ModelConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            api_version=os.getenv("OPENAI_API_VERSION"),
        ),
        ModelConfig(
            provider=ProviderType.DEEPSEEK,
            model_name="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        ),
    ]
    
    factory = LLMFactory(configs)
    
    for i in range(5):
        response = await factory.chat_async(f"Request {i}: Tell me a joke")
        print(f"Response {i}:", response.choices[0]["message"]["content"][:100])


async def streaming_example():
    """Streaming response example."""
    config = ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_API_BASE"),
        api_version=os.getenv("OPENAI_API_VERSION"),
    )
    
    factory = LLMFactory(config)
    
    print("Streaming response:")
    async for chunk in factory.stream_async("Write a short story about a robot"):
        if chunk.choices and chunk.choices[0].get("delta", {}).get("content"):
            print(chunk.choices[0]["delta"]["content"], end="", flush=True)
    print("\n")


def synchronous_example():
    """Synchronous usage example."""
    config = ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_API_BASE"),
        api_version=os.getenv("OPENAI_API_VERSION"),
    )
    
    factory = LLMFactory(config)
    
    response = factory.chat("What is 2+2?")
    print("Sync Response:", response.choices[0]["message"]["content"])
    
    response = factory("Explain quantum computing in simple terms")
    print("Callable Response:", response.choices[0]["message"]["content"])


if __name__ == "__main__":
    print("=== Basic Example ===")
    asyncio.run(basic_example())
    
    print("\n=== Load Balancing Example ===")
    asyncio.run(load_balancing_example())
    
    print("\n=== Streaming Example ===")
    asyncio.run(streaming_example())
    
    print("\n=== Synchronous Example ===")
    synchronous_example()
