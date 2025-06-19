# LLM Factory

A unified LLM factory system that provides a consistent interface for multiple AI model providers.

## Supported Providers

- **Qwen** - Alibaba's Qwen models
- **DeepSeek** - DeepSeek models  
- **OpenAI** - Azure OpenAI implementation (GPT-4, GPT-4o, etc.)
- **Claude** - AWS Bedrock implementation
- **Gemini** - Google Gemini models

## Features

- Unified interface for multiple LLM providers
- Load balancing across multiple API keys for each provider
- Automatic failover support
- Support for both streaming and non-streaming responses
- Both synchronous and asynchronous APIs
- Multiple load balancing strategies (round-robin, random, first-available)
- OpenAI-compatible JSON output format
- Extensible factory pattern for easy addition of new providers

## Usage

### Basic Usage

```python
from llm_factory import LLMFactory, ModelConfig, ProviderType
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Single provider configuration
config = ModelConfig(
    provider=ProviderType.OPENAI,
    model_name="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_API_BASE"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

factory = LLMFactory(config)

# Synchronous call
response = factory.chat("Hello, world!")
print(response.choices[0]["message"]["content"])

# Asynchronous call
response = await factory.chat_async("Hello, world!")
```

### Load Balancing Usage

```python
# Multiple providers configuration for load balancing
configs = [
    ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="key1",
        api_base="https://account1.openai.azure.com/",
        api_version="2024-02-15-preview",
    ),
    ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="key2",
        api_base="https://account2.openai.azure.com/",
        api_version="2024-02-15-preview",
    ),
]

factory = LLMFactory(configs)

# Use different load balancing strategies
response = await factory.chat_async(
    "Hello!",
    load_balance_strategy="round_robin"  # or "random", "first_available"
)
```

### Streaming Usage

```python
# Streaming responses
async for chunk in factory.stream_async("Write a story"):
    if chunk.choices and chunk.choices[0].get("delta", {}).get("content"):
        print(chunk.choices[0]["delta"]["content"], end="", flush=True)
```

### Environment Variables Setup

Create a `.env` or `.makeenv` file:

```bash
# OpenAI Configuration
OPENAI_API_KEYS="key1,key2,key3"  # Multiple API keys for load balancing
OPENAI_API_BASES="https://account1.openai.azure.com/,https://account2.openai.azure.com/"
OPENAI_MODEL="gpt-4"
OPENAI_API_VERSION="2024-02-15-preview"

# Other provider configurations...
QWEN_API_KEYS="key1,key2"
DEEPSEEK_API_KEYS="key1,key2"
CLAUDE_ACCESS_KEYS="key1,key2"
GEMINI_API_KEYS="key1,key2"
```


### FastAPI Usage

```bash
# Start the API server
python -m llm_factory.api

# Make requests to the API
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```
