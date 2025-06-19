# LLM Factory

A unified LLM factory system that provides a consistent interface for multiple AI model providers.

## Supported Providers

- **Qwen** - Alibaba's Qwen models
- **DeepSeek** - DeepSeek models  
- **OpenAI** - Azure OpenAI implementation (GPT-4, GPT-4o, etc.)
- **Claude** - AWS Bedrock implementation
- **Gemini** - Google Gemini models

## Features

- Unified interface with `__call__` and `call_model_async()` methods
- Support for both streaming and non-streaming responses
- Proxy configuration support
- Cost billing and metrics collection
- OpenAI-compatible JSON output, tool calls, and image processing
- Both SDK and FastAPI interfaces
- Extensible factory pattern for easy addition of new providers

## Usage

### SDK Usage

```python
from llm_factory import LLMFactory

# Create factory instance
factory = LLMFactory(
    provider="openai",
    model="gpt-4o",
    api_key="your-api-key",
    proxy_config={"http": "http://proxy:8080"}
)

# Synchronous call
response = factory("Hello, world!")

# Asynchronous call
response = await factory.call_model_async("Hello, world!", stream=True)
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
