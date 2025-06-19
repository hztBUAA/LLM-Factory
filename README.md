# LLM Factory

A unified LLM factory system that provides a consistent interface for multiple AI model providers.

## Supported Providers

- **OpenAI** - Azure OpenAI implementation (GPT-4o, GPT-4, etc.)
- **Qwen** - Alibaba's Qwen models
- **DeepSeek** - DeepSeek models  
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
- Simple singleton factory pattern with environment variable support

## Usage

### Environment Variables Setup

Create a `.env` file with your configuration:

```bash
# Azure OpenAI Configuration
OPENAI_API_KEYS="key1,key2,key3"  # Multiple API keys for load balancing
OPENAI_API_BASES="https://account1.openai.azure.com,https://account2.openai.azure.com,https://account3.openai.azure.com"
OPENAI_MODEL="gpt-4o"  # or other models like gpt-4
OPENAI_API_VERSION="2024-02-15-preview"

# Optional: Other Provider Configurations
QWEN_API_KEYS="key1,key2"
QWEN_MODEL="qwen-turbo"
QWEN_API_BASE="your-qwen-api-base"

DEEPSEEK_API_KEYS="key1,key2"
DEEPSEEK_MODEL="deepseek-chat"
DEEPSEEK_API_BASE="your-deepseek-api-base"

CLAUDE_ACCESS_KEYS="key1,key2"
CLAUDE_MODEL="anthropic.claude-3-5-sonnet-20241022-v2:0"
CLAUDE_SECRET_KEY="your-claude-secret-key"
CLAUDE_REGION="us-east-1"

GEMINI_API_KEYS="key1,key2"
GEMINI_MODEL="gemini-2.0-flash-exp"
GEMINI_PROJECT_ID="your-project-id"
GEMINI_REGION="your-region"
```

### Basic Usage

```python
from llm_factory import LLMFactory

# Create factory instance (automatically loads from .env)
factory = LLMFactory.create()

# Or specify a different env file
factory = LLMFactory.create(".env.local")

# Synchronous call
response = factory.chat("Hello, world!")
print(response.choices[0]["message"]["content"])

# Asynchronous call
response = await factory.chat_async("Hello, world!")
```

### Load Balancing Example

The factory automatically handles load balancing across all configured API keys:

```python
# Load balancing is automatic based on your environment variables
factory = LLMFactory.create()

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

### FastAPI Usage

```bash
# First, ensure your .env file is properly configured

# Start the API server
python -m llm_factory.api

# Make requests to the API
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 1000
  }'
```

### API Request Format

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false,
  "load_balance_strategy": "round_robin"
}
```
