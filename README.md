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
- Flexible configuration support (environment variables, YAML, or programmatic)

## Installation

### Using venv

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
## Windows
venv\Scripts\activate
## Linux/MacOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Using Conda

```bash
# Create new conda environment
conda create -n llm-factory python=3.11

# Activate environment
conda activate llm-factory

# Install dependencies
pip install -r requirements.txt
```

## Configuration and Usage

There are three ways to configure and use LLM Factory:

### 1. Direct Model Configuration

The most straightforward way to use a specific model:

```python
from llm_factory import LLMFactory, ModelConfig, ProviderType

# Configure a single model
config = ModelConfig(
    provider=ProviderType.OPENAI,
    model_name="gpt-4o",
    api_key="your-api-key",
    api_base="your-api-base",
    api_version="2024-02-15-preview"
)

# Create factory with single model
factory = LLMFactory(config)

# Use the factory
response = factory.chat("Hello, world!")
```

### 2. Multi-Model Configuration

For load balancing across multiple models or providers:

```python
# Configure multiple models
configs = [
    ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4o",
        api_key="key1",
        api_base="base1",
        api_version="2024-02-15-preview"
    ),
    ModelConfig(
        provider=ProviderType.QWEN,
        model_name="qwen-turbo",
        api_key="key2",
        api_base="https://dashscope.aliyuncs.com/api/v1"
    ),
    ModelConfig(
        provider=ProviderType.DEEPSEEK,
        model_name="deepseek-chat",
        api_key="key3",
        api_base="your-deepseek-base"
    )
]

# Create factory with multiple models
factory = LLMFactory(configs)

# Use with load balancing
response = factory.chat(
    "Hello!",
    load_balance_strategy="round_robin"  # or "random", "first_available"
)
```

### 3. Configuration File Based

#### Environment Variables (.env)

Create a `.env` file:

```bash
# Azure OpenAI Configuration
OPENAI_API_KEYS="key1,key2,key3"  # Multiple API keys for load balancing
OPENAI_API_BASES="https://account1.openai.azure.com,https://account2.openai.azure.com"
OPENAI_MODEL="gpt-4o"
OPENAI_API_VERSION="2024-02-15-preview"

# Optional: Other Provider Configurations
QWEN_API_KEYS="key1,key2"
QWEN_MODEL="qwen-turbo"
QWEN_API_BASE="your-qwen-api-base"
```

Then use:

```python
# Create factory from .env file
factory = LLMFactory.create()

# Or specify a different env file
factory = LLMFactory.create(env_file=".env.local")
```

#### YAML Configuration

Create a `config.yaml` file:

```yaml
providers:
  - provider: "openai"
    model_name: "gpt-4o"
    api_key: "${OPENAI_API_KEY}"  # Support reading from env vars
    api_base: "${OPENAI_API_BASE}"
    api_version: "2024-02-01"
    max_tokens: 4096
    temperature: 0.7

  - provider: "qwen"
    model_name: "qwen-turbo"
    api_key: "${QWEN_API_KEY}"
    api_base: "https://dashscope.aliyuncs.com/api/v1"
    max_tokens: 2000
```

Then use:

```python
# Create factory from YAML config
factory = LLMFactory.create_from_config("config.yaml")
```

## Synchronous vs Asynchronous Usage

The factory supports both synchronous and asynchronous usage:

1. In Synchronous Environments:
   ```python
   # For scripts, command-line tools, etc.
   response = factory.chat("Hello, world!")
   ```

2. In Asynchronous Environments:
   ```python
   # For FastAPI, aiohttp, etc.
   async def chat():
       response = await factory.chat_async("Hello, world!")
   ```

3. Streaming Support:
   ```python
   # Stream responses
   async for chunk in factory.stream_async("Write a story"):
       if chunk.choices and chunk.choices[0].get("delta", {}).get("content"):
           print(chunk.choices[0]["delta"]["content"], end="", flush=True)
   ```

## API Integration

The LLM Factory provides an OpenAI-compatible API interface, allowing you to use various non-OpenAI models (like Qwen, DeepSeek, Claude, etc.) in applications that are designed for OpenAI's API.

### Starting the API Server

Simply run:
```bash
# Start the API server (default: http://localhost:8000)
python main.py
```

The server will automatically load configurations from your environment variables or config files.

### Using as OpenAI API Alternative

Once the server is running, you can use it as a drop-in replacement for OpenAI's API in your applications:

1. For OpenAI's official Python client:
```python
from openai import OpenAI

client = OpenAI(
    api_key="any-key",  # Can be any string as we're using local server
    base_url="http://localhost:8000/v1"  # Point to your local LLM Factory server
)

# Use it just like official OpenAI client
response = client.chat.completions.create(
    model="gpt-4o",  # Or any model configured in your LLM Factory
    messages=[{"role": "user", "content": "Hello!"}]
)
```

2. For applications using OpenAI API:
   - Replace the API base URL with your LLM Factory server address
   - Examples for common platforms:

   ```python
   # LangChain
   from langchain.chat_models import ChatOpenAI

   chat = ChatOpenAI(
       model_name="gpt-4o",  # Your configured model
       openai_api_key="any-key",
       openai_api_base="http://localhost:8000/v1"
   )

   # LlamaIndex
   from llama_index.llms import OpenAI

   llm = OpenAI(
       model="gpt-4o",
       api_key="any-key",
       api_base="http://localhost:8000/v1"
   )

   # AutoGPT
   {
       "OPENAI_API_KEY": "any-key",
       "OPENAI_API_BASE_URL": "http://localhost:8000/v1",
       "OPENAI_API_MODEL": "gpt-4o"
   }
   ```

3. For Curl requests:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any-key" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Model Mapping

The model mapping in LLM Factory is determined by your configuration. When using it as an OpenAI API alternative, the model name you specify in API calls should match the `model_name` in your configuration.

#### Configuration-Based Mapping

1. Using Environment Variables:
```bash
# Your .env configuration
OPENAI_MODEL="gpt-4"              # This will be your model name in API calls
QWEN_MODEL="qwen-turbo"           # This will be your model name in API calls
DEEPSEEK_MODEL="deepseek-chat"    # This will be your model name in API calls
```

2. Using YAML Configuration:
```yaml
providers:
  - provider: "openai"
    model_name: "gpt-4"           # Use this name in API calls
    api_key: "${OPENAI_API_KEY}"
    api_base: "${OPENAI_API_BASE}"

  - provider: "qwen"
    model_name: "qwen-turbo"      # Use this name in API calls
    api_key: "${QWEN_API_KEY}"
    api_base: "https://dashscope.aliyuncs.com/api/v1"

  - provider: "deepseek"
    model_name: "deepseek-chat"   # Use this name in API calls
    api_key: "${DEEPSEEK_API_KEY}"
```

3. Using Direct Configuration:
```python
configs = [
    ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",       # Use this name in API calls
        api_key="your-key",
        api_base="your-base"
    ),
    ModelConfig(
        provider=ProviderType.QWEN,
        model_name="qwen-turbo",  # Use this name in API calls
        api_key="your-key"
    )
]
factory = LLMFactory(configs)
```

#### Using Models in API Calls

After configuration, use the configured model names in your API calls:

```python
# Using OpenAI client
client = OpenAI(
    api_key="any-key",
    base_url="http://localhost:8000/v1"
)

# Use the exact model name from your configuration
response = client.chat.completions.create(
    model="gpt-4",               # Must match model_name in config
    messages=[{"role": "user", "content": "Hello!"}]
)

# You can use any configured model
response = client.chat.completions.create(
    model="qwen-turbo",         # Must match model_name in config
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### Load Balancing with Multiple Models

If you configure multiple instances of the same model, LLM Factory will automatically handle load balancing:

```yaml
providers:
  - provider: "openai"
    model_name: "gpt-4"          # Same model name
    api_key: "key1"
    api_base: "base1"

  - provider: "openai"
    model_name: "gpt-4"          # Same model name
    api_key: "key2"
    api_base: "base2"
```

In this case, requests to "gpt-4" will be automatically load balanced between the two configurations using the specified strategy (round-robin, random, or first-available).

#### Model Name Flexibility

- You can use any model name in your configuration
- The model name in API calls must exactly match your configuration
- Multiple providers can use the same model name for load balancing
- Different model names can point to the same provider type

For example:
```yaml
providers:
  - provider: "openai"
    model_name: "my-fast-model"     # Custom name
    api_key: "${OPENAI_API_KEY}"

  - provider: "openai"
    model_name: "my-smart-model"    # Different name, same provider
    api_key: "${OPENAI_API_KEY2}"
```

### Available Endpoints

1. Chat Completion: `/v1/chat/completions`
   ```bash
   curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gpt-4o",
       "messages": [{"role": "user", "content": "Hello!"}],
       "temperature": 0.7,
       "max_tokens": 1000,
       "stream": false
     }'
   ```

2. List Models: `/v1/models`
   ```bash
   curl "http://localhost:8000/v1/models"
   ```

3. Provider Status: `/v1/providers/status`
   ```bash
   curl "http://localhost:8000/v1/providers/status"
   ```

### Request Format

```json
{
  "model": "gpt-4o",           // Model to use
  "messages": [                // Chat messages
    {
      "role": "system",       // Optional system message
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "temperature": 0.7,         // Optional: 0.0 to 1.0
  "max_tokens": 1000,         // Optional: max tokens to generate
  "stream": false,            // Optional: enable streaming
  "load_balance_strategy": "round_robin"  // Optional: load balancing strategy
}
```

### Streaming Example

Using curl:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Write a story"}],
    "stream": true
  }' --no-buffer
```

Using Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Write a story"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### Custom Integration

You can also integrate the API into your existing FastAPI application:

```python
from fastapi import FastAPI
from llm_factory.api import router as llm_router

app = FastAPI()
app.include_router(llm_router, prefix="/v1")
```

### API Response Format

1. Chat Completion Response:
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
```

2. Streaming Response Format:
```json
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: [DONE]
```

### Error Handling

The API follows standard HTTP status codes:
- 400: Bad Request (invalid parameters)
- 401: Unauthorized (invalid API key)
- 404: Not Found (invalid endpoint or model)
- 429: Too Many Requests (rate limit exceeded)
- 500: Internal Server Error (provider error)

Error response format:
```json
{
  "error": {
    "message": "Detailed error message",
    "type": "error_type",
    "code": 400
  }
}
```

## Error Handling

The factory automatically handles:
- API key rotation on failure
- Connection errors
- Rate limiting
- Model availability issues

Example with error handling:

```python
try:
    response = await factory.chat_async("Hello")
except Exception as e:
    logger.error(f"Chat failed: {e}")
    # Handle error or try alternative provider
```

## Best Practices

1. Configuration:
   - Use environment variables for sensitive data
   - Use YAML for complex configurations
   - Use direct ModelConfig for dynamic configurations

2. Load Balancing:
   - Use multiple API keys for high availability
   - Choose appropriate load balancing strategy
   - Monitor provider status

3. Error Handling:
   - Always implement proper error handling
   - Use automatic failover for critical applications
   - Monitor and log errors

4. Async Usage:
   - Use async methods in async applications
   - Use sync methods only in sync contexts
   - Don't mix sync and async calls
