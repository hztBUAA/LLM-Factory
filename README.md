# LLM Factory

A unified LLM factory system that provides a consistent interface for multiple AI model providers.

## Supported Providers

- **OpenAI** - Azure OpenAI implementation (GPT-4o, GPT-4, etc.)
- **Qwen** - Alibaba's Qwen models
- **DeepSeek** - DeepSeek models  
- **Claude** - AWS Bedrock implementation
- **Gemini** - Google Gemini models

## Installation Guide

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

### Configuration

You can initialize LLM Factory using either environment variables or a configuration file:

#### Option 1: Environment Variables

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

#### Option 2: YAML Configuration File

Create a `config.yaml` file:

```yaml
providers:
  - provider: "openai"
    model_name: "gpt-4o"
    api_key: "${OPENAI_API_KEY}"  # Still supports reading sensitive info from env vars
    api_base: "${OPENAI_API_BASE}"
    api_version: "2024-02-01"
    max_tokens: 4096
    temperature: 0.7
    timeout: 60
    max_retries: 3
    proxy_config:
      http: "${HTTP_PROXY}"
      https: "${HTTPS_PROXY}"

  - provider: "qwen"
    model_name: "qwen-turbo"
    api_key: "${QWEN_API_KEY}"
    api_base: "https://dashscope.aliyuncs.com/api/v1"
    max_tokens: 2000
    temperature: 0.7
    timeout: 60
    max_retries: 3
  
  # Other provider configurations...
```

### Basic Usage

```python
from llm_factory import LLMFactory

# Option 1: Create factory instance using environment variables (auto-loads from .env)
factory = LLMFactory.create()

# Or specify a different env file
factory = LLMFactory.create(env_file=".env.local")

# Option 2: Create factory instance using config file
factory = LLMFactory.create_from_config("config.yaml")

# Synchronous call
response = factory.chat("Hello, world!")
print(response.choices[0]["message"]["content"])

# Asynchronous call
async def async_chat():
    response = await factory.chat_async("Hello, world!")
    print(response.choices[0]["message"]["content"])
```

### Synchronous vs Asynchronous Usage

1. In Synchronous Environments:
   - Use `factory.chat()` method
   - Suitable for scripts, command-line tools, and non-async contexts
   - Example:
   ```python
   response = factory.chat("Hello, world!")
   ```

2. In Asynchronous Environments (FastAPI, aiohttp, etc.):
   - Must use `await factory.chat_async()`
   - Do not use `factory.chat()` as it will raise RuntimeError
   - Example in FastAPI:
   ```python
   @app.post("/chat")
   async def chat_endpoint(request: ChatRequest):
       response = await factory.chat_async(request.message)
       return response
   ```

3. Best Practices:
   - Always use `chat_async()` in async applications
   - Only use `chat()` in purely synchronous environments
   - Avoid mixing both methods in the same application
   - Handle potential RuntimeError when using `chat()` in async contexts

### 使用特定 ModelConfig 配置

您也可以直接使用 ModelConfig 来初始化 LLMFactory，这样可以更灵活地控制模型配置：

```python
from llm_factory import LLMFactory, ModelConfig, ProviderType

# 创建单个 ModelConfig
config = ModelConfig(
    provider=ProviderType.OPENAI,
    model_name="gpt-4o",
    api_key="your-api-key",
    api_base="your-api-base",
    api_version="2024-02-15-preview"
)

# 使用单个配置初始化工厂
factory = LLMFactory(config)

# 或者使用多个配置进行负载均衡
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

# 使用多个配置初始化工厂
factory = LLMFactory(configs)

# 使用方式与之前相同
response = factory.chat("Hello, world!")
# 或者异步调用
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
