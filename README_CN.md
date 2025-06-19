# LLM Factory

统一的 LLM 工厂系统，为多个 AI 模型提供商提供一致的接口。

## 支持的提供商

- **OpenAI** - Azure OpenAI 实现（支持 GPT-4o、GPT-4 等）
- **Qwen** - 阿里云通义千问
- **DeepSeek** - DeepSeek 模型
- **Claude** - AWS Bedrock Claude 实现
- **Gemini** - Google Gemini 模型

## 特性

- 统一的多 LLM 提供商接口
- 支持每个提供商的多 API 密钥负载均衡
- 自动故障转移支持
- 支持流式和非流式响应
- 同时支持同步和异步 API
- 多种负载均衡策略（轮询、随机、优先可用）
- OpenAI 兼容的 JSON 输出格式
- 可扩展的工厂模式，便于添加新的提供商
- 灵活的配置支持（环境变量、YAML 或编程方式）

## 安装

### 使用 venv

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
## Windows
venv\Scripts\activate
## Linux/MacOS
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 使用 Conda

```bash
# 创建新的 conda 环境
conda create -n llm-factory python=3.11

# 激活环境
conda activate llm-factory

# 安装依赖
pip install -r requirements.txt
```

## 配置和使用

LLM Factory 提供三种配置和使用方式：

### 1. 直接模型配置

最直接的方式是使用特定模型：

```python
from llm_factory import LLMFactory, ModelConfig, ProviderType

# 配置单个模型
config = ModelConfig(
    provider=ProviderType.OPENAI,
    model_name="gpt-4o",
    api_key="your-api-key",
    api_base="your-api-base",
    api_version="2024-02-15-preview"
)

# 使用单个配置创建工厂
factory = LLMFactory(config)

# 使用工厂
response = factory.chat("你好，世界！")
```

### 2. 多模型配置

用于跨多个模型或提供商的负载均衡：

```python
# 配置多个模型
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

# 使用多个配置创建工厂
factory = LLMFactory(configs)

# 使用负载均衡
response = factory.chat(
    "你好！",
    load_balance_strategy="round_robin"  # 可选："random"（随机）, "first_available"（优先可用）
)
```

### 3. 基于配置文件

#### 环境变量配置 (.env)

创建 `.env` 文件：

```bash
# Azure OpenAI 配置
OPENAI_API_KEYS="key1,key2,key3"  # 多个 API 密钥用于负载均衡
OPENAI_API_BASES="https://account1.openai.azure.com,https://account2.openai.azure.com"
OPENAI_MODEL="gpt-4o"
OPENAI_API_VERSION="2024-02-15-preview"

# 可选：其他提供商配置
QWEN_API_KEYS="key1,key2"
QWEN_MODEL="qwen-turbo"
QWEN_API_BASE="your-qwen-api-base"
```

使用方式：

```python
# 从 .env 文件创建工厂
factory = LLMFactory.create()

# 或指定其他环境变量文件
factory = LLMFactory.create(env_file=".env.local")
```

#### YAML 配置

创建 `config.yaml` 文件：

```yaml
providers:
  - provider: "openai"
    model_name: "gpt-4o"
    api_key: "${OPENAI_API_KEY}"  # 支持从环境变量读取
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

使用方式：

```python
# 从 YAML 配置创建工厂
factory = LLMFactory.create_from_config("config.yaml")
```

## 同步与异步使用

工厂支持同步和异步两种使用方式：

1. 同步环境：
   ```python
   # 适用于脚本、命令行工具等
   response = factory.chat("你好，世界！")
   ```

2. 异步环境：
   ```python
   # 适用于 FastAPI、aiohttp 等
   async def chat():
       response = await factory.chat_async("你好，世界！")
   ```

3. 流式输出：
   ```python
   # 流式响应
   async for chunk in factory.stream_async("写一个故事"):
       if chunk.choices and chunk.choices[0].get("delta", {}).get("content"):
           print(chunk.choices[0]["delta"]["content"], end="", flush=True)
   ```

## API 集成

LLM Factory 提供了一个兼容 OpenAI API 的接口，让你可以在原本为 OpenAI API 设计的应用中使用各种非 OpenAI 模型（如通义千问、DeepSeek、Claude 等）。

### 启动 API 服务器

直接运行：
```bash
# 启动 API 服务器（默认地址：http://localhost:8000）
python main.py
```

服务器会自动从环境变量或配置文件中加载配置。

### 作为 OpenAI API 替代使用

服务器启动后，你可以将其作为 OpenAI API 的直接替代品在你的应用中使用：

1. 使用 OpenAI 官方 Python 客户端：
```python
from openai import OpenAI

client = OpenAI(
    api_key="any-key",  # 可以是任意字符串，因为我们使用的是本地服务器
    base_url="http://localhost:8000/v1"  # 指向你的本地 LLM Factory 服务器
)

# 像使用官方 OpenAI 客户端一样使用
response = client.chat.completions.create(
    model="gpt-4o",  # 或任何在你的 LLM Factory 中配置的模型
    messages=[{"role": "user", "content": "你好！"}]
)
```

2. 对于使用 OpenAI API 的应用：
   - 将 API 基础 URL 替换为你的 LLM Factory 服务器地址
   - 常见平台的示例：

   ```python
   # LangChain
   from langchain.chat_models import ChatOpenAI

   chat = ChatOpenAI(
       model_name="gpt-4o",  # 你配置的模型
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

3. 使用 Curl 请求：
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any-key" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "你好！"}]
  }'
```

### 模型映射

LLM Factory 中的模型映射是由你的配置决定的。当将其作为 OpenAI API 替代品使用时，在 API 调用中指定的模型名称必须与配置中的 `model_name` 完全匹配。

#### 基于配置的映射

1. 使用环境变量：
```bash
# 你的 .env 配置
OPENAI_MODEL="gpt-4"              # 这将是你在 API 调用中使用的模型名称
QWEN_MODEL="qwen-turbo"           # 这将是你在 API 调用中使用的模型名称
DEEPSEEK_MODEL="deepseek-chat"    # 这将是你在 API 调用中使用的模型名称
```

2. 使用 YAML 配置：
```yaml
providers:
  - provider: "openai"
    model_name: "gpt-4"           # 在 API 调用中使用此名称
    api_key: "${OPENAI_API_KEY}"
    api_base: "${OPENAI_API_BASE}"

  - provider: "qwen"
    model_name: "qwen-turbo"      # 在 API 调用中使用此名称
    api_key: "${QWEN_API_KEY}"
    api_base: "https://dashscope.aliyuncs.com/api/v1"

  - provider: "deepseek"
    model_name: "deepseek-chat"   # 在 API 调用中使用此名称
    api_key: "${DEEPSEEK_API_KEY}"
```

3. 使用直接配置：
```python
configs = [
    ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",       # 在 API 调用中使用此名称
        api_key="your-key",
        api_base="your-base"
    ),
    ModelConfig(
        provider=ProviderType.QWEN,
        model_name="qwen-turbo",  # 在 API 调用中使用此名称
        api_key="your-key"
    )
]
factory = LLMFactory(configs)
```

#### 在 API 调用中使用模型

配置完成后，在 API 调用中使用配置的模型名称：

```python
# 使用 OpenAI 客户端
client = OpenAI(
    api_key="any-key",
    base_url="http://localhost:8000/v1"
)

# 使用配置中的精确模型名称
response = client.chat.completions.create(
    model="gpt-4",               # 必须与配置中的 model_name 匹配
    messages=[{"role": "user", "content": "你好！"}]
)

# 你可以使用任何已配置的模型
response = client.chat.completions.create(
    model="qwen-turbo",         # 必须与配置中的 model_name 匹配
    messages=[{"role": "user", "content": "你好！"}]
)
```

#### 多模型负载均衡

如果你配置了同一模型的多个实例，LLM Factory 将自动处理负载均衡：

```yaml
providers:
  - provider: "openai"
    model_name: "gpt-4"          # 相同的模型名称
    api_key: "key1"
    api_base: "base1"

  - provider: "openai"
    model_name: "gpt-4"          # 相同的模型名称
    api_key: "key2"
    api_base: "base2"
```

在这种情况下，对 "gpt-4" 的请求将使用指定的策略（轮询、随机或优先可用）自动在两个配置之间进行负载均衡。

#### 模型名称的灵活性

- 你可以在配置中使用任意模型名称
- API 调用中的模型名称必须与配置完全匹配
- 多个提供商可以使用相同的模型名称来实现负载均衡
- 不同的模型名称可以指向相同的提供商类型

例如：
```yaml
providers:
  - provider: "openai"
    model_name: "my-fast-model"     # 自定义名称
    api_key: "${OPENAI_API_KEY}"

  - provider: "openai"
    model_name: "my-smart-model"    # 不同名称，相同提供商
    api_key: "${OPENAI_API_KEY2}"
```

### 可用接口

1. 聊天补全：`/v1/chat/completions`
   ```bash
   curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gpt-4o",
       "messages": [{"role": "user", "content": "你好！"}],
       "temperature": 0.7,
       "max_tokens": 1000,
       "stream": false
     }'
   ```

2. 列出模型：`/v1/models`
   ```bash
   curl "http://localhost:8000/v1/models"
   ```

3. 提供商状态：`/v1/providers/status`
   ```bash
   curl "http://localhost:8000/v1/providers/status"
   ```

### 请求格式

```json
{
  "model": "gpt-4o",           // 使用的模型
  "messages": [                // 对话消息
    {
      "role": "system",       // 可选的系统消息
      "content": "你是一个有帮助的助手。"
    },
    {
      "role": "user",
      "content": "你好！"
    }
  ],
  "temperature": 0.7,         // 可选：温度参数，0.0 到 1.0
  "max_tokens": 1000,         // 可选：生成的最大 token 数
  "stream": false,            // 可选：启用流式输出
  "load_balance_strategy": "round_robin"  // 可选：负载均衡策略
}
```

### 流式输出示例

使用 curl：
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "写一个故事"}],
    "stream": true
  }' --no-buffer
```

使用 Python：
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "写一个故事"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### 自定义集成

你也可以将 API 集成到现有的 FastAPI 应用中：

```python
from fastapi import FastAPI
from llm_factory.api import router as llm_router

app = FastAPI()
app.include_router(llm_router, prefix="/v1")
```

### API 响应格式

1. 聊天补全响应：
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "你好！我能帮你什么忙？"
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

2. 流式响应格式：
```json
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"choices":[{"index":0,"delta":{"content":"！"},"finish_reason":null}]}

data: [DONE]
```

### 错误处理

API 遵循标准 HTTP 状态码：
- 400：请求错误（无效参数）
- 401：未授权（无效的 API 密钥）
- 404：未找到（无效的接口或模型）
- 429：请求过多（超出速率限制）
- 500：内部服务器错误（提供商错误）

错误响应格式：
```json
{
  "error": {
    "message": "详细错误信息",
    "type": "error_type",
    "code": 400
  }
}
```

## 错误处理

工厂自动处理以下情况：
- API 密钥轮换失败
- 连接错误
- 速率限制
- 模型可用性问题

错误处理示例：

```python
try:
    response = await factory.chat_async("你好")
except Exception as e:
    logger.error(f"聊天失败: {e}")
    # 处理错误或尝试其他提供商
```

## 最佳实践

1. 配置：
   - 使用环境变量存储敏感数据
   - 使用 YAML 进行复杂配置
   - 使用直接的 ModelConfig 进行动态配置

2. 负载均衡：
   - 使用多个 API 密钥实现高可用
   - 选择合适的负载均衡策略
   - 监控提供商状态

3. 错误处理：
   - 始终实现适当的错误处理
   - 对关键应用使用自动故障转移
   - 监控和记录错误

4. 异步使用：
   - 在异步应用中使用异步方法
   - 仅在同步上下文中使用同步方法
   - 不要混用同步和异步调用 