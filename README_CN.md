# LLM Factory

统一的 LLM 工厂系统，为多个 AI 模型提供商提供一致的接口。

## 支持的提供商

- **OpenAI** - Azure OpenAI 实现（支持 GPT-4o、GPT-4 等）
- **Qwen** - 阿里云通义千问
- **DeepSeek** - DeepSeek 模型
- **Claude** - AWS Bedrock Claude 实现
- **Gemini** - Google Gemini 模型

## 安装指南

### 使用 venv 安装

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

### 使用 Conda 安装

```bash
# 创建新的 conda 环境
conda create -n llm-factory python=3.11

# 激活环境
conda activate llm-factory

# 安装依赖
pip install -r requirements.txt
```

## 特性

- 统一的多 LLM 提供商接口
- 支持每个提供商的多 API 密钥负载均衡
- 自动故障转移支持
- 支持流式和非流式响应
- 同时支持同步和异步 API
- 多种负载均衡策略（轮询、随机、优先可用）
- OpenAI 兼容的 JSON 输出格式
- 可扩展的工厂模式，便于添加新的提供商
- 简单的单例工厂模式，支持环境变量配置

## 使用方法

### 配置方式

你可以选择使用环境变量或配置文件来初始化 LLM Factory：

#### 方式一：环境变量配置

创建 `.env` 文件，配置你的 API 密钥：

```bash
# Azure OpenAI 配置
OPENAI_API_KEYS="key1,key2,key3"  # 多个 API 密钥用逗号分隔，用于负载均衡
OPENAI_API_BASES="https://account1.openai.azure.com,https://account2.openai.azure.com,https://account3.openai.azure.com"
OPENAI_MODEL="gpt-4o"  # 或其他模型如 gpt-4
OPENAI_API_VERSION="2024-02-15-preview"

# 其他可选的提供商配置
QWEN_API_KEYS="key1,key2"
QWEN_MODEL="qwen-turbo"
QWEN_API_BASE="你的千问API地址"

DEEPSEEK_API_KEYS="key1,key2"
DEEPSEEK_MODEL="deepseek-chat"
DEEPSEEK_API_BASE="你的DeepSeek API地址"

CLAUDE_ACCESS_KEYS="key1,key2"
CLAUDE_MODEL="anthropic.claude-3-5-sonnet-20241022-v2:0"
CLAUDE_SECRET_KEY="你的Claude密钥"
CLAUDE_REGION="us-east-1"

GEMINI_API_KEYS="key1,key2"
GEMINI_MODEL="gemini-2.0-flash-exp"
GEMINI_PROJECT_ID="你的项目ID"
GEMINI_REGION="你的地区"
```

#### 方式二：YAML 配置文件

创建 `config.yaml` 文件：

```yaml
providers:
  - provider: "openai"
    model_name: "gpt-4o"
    api_key: "${OPENAI_API_KEY}"  # 仍然支持从环境变量读取敏感信息
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
  
  # 其他提供商配置...
```

### 基本用法

```python
from llm_factory import LLMFactory

# 方式一：使用环境变量创建工厂实例（自动从 .env 加载配置）
factory = LLMFactory.create()

# 指定其他环境变量文件
factory = LLMFactory.create(env_file=".env.local")

# 方式二：使用配置文件创建工厂实例
factory = LLMFactory.create_from_config("config.yaml")

# 同步调用
response = factory.chat("你好，世界！")
print(response.choices[0]["message"]["content"])

# 异步调用
async def async_chat():
    response = await factory.chat_async("你好，世界！")
    print(response.choices[0]["message"]["content"])
```

### 同步和异步使用说明

1. 同步环境使用：
   - 使用 `factory.chat()` 方法
   - 适用于脚本、命令行工具等非异步场景
   - 示例：
   ```python
   response = factory.chat("你好，世界！")
   ```

2. 异步环境使用（如 FastAPI、aiohttp 等）：
   - 必须使用 `await factory.chat_async()`
   - 不要使用 `factory.chat()`，这会抛出 RuntimeError
   - FastAPI 示例：
   ```python
   @app.post("/chat")
   async def chat_endpoint(request: ChatRequest):
       response = await factory.chat_async(request.message)
       return response
   ```

3. 最佳实践：
   - 在异步应用中始终使用 `chat_async()`
   - 只在纯同步环境中使用 `chat()`
   - 避免在同一应用中混用两种方法
   - 注意处理在异步上下文中使用 `chat()` 可能抛出的 RuntimeError

### 负载均衡示例

工厂会自动处理所有配置的 API 密钥之间的负载均衡：

```python
# 基于环境变量自动进行负载均衡
factory = LLMFactory.create()

# 使用不同的负载均衡策略
response = await factory.chat_async(
    "你好！",
    load_balance_strategy="round_robin"  # 可选："random"（随机）, "first_available"（优先可用）
)
```

### 流式输出用法

```python
# 流式响应
async for chunk in factory.stream_async("写一个故事"):
    if chunk.choices and chunk.choices[0].get("delta", {}).get("content"):
        print(chunk.choices[0]["delta"]["content"], end="", flush=True)
```

### FastAPI 接口使用

```bash
# 首先确保 .env 文件配置正确

# 启动 API 服务器
python -m llm_factory.api

# 发送 API 请求
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "你好！"}],
    "temperature": 0.7,
    "max_tokens": 1000
  }'
```

### API 请求格式

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": "你好！"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false,
  "load_balance_strategy": "round_robin"
}
```

## 主要参数说明

- `model`: 使用的模型名称
- `messages`: 对话消息列表
- `temperature`: 温度参数，控制输出的随机性（0-1）
- `max_tokens`: 最大输出 token 数
- `stream`: 是否使用流式输出
- `load_balance_strategy`: 负载均衡策略
  - `round_robin`: 轮询（默认）
  - `random`: 随机
  - `first_available`: 优先使用第一个可用的

## 错误处理

系统会自动处理以下情况：
1. API 密钥无效或过期
2. 服务器连接失败
3. 配额超限
4. 模型不可用

当遇到错误时，系统会：
1. 自动尝试其他可用的 API 密钥
2. 如果所有密钥都失败，返回详细的错误信息
3. 记录错误日志以便调试

## 开发说明

1. 添加新的模型提供商：
   - 在 `providers` 目录下创建新的提供商类
   - 实现必要的接口方法
   - 在工厂类中注册新的提供商

2. 自定义配置：
   - 可以通过环境变量覆盖默认配置
   - 支持运行时动态更新配置

3. 日志记录：
   - 使用 loguru 进行日志记录
   - 可配置日志级别和输出位置

## 注意事项

1. API 密钥安全：
   - 不要将 API 密钥直接硬编码在代码中
   - 使用环境变量或配置文件管理密钥
   - 在生产环境中使用加密存储

2. 性能优化：
   - 合理设置最大 token 数
   - 适当使用流式输出
   - 根据需求选择合适的负载均衡策略 