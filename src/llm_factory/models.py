"""
Data models for the LLM Factory system.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message roles for chat conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ProxyConfig(BaseModel):
    """Proxy configuration for HTTP requests."""
    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None
    http: Optional[str] = None  # Alternative naming for compatibility
    https: Optional[str] = None  # Alternative naming for compatibility
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format for libraries that expect dict."""
        result = {}
        if self.http_proxy:
            result["http"] = self.http_proxy
        if self.https_proxy:
            result["https"] = self.https_proxy
        if self.http:
            result["http"] = self.http
        if self.https:
            result["https"] = self.https
        return result


class ChatMessage(BaseModel):
    """A single chat message."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: Optional[float] = None


class ChatResponse(BaseModel):
    """Response from a chat completion."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class StreamChunk(BaseModel):
    """A chunk from a streaming response."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Usage] = None


LoadBalanceStrategy = Literal["round_robin", "random", "first_available"]
ProviderName = Literal["openai", "qwen", "deepseek", "claude", "gemini"]


class ModelConfig(BaseModel):
    """Configuration for a model."""
    provider: ProviderName  # Use literal type for standardized provider names
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    region: Optional[str] = None
    project_id: Optional[str] = None
    proxy_config: Optional[Union[ProxyConfig, Dict[str, str]]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    timeout: int = 60
    max_retries: int = 3


class ToolCall(BaseModel):
    """A tool call in a message."""
    id: str
    type: str = "function"
    function: Dict[str, Any]


class FunctionDefinition(BaseModel):
    """Definition of a function that can be called."""
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]


class Tool(BaseModel):
    """A tool that can be used by the model."""
    type: str = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, str]] = None
