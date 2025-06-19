"""
Qwen provider implementation.
"""

import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
from loguru import logger

from ..models import ChatMessage, ChatResponse, ModelConfig, StreamChunk, Usage
from .base import BaseProvider


class QwenProvider(BaseProvider):
    """Qwen provider implementation."""
    
    def _setup_client(self) -> None:
        """Setup Qwen client."""
        self.base_url = self.config.api_base or "https://dashscope.aliyuncs.com/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
    
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        **kwargs: Any
    ) -> Union[ChatResponse, AsyncGenerator[StreamChunk, None]]:
        """Generate a chat completion."""
        if kwargs.get("stream", self.config.stream):
            return self._chat_completion_stream_wrapper(messages, **kwargs)
        
        return await self._chat_completion_non_stream(messages, **kwargs)
    
    async def _chat_completion_stream_wrapper(
        self,
        messages: List[ChatMessage],
        **kwargs: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """Wrapper for streaming chat completion."""
        async for chunk in self.chat_completion_stream(messages, **kwargs):
            yield chunk
    
    async def _chat_completion_non_stream(
        self,
        messages: List[ChatMessage],
        **kwargs: Any
    ) -> ChatResponse:
        """Generate a non-streaming chat completion."""
        try:
            qwen_messages = [
                {"role": msg.role.value, "content": msg.content}
                for msg in messages
            ]
            
            payload = {
                "model": self.config.model_name,
                "input": {
                    "messages": qwen_messages
                },
                "parameters": {
                    "temperature": kwargs.get("temperature", self.config.temperature or 0.7),
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens or 2000),
                    "top_p": kwargs.get("top_p", self.config.top_p or 0.8),
                }
            }
            
            if kwargs.get("tools"):
                payload["parameters"]["tools"] = kwargs["tools"]
            if kwargs.get("response_format") and kwargs["response_format"].get("type") == "json_object":
                payload["parameters"]["result_format"] = "message"
            
            proxy_config = self._setup_proxy()
            timeout = httpx.Timeout(self.config.timeout)
            
            proxy_url = None
            if proxy_config:
                proxy_url = proxy_config.get("http") or proxy_config.get("https")
            
            async with httpx.AsyncClient(proxy=proxy_url, timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/services/aigc/text-generation/generation",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
            
            if "output" not in result:
                raise ValueError(f"Invalid Qwen response: {result}")
            
            output = result["output"]
            usage_info = result.get("usage", {})
            
            usage = Usage(
                prompt_tokens=usage_info.get("input_tokens", 0),
                completion_tokens=usage_info.get("output_tokens", 0),
                total_tokens=usage_info.get("total_tokens", 0),
                cost=self._calculate_cost(
                    usage_info.get("input_tokens", 0),
                    usage_info.get("output_tokens", 0),
                    self.config.model_name
                )
            )
            
            choices = [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output.get("text", ""),
                    "tool_calls": output.get("tool_calls")
                },
                "finish_reason": output.get("finish_reason", "stop")
            }]
            
            return ChatResponse(
                id=self._generate_id(),
                created=self._get_current_timestamp(),
                model=self.config.model_name,
                choices=choices,
                usage=usage
            )
            
        except Exception as e:
            logger.error(f"Qwen completion error: {e}")
            raise
    
    async def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        **kwargs: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate a streaming chat completion."""
        try:
            qwen_messages = [
                {"role": msg.role.value, "content": msg.content}
                for msg in messages
            ]
            
            payload = {
                "model": self.config.model_name,
                "input": {
                    "messages": qwen_messages
                },
                "parameters": {
                    "temperature": kwargs.get("temperature", self.config.temperature or 0.7),
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens or 2000),
                    "top_p": kwargs.get("top_p", self.config.top_p or 0.8),
                    "incremental_output": True
                }
            }
            
            if kwargs.get("tools"):
                payload["parameters"]["tools"] = kwargs["tools"]
            
            proxy_config = self._setup_proxy()
            timeout = httpx.Timeout(self.config.timeout)
            
            proxy_url = None
            if proxy_config:
                proxy_url = proxy_config.get("http") or proxy_config.get("https")
            
            async with httpx.AsyncClient(proxy=proxy_url, timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/services/aigc/text-generation/generation",
                    headers=self.headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            
                            try:
                                chunk_data = json.loads(data)
                                output = chunk_data.get("output", {})
                                usage_info = chunk_data.get("usage", {})
                                
                                usage = None
                                if usage_info:
                                    usage = Usage(
                                        prompt_tokens=usage_info.get("input_tokens", 0),
                                        completion_tokens=usage_info.get("output_tokens", 0),
                                        total_tokens=usage_info.get("total_tokens", 0),
                                        cost=self._calculate_cost(
                                            usage_info.get("input_tokens", 0),
                                            usage_info.get("output_tokens", 0),
                                            self.config.model_name
                                        )
                                    )
                                
                                choices = [{
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": output.get("text", "")
                                    },
                                    "finish_reason": output.get("finish_reason")
                                }]
                                
                                yield StreamChunk(
                                    id=self._generate_id(),
                                    created=self._get_current_timestamp(),
                                    model=self.config.model_name,
                                    choices=choices,
                                    usage=usage
                                )
                                
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Qwen streaming error: {e}")
            raise
    
    def _get_input_cost_per_1k(self, model: str) -> Optional[float]:
        """Get input cost per 1K tokens for Qwen models."""
        costs = {
            "qwen-turbo": 0.002,
            "qwen-plus": 0.004,
            "qwen-max": 0.02,
            "qwen2-72b-instruct": 0.004,
            "qwen2-7b-instruct": 0.001,
        }
        return costs.get(model.lower())
    
    def _get_output_cost_per_1k(self, model: str) -> Optional[float]:
        """Get output cost per 1K tokens for Qwen models."""
        costs = {
            "qwen-turbo": 0.006,
            "qwen-plus": 0.012,
            "qwen-max": 0.06,
            "qwen2-72b-instruct": 0.012,
            "qwen2-7b-instruct": 0.003,
        }
        return costs.get(model.lower())
