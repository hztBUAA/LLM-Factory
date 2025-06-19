"""
DeepSeek provider implementation.
"""

import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
from loguru import logger

from ..models import ChatMessage, ChatResponse, ModelConfig, StreamChunk, Usage
from .base import BaseProvider


class DeepSeekProvider(BaseProvider):
    """DeepSeek provider implementation."""

    def _setup_client(self) -> None:
        """Setup DeepSeek client."""
        self.base_url = self.config.api_base or "https://api.deepseek.com/v1"
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
            deepseek_messages = [
                {"role": msg.role.value, "content": msg.content}
                for msg in messages
            ]

            payload = {
                "model": self.config.model_name,
                "messages": deepseek_messages,
                "temperature": kwargs.get("temperature", self.config.temperature or 1.0),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p or 1.0),
                "stream": False,
            }

            if kwargs.get("tools"):
                payload["tools"] = kwargs["tools"]
            if kwargs.get("tool_choice"):
                payload["tool_choice"] = kwargs["tool_choice"]
            if kwargs.get("response_format"):
                payload["response_format"] = kwargs["response_format"]

            proxy_config = self._setup_proxy()
            timeout = httpx.Timeout(self.config.timeout)

            proxy_url = None
            if proxy_config:
                proxy_url = proxy_config.get("http") or proxy_config.get("https")
            
            async with httpx.AsyncClient(proxy=proxy_url, timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

            usage_info = result.get("usage", {})
            usage = Usage(
                prompt_tokens=usage_info.get("prompt_tokens", 0),
                completion_tokens=usage_info.get("completion_tokens", 0),
                total_tokens=usage_info.get("total_tokens", 0),
                cost=self._calculate_cost(
                    usage_info.get("prompt_tokens", 0),
                    usage_info.get("completion_tokens", 0),
                    self.config.model_name
                )
            )

            return ChatResponse(
                id=result.get("id", self._generate_id()),
                created=result.get("created", self._get_current_timestamp()),
                model=result.get("model", self.config.model_name),
                choices=result.get("choices", []),
                usage=usage,
                system_fingerprint=result.get("system_fingerprint")
            )

        except Exception as e:
            logger.error(f"DeepSeek completion error: {e}")
            raise

    async def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        **kwargs: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate a streaming chat completion."""
        try:
            deepseek_messages = [
                {"role": msg.role.value, "content": msg.content}
                for msg in messages
            ]

            payload = {
                "model": self.config.model_name,
                "messages": deepseek_messages,
                "temperature": kwargs.get("temperature", self.config.temperature or 1.0),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p or 1.0),
                "stream": True,
            }

            if kwargs.get("tools"):
                payload["tools"] = kwargs["tools"]
            if kwargs.get("tool_choice"):
                payload["tool_choice"] = kwargs["tool_choice"]
            if kwargs.get("response_format"):
                payload["response_format"] = kwargs["response_format"]

            proxy_config = self._setup_proxy()
            timeout = httpx.Timeout(self.config.timeout)

            proxy_url = None
            if proxy_config:
                proxy_url = proxy_config.get("http") or proxy_config.get("https")
            
            async with httpx.AsyncClient(proxy=proxy_url, timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
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
                                usage_info = chunk_data.get("usage", {})

                                usage = None
                                if usage_info:
                                    usage = Usage(
                                        prompt_tokens=usage_info.get("prompt_tokens", 0),
                                        completion_tokens=usage_info.get("completion_tokens", 0),
                                        total_tokens=usage_info.get("total_tokens", 0),
                                        cost=self._calculate_cost(
                                            usage_info.get("prompt_tokens", 0),
                                            usage_info.get("completion_tokens", 0),
                                            self.config.model_name
                                        )
                                    )

                                yield StreamChunk(
                                    id=chunk_data.get("id", self._generate_id()),
                                    created=chunk_data.get("created", self._get_current_timestamp()),
                                    model=chunk_data.get("model", self.config.model_name),
                                    choices=chunk_data.get("choices", []),
                                    usage=usage
                                )

                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"DeepSeek streaming error: {e}")
            raise

    def _get_input_cost_per_1k(self, model: str) -> Optional[float]:
        """Get input cost per 1K tokens for DeepSeek models."""
        costs = {
            "deepseek-chat": 0.00014,
            "deepseek-coder": 0.00014,
            "deepseek-r1": 0.00055,
            "deepseek-r1-distill-qwen-32b": 0.00027,
            "deepseek-r1-distill-llama-8b": 0.00014,
        }
        return costs.get(model.lower())

    def _get_output_cost_per_1k(self, model: str) -> Optional[float]:
        """Get output cost per 1K tokens for DeepSeek models."""
        costs = {
            "deepseek-chat": 0.00028,
            "deepseek-coder": 0.00028,
            "deepseek-r1": 0.0022,
            "deepseek-r1-distill-qwen-32b": 0.0011,
            "deepseek-r1-distill-llama-8b": 0.00028,
        }
        return costs.get(model.lower())
