"""
OpenAI provider implementation using Azure OpenAI.
"""

import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
from openai import AsyncAzureOpenAI
from loguru import logger

from ..models import ChatMessage, ChatResponse, ModelConfig, StreamChunk, Usage
from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI provider using Azure OpenAI."""
    
    def _setup_client(self) -> None:
        """Setup Azure OpenAI client."""
        client_kwargs = {
            "api_key": self.config.api_key,
            "azure_endpoint": self.config.api_base,
            "api_version": self.config.api_version or "2024-02-01",
        }
        
        if self.config.timeout:
            client_kwargs["timeout"] = self.config.timeout
        if self.config.max_retries:
            client_kwargs["max_retries"] = self.config.max_retries
        
        proxy_dict = self._setup_proxy()
        if proxy_dict:
            proxy_url = proxy_dict.get("http") or proxy_dict.get("https")
            if proxy_url:
                client_kwargs["http_client"] = httpx.AsyncClient(
                    proxy=proxy_url
                )
        
        self.client = AsyncAzureOpenAI(**client_kwargs)
    
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
            openai_messages = [
                {"role": msg.role.value, "content": msg.content}
                for msg in messages
            ]
            
            completion_kwargs = {
                "model": self.config.model_name,
                "messages": openai_messages,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "stream": False,
            }
            
            if kwargs.get("tools"):
                completion_kwargs["tools"] = kwargs["tools"]
            if kwargs.get("tool_choice"):
                completion_kwargs["tool_choice"] = kwargs["tool_choice"]
            if kwargs.get("response_format"):
                completion_kwargs["response_format"] = kwargs["response_format"]
            
            response = await self.client.chat.completions.create(**completion_kwargs)
            
            usage = None
            if response.usage:
                cost = self._calculate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    self.config.model_name
                )
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    cost=cost
                )
            
            return ChatResponse(
                id=response.id,
                created=response.created,
                model=response.model,
                choices=[choice.model_dump() for choice in response.choices],
                usage=usage,
                system_fingerprint=response.system_fingerprint
            )
            
        except Exception as e:
            logger.error(f"OpenAI completion error: {e}")
            raise
    
    async def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        **kwargs: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate a streaming chat completion."""
        try:
            openai_messages = [
                {"role": msg.role.value, "content": msg.content}
                for msg in messages
            ]
            
            completion_kwargs = {
                "model": self.config.model_name,
                "messages": openai_messages,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            
            if kwargs.get("tools"):
                completion_kwargs["tools"] = kwargs["tools"]
            if kwargs.get("tool_choice"):
                completion_kwargs["tool_choice"] = kwargs["tool_choice"]
            if kwargs.get("response_format"):
                completion_kwargs["response_format"] = kwargs["response_format"]
            
            stream = await self.client.chat.completions.create(**completion_kwargs)
            
            async for chunk in stream:
                usage = None
                if chunk.usage:
                    cost = self._calculate_cost(
                        chunk.usage.prompt_tokens,
                        chunk.usage.completion_tokens,
                        self.config.model_name
                    )
                    usage = Usage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                        cost=cost
                    )
                
                yield StreamChunk(
                    id=chunk.id,
                    created=chunk.created,
                    model=chunk.model,
                    choices=[choice.model_dump() for choice in chunk.choices],
                    usage=usage
                )
                
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
    
    def _get_input_cost_per_1k(self, model: str) -> Optional[float]:
        """Get input cost per 1K tokens for OpenAI models."""
        costs = {
            "gpt-4o": 0.005,
            "gpt-4o-mini": 0.00015,
            "gpt-4": 0.03,
            "gpt-4-32k": 0.06,
            "gpt-3.5-turbo": 0.0015,
        }
        return costs.get(model.lower())
    
    def _get_output_cost_per_1k(self, model: str) -> Optional[float]:
        """Get output cost per 1K tokens for OpenAI models."""
        costs = {
            "gpt-4o": 0.015,
            "gpt-4o-mini": 0.0006,
            "gpt-4": 0.06,
            "gpt-4-32k": 0.12,
            "gpt-3.5-turbo": 0.002,
        }
        return costs.get(model.lower())
