"""
Base provider class for LLM implementations.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from ..models import ChatMessage, ChatResponse, ModelConfig, StreamChunk, Usage


class BaseProvider(ABC):
    """Base class for all LLM providers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._setup_client()
    
    @abstractmethod
    def _setup_client(self) -> None:
        """Setup the client for this provider."""
        pass
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        **kwargs: Any
    ) -> Union[ChatResponse, AsyncGenerator[StreamChunk, None]]:
        """Generate a chat completion."""
        pass
    
    @abstractmethod
    async def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        **kwargs: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate a streaming chat completion."""
        pass
    
    def _create_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: Optional[float] = None
    ) -> Usage:
        """Create usage information."""
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost
        )
    
    def _generate_id(self) -> str:
        """Generate a unique ID for responses."""
        return f"chatcmpl-{int(time.time() * 1000)}"
    
    def _get_current_timestamp(self) -> int:
        """Get current timestamp."""
        return int(time.time())
    
    def _calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str
    ) -> Optional[float]:
        """Calculate cost based on token usage."""
        cost_per_1k_input = self._get_input_cost_per_1k(model)
        cost_per_1k_output = self._get_output_cost_per_1k(model)
        
        if cost_per_1k_input is None or cost_per_1k_output is None:
            return None
            
        input_cost = (prompt_tokens / 1000) * cost_per_1k_input
        output_cost = (completion_tokens / 1000) * cost_per_1k_output
        
        return input_cost + output_cost
    
    def _get_input_cost_per_1k(self, model: str) -> Optional[float]:
        """Get input cost per 1K tokens for a model."""
        return None
    
    def _get_output_cost_per_1k(self, model: str) -> Optional[float]:
        """Get output cost per 1K tokens for a model."""
        return None
    
    def _setup_proxy(self) -> Optional[Dict[str, str]]:
        """Setup proxy configuration if provided."""
        if not self.config.proxy_config:
            return None
        
        if hasattr(self.config.proxy_config, 'to_dict'):
            return self.config.proxy_config.to_dict()
        
        return self.config.proxy_config
