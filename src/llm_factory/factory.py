"""
Main LLM Factory class with load balancing and unified interface.
"""

import asyncio
import random
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from loguru import logger

from .models import ChatMessage, ChatResponse, ModelConfig, StreamChunk
from .providers import (
    BaseProvider,
    ClaudeProvider,
    DeepSeekProvider,
    GeminiProvider,
    OpenAIProvider,
    ProviderType,
    QwenProvider,
)


class LLMFactory:
    """
    Unified LLM Factory with load balancing and simple interface.
    
    Supports multiple providers with automatic failover and load balancing.
    """
    
    def __init__(self, configs: Union[ModelConfig, List[ModelConfig]]):
        """
        Initialize the factory with one or more model configurations.
        
        Args:
            configs: Single config or list of configs for load balancing
        """
        if isinstance(configs, ModelConfig):
            configs = [configs]
        
        self.configs = configs
        self.providers: List[BaseProvider] = []
        self._setup_providers()
        
        if not self.providers:
            raise ValueError("No valid providers could be initialized")
    
    def _setup_providers(self) -> None:
        """Setup all providers based on configurations."""
        provider_map = {
            ProviderType.OPENAI: OpenAIProvider,
            ProviderType.QWEN: QwenProvider,
            ProviderType.DEEPSEEK: DeepSeekProvider,
            ProviderType.CLAUDE: ClaudeProvider,
            ProviderType.GEMINI: GeminiProvider,
        }
        
        for config in self.configs:
            try:
                provider_class = provider_map.get(config.provider)
                if provider_class:
                    provider = provider_class(config)
                    self.providers.append(provider)
                    logger.info(f"Initialized {config.provider} provider with model {config.model_name}")
                else:
                    logger.warning(f"Unknown provider: {config.provider}")
            except ImportError as e:
                logger.warning(f"Skipping {config.provider} provider due to missing dependencies: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize {config.provider} provider: {e}")
    
    def _get_provider_for_model(self, model_name: Optional[str] = None, strategy: str = "round_robin") -> BaseProvider:
        """
        Get a provider that supports the specified model with load balancing.
        
        Args:
            model_name: The model name to find a provider for
            strategy: Load balancing strategy to use among matching providers
        """
        if not self.providers:
            raise RuntimeError("No providers available")
        
        if model_name:
            matching_providers = [
                provider for provider in self.providers 
                if provider.config.model_name == model_name
            ]
            
            if matching_providers:
                return self._get_provider_from_list(matching_providers, strategy)
            
            logger.warning(f"No provider found for model {model_name}, using all providers for load balancing")
        
        return self._get_provider(strategy)
    def _get_provider_from_list(self, providers: List[BaseProvider], strategy: str = "round_robin") -> BaseProvider:
        """
        Get a provider from a specific list using load balancing strategy.
        
        Args:
            providers: List of providers to choose from
            strategy: Load balancing strategy ('round_robin', 'random', 'first_available')
        """
        if not providers:
            raise RuntimeError("No providers available in list")
        
        if strategy == "random":
            return random.choice(providers)
        elif strategy == "first_available":
            return providers[0]
        else:  # round_robin (default)
            list_id = id(providers)
            if not hasattr(self, '_provider_list_indices'):
                self._provider_list_indices = {}
            
            if list_id not in self._provider_list_indices:
                self._provider_list_indices[list_id] = 0
            
            provider = providers[self._provider_list_indices[list_id]]
            self._provider_list_indices[list_id] = (self._provider_list_indices[list_id] + 1) % len(providers)
            return provider


    
    def _get_provider(self, strategy: str = "round_robin") -> BaseProvider:
        """
        Get a provider based on load balancing strategy.
        
        Args:
            strategy: Load balancing strategy ('round_robin', 'random', 'first_available')
        """
        if not self.providers:
            raise RuntimeError("No providers available")
        
        if strategy == "random":
            return random.choice(self.providers)
        elif strategy == "first_available":
            return self.providers[0]
        else:  # round_robin (default)
            if not hasattr(self, '_current_provider_index'):
                self._current_provider_index = 0
            
            provider = self.providers[self._current_provider_index]
            self._current_provider_index = (self._current_provider_index + 1) % len(self.providers)
            return provider
    
    def chat(
        self,
        messages: Union[str, List[ChatMessage]],
        **kwargs: Any
    ) -> ChatResponse:
        """
        Synchronous chat completion.
        
        Args:
            messages: String message or list of ChatMessage objects
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        """
        return asyncio.run(self.chat_async(messages, **kwargs))
    
    async def chat_async(
        self,
        messages: Union[str, List[ChatMessage]],
        **kwargs: Any
    ) -> ChatResponse:
        """
        Asynchronous chat completion.
        
        Args:
            messages: String message or list of ChatMessage objects
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        """
        if isinstance(messages, str):
            messages = [ChatMessage(role="user", content=messages)]
        
        model_name = kwargs.get("model")
        strategy = kwargs.get("load_balance_strategy", "round_robin")
        if model_name:
            provider = self._get_provider_for_model(model_name, strategy)
        else:
            provider = self._get_provider(strategy)
        
        try:
            result = await provider.chat_completion(messages, **kwargs)
            if isinstance(result, ChatResponse):
                return result
            else:
                raise ValueError("Expected ChatResponse but got streaming response")
        except Exception as e:
            logger.error(f"Chat completion failed with {provider.__class__.__name__}: {e}")
            if len(self.providers) > 1:
                logger.info("Attempting failover to next provider")
                return await self._failover_chat(messages, provider, **kwargs)
            raise
    
    def stream(
        self,
        messages: Union[str, List[ChatMessage]],
        **kwargs: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Streaming chat completion.
        
        Args:
            messages: String message or list of ChatMessage objects
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        """
        return self.stream_async(messages, **kwargs)
    
    async def stream_async(
        self,
        messages: Union[str, List[ChatMessage]],
        **kwargs: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Asynchronous streaming chat completion.
        
        Args:
            messages: String message or list of ChatMessage objects
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        """
        if isinstance(messages, str):
            messages = [ChatMessage(role="user", content=messages)]
        
        model_name = kwargs.get("model")
        strategy = kwargs.get("load_balance_strategy", "round_robin")
        if model_name:
            provider = self._get_provider_for_model(model_name, strategy)
        else:
            provider = self._get_provider(strategy)
        
        try:
            async for chunk in provider.chat_completion_stream(messages, **kwargs):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming failed with {provider.__class__.__name__}: {e}")
            if len(self.providers) > 1:
                logger.info("Attempting failover to next provider")
                async for chunk in self._failover_stream(messages, provider, **kwargs):
                    yield chunk
            else:
                raise
    
    async def _failover_chat(
        self,
        messages: List[ChatMessage],
        failed_provider: BaseProvider,
        **kwargs: Any
    ) -> ChatResponse:
        """Failover to next available provider for chat completion."""
        for provider in self.providers:
            if provider != failed_provider:
                try:
                    result = await provider.chat_completion(messages, **kwargs)
                    if isinstance(result, ChatResponse):
                        return result
                except Exception as e:
                    logger.error(f"Failover failed with {provider.__class__.__name__}: {e}")
                    continue
        
        raise RuntimeError("All providers failed")
    
    async def _failover_stream(
        self,
        messages: List[ChatMessage],
        failed_provider: BaseProvider,
        **kwargs: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """Failover to next available provider for streaming."""
        for provider in self.providers:
            if provider != failed_provider:
                try:
                    async for chunk in provider.chat_completion_stream(messages, **kwargs):
                        yield chunk
                    return
                except Exception as e:
                    logger.error(f"Streaming failover failed with {provider.__class__.__name__}: {e}")
                    continue
        
        raise RuntimeError("All providers failed")
    
    def __call__(
        self,
        messages: Union[str, List[ChatMessage]],
        **kwargs: Any
    ) -> ChatResponse:
        """
        Callable interface for synchronous chat completion.
        
        Args:
            messages: String message or list of ChatMessage objects
            **kwargs: Additional parameters
        """
        return self.chat(messages, **kwargs)
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status information about all providers."""
        status = {
            "total_providers": len(self.providers),
            "providers": []
        }
        
        for i, provider in enumerate(self.providers):
            provider_info = {
                "index": i,
                "type": provider.__class__.__name__,
                "model": provider.config.model_name,
                "provider": provider.config.provider,
            }
            status["providers"].append(provider_info)
        
        return status
