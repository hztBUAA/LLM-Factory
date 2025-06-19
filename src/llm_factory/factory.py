"""
Main LLM Factory class with load balancing and unified interface.
"""

import asyncio
import os
import random
import yaml
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from loguru import logger
from dotenv import load_dotenv

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
    
    _instance = None
    
    @classmethod
    def create(cls, env_file: str = ".env") -> 'LLMFactory':
        """
        Create a LLMFactory instance with configuration from environment variables.
        
        Args:
            env_file: Path to the environment file. Defaults to ".env"
            
        Returns:
            LLMFactory instance
        """
        if cls._instance is None:
            load_dotenv(env_file)
            configs = cls._load_configs_from_env()
            cls._instance = cls(configs)
        return cls._instance

    @classmethod
    def create_from_config(cls, config_file: str) -> 'LLMFactory':
        """
        Create a LLMFactory instance with configuration from a YAML file.
        
        Args:
            config_file: Path to the YAML configuration file
            
        Returns:
            LLMFactory instance
        """
        if cls._instance is None:
            configs = cls._load_configs_from_yaml(config_file)
            cls._instance = cls(configs)
        return cls._instance

    @staticmethod
    def _load_configs_from_yaml(config_file: str) -> List[ModelConfig]:
        """Load provider configurations from YAML file."""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        if not config_data or 'providers' not in config_data:
            raise ValueError("Invalid configuration file: 'providers' section not found")

        configs = []
        for provider_config in config_data['providers']:
            provider_type = provider_config.get('provider', '').upper()
            if not provider_type or not hasattr(ProviderType, provider_type):
                logger.warning(f"Skipping invalid provider type: {provider_type}")
                continue

            # 处理环境变量替换
            for key, value in provider_config.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    env_value = os.getenv(env_var)
                    if env_value:
                        if ',' in env_value:  # 处理多个值的情况
                            provider_config[key + 's'] = [v.strip() for v in env_value.split(',') if v.strip()]
                        else:
                            provider_config[key] = env_value
                    else:
                        logger.warning(f"Environment variable not found: {env_var}")

            # 处理多账户配置
            api_keys = provider_config.get('api_keys', [provider_config.get('api_key')])
            api_bases = provider_config.get('api_bases', [provider_config.get('api_base')])
            
            if isinstance(api_keys, str):
                api_keys = [api_keys]
            if isinstance(api_bases, str):
                api_bases = [api_bases]

            # 确保api_keys和api_bases长度匹配
            if len(api_bases) == 1 and len(api_keys) > 1:
                api_bases = api_bases * len(api_keys)
            elif len(api_bases) > 1 and len(api_keys) == 1:
                api_keys = api_keys * len(api_bases)

            # 为每个API密钥创建一个配置
            for i, api_key in enumerate(api_keys):
                if not api_key:
                    continue

                config_dict = {
                    'provider': getattr(ProviderType, provider_type),
                    'model_name': provider_config.get('model_name'),
                    'api_key': api_key,
                }

                # 添加API base（如果存在）
                if i < len(api_bases) and api_bases[i]:
                    config_dict['api_base'] = api_bases[i]

                # 添加其他配置参数
                for key, value in provider_config.items():
                    if key not in ['provider', 'model_name', 'api_key', 'api_keys', 'api_base', 'api_bases']:
                        config_dict[key] = value

                configs.append(ModelConfig(**config_dict))

        if not configs:
            raise ValueError("No valid provider configurations found in YAML file")

        return configs

    @staticmethod
    def _load_configs_from_env() -> List[ModelConfig]:
        """Load provider configurations from environment variables."""
        configs = []
        
        # OpenAI配置
        openai_keys = [k.strip() for k in os.getenv("OPENAI_API_KEYS", "").split(",") if k.strip()]
        openai_bases = [b.strip() for b in os.getenv("OPENAI_API_BASES", "").split(",") if b.strip()]
        
        if not openai_keys and os.getenv("OPENAI_API_KEY"):  # 兼容单个key的情况
            openai_keys = [os.getenv("OPENAI_API_KEY")]
            openai_bases = [os.getenv("OPENAI_API_BASE")] if os.getenv("OPENAI_API_BASE") else []
        
        if len(openai_keys) == len(openai_bases) and openai_keys:
            for api_key, api_base in zip(openai_keys, openai_bases):
                configs.append(ModelConfig(
                    provider=ProviderType.OPENAI,
                    model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
                    api_key=api_key,
                    api_base=api_base,
                    api_version=os.getenv("OPENAI_API_VERSION", "2024-02-15-preview"),
                ))
        
        # Qwen配置
        qwen_keys = [k.strip() for k in os.getenv("QWEN_API_KEYS", "").split(",") if k.strip()]
        if not qwen_keys and os.getenv("QWEN_API_KEY"):
            qwen_keys = [os.getenv("QWEN_API_KEY")]
            
        for api_key in qwen_keys:
            configs.append(ModelConfig(
                provider=ProviderType.QWEN,
                model_name=os.getenv("QWEN_MODEL", "qwen-turbo"),
                api_key=api_key,
                api_base=os.getenv("QWEN_API_BASE"),
            ))
        
        # Deepseek配置
        deepseek_keys = [k.strip() for k in os.getenv("DEEPSEEK_API_KEYS", "").split(",") if k.strip()]
        if not deepseek_keys and os.getenv("DEEPSEEK_API_KEY"):
            deepseek_keys = [os.getenv("DEEPSEEK_API_KEY")]
            
        for api_key in deepseek_keys:
            configs.append(ModelConfig(
                provider=ProviderType.DEEPSEEK,
                model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                api_key=api_key,
                api_base=os.getenv("DEEPSEEK_API_BASE"),
            ))
        
        # Claude配置
        claude_keys = [k.strip() for k in os.getenv("CLAUDE_ACCESS_KEYS", "").split(",") if k.strip()]
        if not claude_keys and os.getenv("CLAUDE_ACCESS_KEY"):
            claude_keys = [os.getenv("CLAUDE_ACCESS_KEY")]
            
        for api_key in claude_keys:
            configs.append(ModelConfig(
                provider=ProviderType.CLAUDE,
                model_name=os.getenv("CLAUDE_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
                api_key=api_key,
                api_base=os.getenv("CLAUDE_SECRET_KEY"),
                region=os.getenv("CLAUDE_REGION", "us-east-1"),
            ))
        
        # Gemini配置
        gemini_keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
        if not gemini_keys and os.getenv("GEMINI_API_KEY"):
            gemini_keys = [os.getenv("GEMINI_API_KEY")]
            
        for api_key in gemini_keys:
            configs.append(ModelConfig(
                provider=ProviderType.GEMINI,
                model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
                api_key=api_key,
                project_id=os.getenv("GEMINI_PROJECT_ID"),
                region=os.getenv("GEMINI_REGION"),
            ))
        
        if not configs:
            raise ValueError("No valid provider configurations found in environment variables")
        
        return configs
    
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
