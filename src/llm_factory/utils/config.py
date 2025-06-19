"""
Configuration utilities.
"""

import os
from typing import Dict, List, Optional

import yaml
from loguru import logger

from ..models import ModelConfig, ProviderType


def load_config_from_env() -> List[ModelConfig]:
    """Load configuration from environment variables."""
    configs = []
    
    if os.getenv("OPENAI_API_KEY"):
        configs.append(ModelConfig(
            provider=ProviderType.OPENAI,
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            proxy_config=_get_proxy_config("OPENAI"),
        ))
    
    if os.getenv("QWEN_API_KEY"):
        configs.append(ModelConfig(
            provider=ProviderType.QWEN,
            model_name=os.getenv("QWEN_MODEL", "qwen-turbo"),
            api_key=os.getenv("QWEN_API_KEY"),
            api_base=os.getenv("QWEN_API_BASE"),
            proxy_config=_get_proxy_config("QWEN"),
        ))
    
    if os.getenv("DEEPSEEK_API_KEY"):
        configs.append(ModelConfig(
            provider=ProviderType.DEEPSEEK,
            model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_API_BASE"),
            proxy_config=_get_proxy_config("DEEPSEEK"),
        ))
    
    if os.getenv("CLAUDE_ACCESS_KEY"):
        configs.append(ModelConfig(
            provider=ProviderType.CLAUDE,
            model_name=os.getenv("CLAUDE_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
            api_key=os.getenv("CLAUDE_ACCESS_KEY"),
            api_base=os.getenv("CLAUDE_SECRET_KEY"),
            region=os.getenv("CLAUDE_REGION", "us-east-1"),
            proxy_config=_get_proxy_config("CLAUDE"),
        ))
    
    if os.getenv("GEMINI_API_KEY"):
        configs.append(ModelConfig(
            provider=ProviderType.GEMINI,
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
            api_key=os.getenv("GEMINI_API_KEY"),
            project_id=os.getenv("GEMINI_PROJECT_ID"),
            region=os.getenv("GEMINI_REGION"),
            proxy_config=_get_proxy_config("GEMINI"),
        ))
    
    return configs


def load_config_from_file(file_path: str) -> List[ModelConfig]:
    """Load configuration from YAML file."""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        configs = []
        for provider_data in data.get('providers', []):
            configs.append(ModelConfig(**provider_data))
        
        return configs
        
    except Exception as e:
        logger.error(f"Failed to load config from {file_path}: {e}")
        return []


def _get_proxy_config(provider: str) -> Optional[Dict[str, str]]:
    """Get proxy configuration for a specific provider."""
    http_proxy = os.getenv(f"{provider}_HTTP_PROXY") or os.getenv("HTTP_PROXY")
    https_proxy = os.getenv(f"{provider}_HTTPS_PROXY") or os.getenv("HTTPS_PROXY")
    
    if http_proxy or https_proxy:
        proxy_config = {}
        if http_proxy:
            proxy_config["http"] = http_proxy
        if https_proxy:
            proxy_config["https"] = https_proxy
        return proxy_config
    
    return None
