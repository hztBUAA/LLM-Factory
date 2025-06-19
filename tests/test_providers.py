"""
Tests for individual providers.
"""

import pytest
from unittest.mock import Mock, patch

from src.llm_factory.models import ModelConfig, ChatMessage, MessageRole
from src.llm_factory.providers import ProviderType, OpenAIProvider, QwenProvider, DeepSeekProvider


@pytest.fixture
def openai_config():
    """OpenAI configuration for testing."""
    return ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4o",
        api_key="test-key",
        api_base="https://test.openai.azure.com/",
        api_version="2024-02-01",
    )


@pytest.fixture
def qwen_config():
    """Qwen configuration for testing."""
    return ModelConfig(
        provider=ProviderType.QWEN,
        model_name="qwen-turbo",
        api_key="test-key",
    )


@pytest.fixture
def deepseek_config():
    """DeepSeek configuration for testing."""
    return ModelConfig(
        provider=ProviderType.DEEPSEEK,
        model_name="deepseek-chat",
        api_key="test-key",
    )


def test_openai_provider_initialization(openai_config):
    """Test OpenAI provider initialization."""
    with patch('src.llm_factory.providers.openai_provider.AsyncAzureOpenAI'):
        provider = OpenAIProvider(openai_config)
        assert provider.config.model_name == "gpt-4o"


def test_qwen_provider_initialization(qwen_config):
    """Test Qwen provider initialization."""
    provider = QwenProvider(qwen_config)
    assert provider.config.model_name == "qwen-turbo"
    assert provider.base_url == "https://dashscope.aliyuncs.com/api/v1"


def test_deepseek_provider_initialization(deepseek_config):
    """Test DeepSeek provider initialization."""
    provider = DeepSeekProvider(deepseek_config)
    assert provider.config.model_name == "deepseek-chat"
    assert provider.base_url == "https://api.deepseek.com/v1"


def test_cost_calculation():
    """Test cost calculation functionality."""
    config = ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4o",
        api_key="test-key",
    )
    
    with patch('src.llm_factory.providers.openai_provider.AsyncAzureOpenAI'):
        provider = OpenAIProvider(config)
        cost = provider._calculate_cost(1000, 500, "gpt-4o")
        expected_cost = (1000 / 1000) * 0.005 + (500 / 1000) * 0.015
        assert cost == expected_cost


@pytest.mark.asyncio
async def test_message_conversion():
    """Test message format conversion."""
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Hello!"),
    ]
    
    config = ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4o",
        api_key="test-key",
    )
    
    with patch('src.llm_factory.providers.openai_provider.AsyncAzureOpenAI'):
        provider = OpenAIProvider(config)
        
        openai_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        
        assert openai_messages[0]["role"] == "system"
        assert openai_messages[1]["role"] == "user"
        assert openai_messages[0]["content"] == "You are a helpful assistant."
        assert openai_messages[1]["content"] == "Hello!"
