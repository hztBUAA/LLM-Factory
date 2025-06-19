"""
Tests for the LLM Factory.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from src.llm_factory import LLMFactory, ModelConfig, ProviderType, ChatMessage


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4o",
        api_key="test-key",
        api_base="https://test.openai.azure.com/",
        api_version="2024-02-01",
    )


@pytest.fixture
def factory(mock_config):
    """Factory instance for testing."""
    return LLMFactory(mock_config)


def test_factory_initialization(mock_config):
    """Test factory initialization."""
    factory = LLMFactory(mock_config)
    assert len(factory.providers) == 1
    assert factory.providers[0].__class__.__name__ == "OpenAIProvider"


def test_factory_multiple_configs():
    """Test factory with multiple configurations."""
    configs = [
        ModelConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4o",
            api_key="test-key-1",
            api_base="https://test1.openai.azure.com/",
            api_version="2024-02-01",
        ),
        ModelConfig(
            provider=ProviderType.DEEPSEEK,
            model_name="deepseek-chat",
            api_key="test-key-2",
        ),
    ]
    
    factory = LLMFactory(configs)
    assert len(factory.providers) == 2


def test_get_provider_status(factory):
    """Test provider status retrieval."""
    status = factory.get_provider_status()
    assert status["total_providers"] == 1
    assert len(status["providers"]) == 1
    assert status["providers"][0]["type"] == "OpenAIProvider"


@pytest.mark.asyncio
async def test_chat_async_string_input(factory):
    """Test async chat with string input."""
    mock_response = Mock()
    mock_response.choices = [{"message": {"content": "Hello!"}}]
    
    factory.providers[0].chat_completion = AsyncMock(return_value=mock_response)
    
    response = await factory.chat_async("Hello")
    assert response.choices[0]["message"]["content"] == "Hello!"


def test_chat_sync(factory):
    """Test synchronous chat."""
    mock_response = Mock()
    mock_response.choices = [{"message": {"content": "Hello!"}}]
    
    factory.providers[0].chat_completion = AsyncMock(return_value=mock_response)
    
    response = factory.chat("Hello")
    assert response.choices[0]["message"]["content"] == "Hello!"


def test_callable_interface(factory):
    """Test callable interface."""
    mock_response = Mock()
    mock_response.choices = [{"message": {"content": "Hello!"}}]
    
    factory.providers[0].chat_completion = AsyncMock(return_value=mock_response)
    
    response = factory("Hello")
    assert response.choices[0]["message"]["content"] == "Hello!"
