"""
Basic test script to verify the LLM Factory system works.
"""

import os
import sys
import pytest

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_factory import LLMFactory, ModelConfig, ProviderType, ChatMessage


@pytest.fixture
def basic_config():
    """Basic configuration for testing."""
    return ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4o",
        api_key="test-key",
        api_base="https://test.openai.azure.com/",
        api_version="2024-02-01",
        proxy_config=None,  # Explicitly set to None
    )


def test_factory_creation(basic_config):
    """Test basic factory creation."""
    factory = LLMFactory(basic_config)
    assert len(factory.providers) == 1
    print(f"✓ Factory created successfully with {len(factory.providers)} provider(s)")
    
    status = factory.get_provider_status()
    print(f"✓ Provider status: {status}")


def test_multiple_providers():
    """Test factory with multiple providers."""
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
        ModelConfig(
            provider=ProviderType.GEMINI,
            model_name="gemini-2.0-flash-exp",
            api_key="test-key-3",
        ),
    ]
    
    factory = LLMFactory(configs)
    assert len(factory.providers) >= 1
    print(f"✓ Multi-provider factory created with {len(factory.providers)} providers")
    print(f"  Note: Some providers may have been skipped due to missing dependencies")


def test_message_creation():
    """Test message creation."""
    message = ChatMessage(role="user", content="Hello, world!")
    assert message.role.value == "user"
    assert message.content == "Hello, world!"
    print(f"✓ Message created: {message.role} - {message.content}")


def test_factory_from_env():
    """Test factory creation from environment variables."""
    factory = LLMFactory.create()
    assert factory is not None
    print(f"✓ Factory created from environment variables with {len(factory.providers)} provider(s)") 