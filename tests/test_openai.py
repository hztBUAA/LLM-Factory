"""
Test script to isolate OpenAI provider issues.
"""

import os
import sys
import pytest

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_factory.models import ModelConfig
from src.llm_factory.providers import ProviderType, OpenAIProvider


@pytest.fixture
def openai_config():
    """OpenAI configuration for testing."""
    return ModelConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4o",
        api_key="test-key",
        api_base="https://test.openai.azure.com/",
        api_version="2024-02-01",
        proxy_config=None,
    )


def test_openai_provider_creation(openai_config):
    """Test OpenAI provider creation."""
    try:
        print(f"Config created: {openai_config}")
        
        provider = OpenAIProvider(openai_config)
        assert provider is not None
        print(f"✓ OpenAI provider created successfully")
        
    except Exception as e:
        print(f"✗ OpenAI provider creation failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail("OpenAI provider creation failed") 