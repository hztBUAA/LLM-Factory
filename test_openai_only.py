"""
Test script to isolate OpenAI provider issues.
"""

import sys
sys.path.insert(0, 'src')

from llm_factory.models import ModelConfig
from llm_factory.providers import ProviderType, OpenAIProvider

def test_openai_provider_direct():
    """Test OpenAI provider directly."""
    try:
        config = ModelConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4o",
            api_key="test-key",
            api_base="https://test.openai.azure.com/",
            api_version="2024-02-01",
            proxy_config=None,
        )
        
        print(f"Config created: {config}")
        
        provider = OpenAIProvider(config)
        print(f"✓ OpenAI provider created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ OpenAI provider creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== OpenAI Provider Direct Test ===")
    test_openai_provider_direct()
