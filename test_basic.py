"""
Basic test script to verify the LLM Factory system works.
"""

import os
import sys
sys.path.insert(0, 'src')

from llm_factory import LLMFactory, ModelConfig, ProviderType, ChatMessage

def test_factory_creation():
    """Test basic factory creation."""
    try:
        config = ModelConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4o",
            api_key="test-key",
            api_base="https://test.openai.azure.com/",
            api_version="2024-02-01",
            proxy_config=None,  # Explicitly set to None
        )
        
        factory = LLMFactory(config)
        print(f"✓ Factory created successfully with {len(factory.providers)} provider(s)")
        
        status = factory.get_provider_status()
        print(f"✓ Provider status: {status}")
        
        return True
        
    except Exception as e:
        print(f"✗ Factory creation failed: {e}")
        return False

def test_multiple_providers():
    """Test factory with multiple providers."""
    try:
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
        print(f"✓ Multi-provider factory created with {len(factory.providers)} providers")
        print(f"  Note: Some providers may have been skipped due to missing dependencies")
        
        return True
        
    except Exception as e:
        print(f"✗ Multi-provider factory creation failed: {e}")
        return False

def test_message_creation():
    """Test message creation."""
    try:
        message = ChatMessage(role="user", content="Hello, world!")
        print(f"✓ Message created: {message.role} - {message.content}")
        
        return True
        
    except Exception as e:
        print(f"✗ Message creation failed: {e}")
        return False

if __name__ == "__main__":
    print("=== LLM Factory Basic Tests ===")
    
    tests = [
        test_message_creation,
        test_factory_creation,
        test_multiple_providers,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All basic tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)
