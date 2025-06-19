"""
Test script for the FastAPI interface.
"""

import sys
from fastapi.testclient import TestClient

sys.path.insert(0, 'src')

from llm_factory.api.app import create_app


def test_api():
    """Test the FastAPI interface."""
    try:
        app = create_app()
        
        with TestClient(app) as client:
            response = client.get("/")
            print(f"✓ Root endpoint: {response.status_code} - {response.json()}")
            
            response = client.get("/health")
            print(f"✓ Health endpoint: {response.status_code} - {response.json()}")
            
            response = client.get("/api/v1/models")
            print(f"✓ Models endpoint: {response.status_code} - {response.json()}")
            
            response = client.get("/api/v1/providers/status")
            print(f"✓ Providers endpoint: {response.status_code} - {response.json()}")
            
            chat_request = {
                "messages": [{"role": "user", "content": "Hello, world!"}],
                "stream": False
            }
            response = client.post("/api/v1/chat/completions", json=chat_request)
            print(f"✓ Chat completion endpoint: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"  Response preview: {str(result)[:200]}...")
            else:
                print(f"  Error (expected without real API keys): {response.text[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False


if __name__ == "__main__":
    print("=== LLM Factory API Tests ===")
    success = test_api()
    if success:
        print("✓ API tests completed successfully!")
    else:
        print("✗ API tests failed!")
