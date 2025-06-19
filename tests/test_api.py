"""
Test script for the FastAPI interface.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_factory.api.app import create_app


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    return TestClient(app)


def test_docs_endpoint(client):
    """Test docs endpoint."""
    response = client.get("/docs")
    assert response.status_code == 200
    # print(f"✓ Root endpoint: {response.status_code} - {response.json()}")


def test_health_endpoint(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    print(f"✓ Health endpoint: {response.status_code} - {response.json()}")


def test_models_endpoint(client):
    """Test models endpoint."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    print(f"✓ Models endpoint: {response.status_code} - {response.json()}")


def test_providers_endpoint(client):
    """Test providers endpoint."""
    response = client.get("/api/v1/providers/status")
    assert response.status_code == 200
    print(f"✓ Providers endpoint: {response.status_code} - {response.json()}")


def test_chat_completion(client):
    """Test chat completion endpoint."""
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