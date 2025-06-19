"""
API routes for the LLM Factory.
"""

import os
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from ..factory import LLMFactory
from ..models import ChatCompletionRequest, ChatMessage, ModelConfig
from ..providers import ProviderType

from dotenv import load_dotenv
load_dotenv()

router = APIRouter()

_factory: Optional[LLMFactory] = None


class FactoryConfig(BaseModel):
    """Configuration for initializing the factory."""
    providers: List[Dict[str, Any]]


def get_factory() -> LLMFactory:
    """Get or create the global factory instance."""
    global _factory
    
    if _factory is None:
        _factory = LLMFactory.create()
    
    return _factory


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Create a chat completion, optionally streaming.
    
    Compatible with OpenAI API format.
    """
    try:
        factory = get_factory()
        
        if request.stream:
            async def generate():
                async for chunk in factory.stream_async(request.messages, **request.dict(exclude={"messages", "model"})):
                    yield f"data: {chunk.json()}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            response = await factory.chat_async(request.messages, **request.dict(exclude={"messages", "model"}))
            return response
            
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """List available models."""
    try:
        factory = get_factory()
        status = factory.get_provider_status()
        
        models = []
        for provider in status["providers"]:
            models.append({
                "id": provider["model"],
                "object": "model",
                "created": 1677610602,
                "owned_by": provider["provider"],
                "provider_type": provider["type"],
            })
        
        return {"object": "list", "data": models}
        
    except Exception as e:
        logger.error(f"List models error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/providers/status")
async def provider_status():
    """Get status of all providers."""
    try:
        factory = get_factory()
        return factory.get_provider_status()
        
    except Exception as e:
        logger.error(f"Provider status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/factory/configure")
async def configure_factory(config: FactoryConfig):
    """
    Reconfigure the factory with new providers.
    
    This will replace the current factory instance.
    """
    global _factory
    
    try:
        configs = []
        for provider_config in config.providers:
            configs.append(ModelConfig(**provider_config))
        
        _factory = LLMFactory(configs)
        logger.info(f"Reconfigured factory with {len(configs)} providers")
        
        return {"message": "Factory reconfigured successfully", "providers": len(configs)}
        
    except Exception as e:
        logger.error(f"Factory configuration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
