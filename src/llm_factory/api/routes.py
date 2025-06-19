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

router = APIRouter()

_factory: Optional[LLMFactory] = None


class FactoryConfig(BaseModel):
    """Configuration for initializing the factory."""
    providers: List[Dict[str, Any]]


def get_factory() -> LLMFactory:
    """Get or create the global factory instance."""
    global _factory
    
    if _factory is None:
        configs = []
        
        # OpenAI配置
        openai_keys = [k.strip() for k in os.getenv("OPENAI_API_KEYS", "").split(",") if k.strip()]
        if not openai_keys and os.getenv("OPENAI_API_KEY"):  # 兼容单个key的情况
            openai_keys = [os.getenv("OPENAI_API_KEY")]
        
        for api_key in openai_keys:
            configs.append(ModelConfig(
                provider=ProviderType.OPENAI,
                model_name=os.getenv("OPENAI_MODEL", "gpt-4"),
                api_key=api_key,
                api_base=os.getenv("OPENAI_API_BASE"),
                api_version=os.getenv("OPENAI_API_VERSION"),
            ))
        
        # Qwen配置
        qwen_keys = [k.strip() for k in os.getenv("QWEN_API_KEYS", "").split(",") if k.strip()]
        if not qwen_keys and os.getenv("QWEN_API_KEY"):
            qwen_keys = [os.getenv("QWEN_API_KEY")]
            
        for api_key in qwen_keys:
            configs.append(ModelConfig(
                provider=ProviderType.QWEN,
                model_name=os.getenv("QWEN_MODEL", "qwen-turbo"),
                api_key=api_key,
                api_base=os.getenv("QWEN_API_BASE"),
            ))
        
        # Deepseek配置
        deepseek_keys = [k.strip() for k in os.getenv("DEEPSEEK_API_KEYS", "").split(",") if k.strip()]
        if not deepseek_keys and os.getenv("DEEPSEEK_API_KEY"):
            deepseek_keys = [os.getenv("DEEPSEEK_API_KEY")]
            
        for api_key in deepseek_keys:
            configs.append(ModelConfig(
                provider=ProviderType.DEEPSEEK,
                model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                api_key=api_key,
                api_base=os.getenv("DEEPSEEK_API_BASE"),
            ))
        
        # Claude配置
        claude_keys = [k.strip() for k in os.getenv("CLAUDE_ACCESS_KEYS", "").split(",") if k.strip()]
        if not claude_keys and os.getenv("CLAUDE_ACCESS_KEY"):
            claude_keys = [os.getenv("CLAUDE_ACCESS_KEY")]
            
        for api_key in claude_keys:
            configs.append(ModelConfig(
                provider=ProviderType.CLAUDE,
                model_name=os.getenv("CLAUDE_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
                api_key=api_key,
                api_base=os.getenv("CLAUDE_SECRET_KEY"),
                region=os.getenv("CLAUDE_REGION", "us-east-1"),
            ))
        
        # Gemini配置
        gemini_keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
        if not gemini_keys and os.getenv("GEMINI_API_KEY"):
            gemini_keys = [os.getenv("GEMINI_API_KEY")]
            
        for api_key in gemini_keys:
            configs.append(ModelConfig(
                provider=ProviderType.GEMINI,
                model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
                api_key=api_key,
                project_id=os.getenv("GEMINI_PROJECT_ID"),
                region=os.getenv("GEMINI_REGION"),
            ))
        
        if not configs:
            raise HTTPException(
                status_code=500,
                detail="No LLM providers configured. Please set environment variables."
            )
        
        _factory = LLMFactory(configs)
        logger.info(f"Initialized factory with {len(configs)} providers")
    
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
