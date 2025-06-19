"""
Gemini provider implementation.
"""

import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from loguru import logger

from ..models import ChatMessage, ChatResponse, ModelConfig, StreamChunk, Usage
from .base import BaseProvider

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    logger.warning("Google Generative AI library not available. Gemini provider will be disabled.")
    GEMINI_AVAILABLE = False
    genai = None
    types = None


class GeminiProvider(BaseProvider):
    """Gemini provider implementation."""
    
    def _setup_client(self) -> None:
        """Setup Gemini client."""
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI library is not installed. Please install it with: pip install google-generativeai")
        
        if self.config.project_id:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.config.api_key or "config/service-account.json"
            
            self.client = genai.Client(
                vertexai=True,
                project=self.config.project_id,
                location=self.config.region or "us-central1",
            )
        else:
            self.client = genai.Client(api_key=self.config.api_key)
    
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        **kwargs: Any
    ) -> Union[ChatResponse, AsyncGenerator[StreamChunk, None]]:
        """Generate a chat completion."""
        if kwargs.get("stream", self.config.stream):
            return self.chat_completion_stream(messages, **kwargs)
        
        try:
            content = ""
            for msg in messages:
                if msg.role.value == "system":
                    content = f"System: {msg.content}\n" + content
                else:
                    content += f"{msg.role.value.title()}: {msg.content}\n"
            
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content)]
                )
            ]
            
            generate_config = types.GenerateContentConfig(
                temperature=kwargs.get("temperature", self.config.temperature or 1.0),
                top_p=kwargs.get("top_p", self.config.top_p or 1.0),
                max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens or 8192),
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                ],
            )
            
            if kwargs.get("response_format") and kwargs["response_format"].get("type") == "json_object":
                generate_config.response_mime_type = "application/json"
            
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=generate_config,
            )
            
            usage = Usage(
                prompt_tokens=response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                completion_tokens=response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
                total_tokens=response.usage_metadata.total_token_count if response.usage_metadata else 0,
                cost=self._calculate_cost(
                    response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                    response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
                    self.config.model_name
                )
            )
            
            choices = [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.text
                },
                "finish_reason": "stop"
            }]
            
            return ChatResponse(
                id=self._generate_id(),
                created=self._get_current_timestamp(),
                model=self.config.model_name,
                choices=choices,
                usage=usage
            )
            
        except Exception as e:
            logger.error(f"Gemini completion error: {e}")
            raise
    
    async def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        **kwargs: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate a streaming chat completion."""
        try:
            content = ""
            for msg in messages:
                if msg.role.value == "system":
                    content = f"System: {msg.content}\n" + content
                else:
                    content += f"{msg.role.value.title()}: {msg.content}\n"
            
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content)]
                )
            ]
            
            generate_config = types.GenerateContentConfig(
                temperature=kwargs.get("temperature", self.config.temperature or 1.0),
                top_p=kwargs.get("top_p", self.config.top_p or 1.0),
                max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens or 8192),
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                ],
            )
            
            if kwargs.get("response_format") and kwargs["response_format"].get("type") == "json_object":
                generate_config.response_mime_type = "application/json"
            
            stream = self.client.models.generate_content_stream(
                model=self.config.model_name,
                contents=contents,
                config=generate_config,
            )
            
            for chunk in stream:
                if chunk.text:
                    choices = [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": chunk.text
                        },
                        "finish_reason": None
                    }]
                    
                    usage = None
                    if chunk.usage_metadata:
                        usage = Usage(
                            prompt_tokens=chunk.usage_metadata.prompt_token_count,
                            completion_tokens=chunk.usage_metadata.candidates_token_count,
                            total_tokens=chunk.usage_metadata.total_token_count,
                            cost=self._calculate_cost(
                                chunk.usage_metadata.prompt_token_count,
                                chunk.usage_metadata.candidates_token_count,
                                self.config.model_name
                            )
                        )
                    
                    yield StreamChunk(
                        id=self._generate_id(),
                        created=self._get_current_timestamp(),
                        model=self.config.model_name,
                        choices=choices,
                        usage=usage
                    )
            
            yield StreamChunk(
                id=self._generate_id(),
                created=self._get_current_timestamp(),
                model=self.config.model_name,
                choices=[{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }],
                usage=None
            )
                            
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise
    
    def _get_input_cost_per_1k(self, model: str) -> Optional[float]:
        """Get input cost per 1K tokens for Gemini models."""
        costs = {
            "gemini-2.0-flash-exp": 0.000075,
            "gemini-1.5-pro": 0.00125,
            "gemini-1.5-flash": 0.000075,
            "gemini-1.0-pro": 0.0005,
        }
        return costs.get(model.lower())
    
    def _get_output_cost_per_1k(self, model: str) -> Optional[float]:
        """Get output cost per 1K tokens for Gemini models."""
        costs = {
            "gemini-2.0-flash-exp": 0.0003,
            "gemini-1.5-pro": 0.005,
            "gemini-1.5-flash": 0.0003,
            "gemini-1.0-pro": 0.0015,
        }
        return costs.get(model.lower())
