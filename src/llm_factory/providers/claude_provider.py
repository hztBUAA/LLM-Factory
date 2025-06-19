"""
Claude provider implementation using AWS Bedrock.
"""

import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import boto3
from botocore.config import Config
from loguru import logger

from ..models import ChatMessage, ChatResponse, ModelConfig, StreamChunk, Usage
from .base import BaseProvider


class ClaudeProvider(BaseProvider):
    """Claude provider using AWS Bedrock."""
    
    def _setup_client(self) -> None:
        """Setup AWS Bedrock client."""
        boto_config = Config(
            connect_timeout=self.config.timeout,
            read_timeout=self.config.timeout,
            retries={'max_attempts': self.config.max_retries},
        )
        
        if self.config.proxy_config:
            boto_config = Config(
                connect_timeout=self.config.timeout,
                read_timeout=self.config.timeout,
                retries={'max_attempts': self.config.max_retries},
                proxies=self.config.proxy_config
            )
        
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.config.region or 'us-east-1',
            aws_access_key_id=self.config.api_key,
            aws_secret_access_key=self.config.api_base,
            config=boto_config
        )
    
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        **kwargs: Any
    ) -> Union[ChatResponse, AsyncGenerator[StreamChunk, None]]:
        """Generate a chat completion."""
        if kwargs.get("stream", self.config.stream):
            return self.chat_completion_stream(messages, **kwargs)
        
        try:
            system_messages = []
            user_messages = []
            
            for msg in messages:
                if msg.role.value == "system":
                    system_messages.append({
                        "type": "text",
                        "text": msg.content
                    })
                else:
                    user_messages.append({
                        "role": msg.role.value,
                        "content": msg.content
                    })
            
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": user_messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens or 4096),
                "temperature": kwargs.get("temperature", self.config.temperature or 0.1),
            }
            
            if system_messages:
                request_body["system"] = system_messages
            
            if "claude-3-7-sonnet" in self.config.model_name:
                request_body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 16000
                }
                request_body["max_tokens"] = 24000
            
            response = self.client.invoke_model(
                modelId=self.config.model_name,
                body=json.dumps(request_body)
            )
            
            result = json.loads(response['body'].read())
            
            usage = Usage(
                prompt_tokens=result.get("usage", {}).get("input_tokens", 0),
                completion_tokens=result.get("usage", {}).get("output_tokens", 0),
                total_tokens=result.get("usage", {}).get("input_tokens", 0) + result.get("usage", {}).get("output_tokens", 0),
                cost=self._calculate_cost(
                    result.get("usage", {}).get("input_tokens", 0),
                    result.get("usage", {}).get("output_tokens", 0),
                    self.config.model_name
                )
            )
            
            choices = [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.get("content", [{}])[0].get("text", "")
                },
                "finish_reason": result.get("stop_reason", "stop")
            }]
            
            return ChatResponse(
                id=self._generate_id(),
                created=self._get_current_timestamp(),
                model=self.config.model_name,
                choices=choices,
                usage=usage
            )
            
        except Exception as e:
            logger.error(f"Claude completion error: {e}")
            raise
    
    async def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        **kwargs: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate a streaming chat completion."""
        try:
            system_messages = []
            user_messages = []
            
            for msg in messages:
                if msg.role.value == "system":
                    system_messages.append({
                        "type": "text",
                        "text": msg.content
                    })
                else:
                    user_messages.append({
                        "role": msg.role.value,
                        "content": msg.content
                    })
            
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": user_messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens or 4096),
                "temperature": kwargs.get("temperature", self.config.temperature or 0.1),
            }
            
            if system_messages:
                request_body["system"] = system_messages
            
            if "claude-3-7-sonnet" in self.config.model_name:
                request_body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 16000
                }
                request_body["max_tokens"] = 24000
            
            response = self.client.invoke_model_with_response_stream(
                modelId=self.config.model_name,
                body=json.dumps(request_body)
            )
            
            stream = response.get('body')
            if stream:
                for event in stream:
                    chunk = event.get("chunk")
                    if chunk:
                        chunk_json = json.loads(chunk.get("bytes").decode())
                        
                        usage = None
                        if chunk_json.get("usage"):
                            usage_info = chunk_json["usage"]
                            usage = Usage(
                                prompt_tokens=usage_info.get("input_tokens", 0),
                                completion_tokens=usage_info.get("output_tokens", 0),
                                total_tokens=usage_info.get("input_tokens", 0) + usage_info.get("output_tokens", 0),
                                cost=self._calculate_cost(
                                    usage_info.get("input_tokens", 0),
                                    usage_info.get("output_tokens", 0),
                                    self.config.model_name
                                )
                            )
                        
                        choices = []
                        if chunk_json.get("type") == "content_block_delta":
                            choices = [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": chunk_json.get("delta", {}).get("text", "")
                                },
                                "finish_reason": None
                            }]
                        elif chunk_json.get("type") == "message_stop":
                            choices = [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        
                        if choices:
                            yield StreamChunk(
                                id=self._generate_id(),
                                created=self._get_current_timestamp(),
                                model=self.config.model_name,
                                choices=choices,
                                usage=usage
                            )
                            
        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            raise
    
    def _get_input_cost_per_1k(self, model: str) -> Optional[float]:
        """Get input cost per 1K tokens for Claude models."""
        costs = {
            "anthropic.claude-3-5-sonnet-20241022-v2:0": 0.003,
            "anthropic.claude-3-5-haiku-20241022-v1:0": 0.0008,
            "anthropic.claude-3-opus-20240229-v1:0": 0.015,
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0": 0.003,
        }
        return costs.get(model.lower())
    
    def _get_output_cost_per_1k(self, model: str) -> Optional[float]:
        """Get output cost per 1K tokens for Claude models."""
        costs = {
            "anthropic.claude-3-5-sonnet-20241022-v2:0": 0.015,
            "anthropic.claude-3-5-haiku-20241022-v1:0": 0.004,
            "anthropic.claude-3-opus-20240229-v1:0": 0.075,
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0": 0.015,
        }
        return costs.get(model.lower())
