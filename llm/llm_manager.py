"""LLM Management and Integration"""
import asyncio
import logging
from typing import AsyncGenerator, Optional, Dict, Any
import litellm
from .llm_config import LLMConfigManager, LLMConfig

logger = logging.getLogger(__name__)

class LLMManager:
    """Manages multiple LLM providers using LiteLLM"""

    def __init__(self):
        self.config_manager = LLMConfigManager()
        self.current_provider = "openai"  # Default
        self._setup_litellm()

    def _setup_litellm(self):
        """Configure LiteLLM settings"""
        litellm.set_verbose = False  # Disable verbose logging
        litellm.drop_params = True   # Drop unsupported parameters

    def set_provider(self, provider_name: str) -> bool:
        """Switch to a different LLM provider"""
        if provider_name not in self.config_manager.configs:
            logger.error(f"Provider {provider_name} not configured")
            return False

        self.current_provider = provider_name
        logger.info(f"Switched to provider: {provider_name}")
        return True

    def get_current_config(self) -> Optional[LLMConfig]:
        """Get the current provider configuration"""
        return self.config_manager.get_config(self.current_provider)

    async def generate_response(
        self, 
        messages: list, 
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Generate response from current LLM provider"""
        config = self.get_current_config()
        if not config:
            raise ValueError(f"No configuration for provider: {self.current_provider}")

        try:
            # Prepare LiteLLM parameters
            llm_params = self._prepare_llm_params(config, messages, stream)

            if stream:
                async for chunk in self._stream_response(llm_params):
                    yield chunk
            else:
                response = await self._get_complete_response(llm_params)
                yield response

        except Exception as e:
            logger.error(f"LLM generation error: {str(e)}")
            raise

    def _prepare_llm_params(self, config: LLMConfig, messages: list, stream: bool) -> Dict[str, Any]:
        """Prepare parameters for LiteLLM call"""
        params = {
            "model": f"{config.provider}/{config.model}",
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": stream
        }

        # Add provider-specific parameters
        if config.api_key:
            params["api_key"] = config.api_key
        if config.base_url:
            params["api_base"] = config.base_url

        return params

    async def _stream_response(self, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream response from LLM"""
        try:
            response = await litellm.acompletion(**params)
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            raise

    async def _get_complete_response(self, params: Dict[str, Any]) -> str:
        """Get complete response from LLM"""
        params["stream"] = False
        response = await litellm.acompletion(**params)
        return response.choices[0].message.content

    def test_connection(self, provider_name: str) -> bool:
        """Test connection to a specific provider"""
        try:
            config = self.config_manager.get_config(provider_name)
            if not config:
                return False

            # Simple test message
            test_messages = [{"role": "user", "content": "Hello"}]
            params = self._prepare_llm_params(config, test_messages, False)

            # Synchronous test call
            response = litellm.completion(**params)
            return bool(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Connection test failed for {provider_name}: {str(e)}")
            return False
