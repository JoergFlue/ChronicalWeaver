"""LLM Management and Integration"""
import asyncio
import logging
from typing import AsyncGenerator, Optional, Dict, Any
import litellm
from .llm_config import LLMConfigManager, LLMConfig
from .host_connection import HostConnection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class LLMManager:
    """Manages multiple LLM providers using LiteLLM and HostConnection"""

    def __init__(self):
        self.config_manager = LLMConfigManager()
        self.host_connection = HostConnection()
        self.current_provider = ""
        self.current_model = ""
        # No provider/model loaded at start
        self._setup_litellm()

    def list_models(self) -> list:
        """List available models for the selected host using endpoint_url"""
        import requests, json, os
        host_name = self.host_connection.selected_host
        hosts = self.host_connection.hosts
        host_info = next((h for h in hosts if h.get("name") == host_name), None)
        if not host_info:
            logger.info("No host selected or host info missing.")
            return []
        host_url = host_info.get("host_url", "")
        endpoint_url = host_info.get("endpoint_url", "")
        if not host_url or not endpoint_url:
            logger.info("Host URL or endpoint URL missing.")
            return []
        url = host_url + endpoint_url
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()
            # Generic extraction: try common keys
            if "models" in data:
                return [m.get("name", m.get("id", "")) for m in data["models"]]
            if "data" in data:
                return [m.get("id", m.get("name", "")) for m in data["data"]]
            # Fallback: try to extract any string values
            return [str(m) for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Model listing error for host {host_name}: {e}")
            return []

    def _setup_litellm(self):
        """Configure LiteLLM settings"""
        litellm.set_verbose = False  # Disable verbose logging
        litellm.drop_params = True   # Drop unsupported parameters

    def set_host(self, host_name: str) -> bool:
        """Switch to a different host"""
        self.host_connection.set_host(host_name)
        self.current_provider = host_name
        logger.info(f"Switched to host: {host_name}")
        return True

    def get_current_config(self) -> Optional[LLMConfig]:
        """Get the current host/model configuration"""
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

    def load_model(self, model_name: str) -> str:
        """Load model using the correct API for the host type"""
        host_name = self.host_connection.selected_host
        hosts = self.host_connection.hosts
        host_info = next((h for h in hosts if h.get("name") == host_name), None)
        if not host_info:
            logger.error("No host info found for provider: %s", host_name)
            return "Host info missing"
        # Detect host type
        host_type = host_info.get("type", "").lower()
        if "ollama" in host_type or "ollama" in host_name.lower():
            return self._load_model_ollama(host_info, model_name)
        elif "lmstudio" in host_type or "lmstudio" in host_name.lower():
            return self._load_model_lmstudio(host_info, model_name)
        else:
            logger.warning(f"Unknown host type for {host_name}, defaulting to LM Studio API.")
            return self._load_model_lmstudio(host_info, model_name)

    def _load_model_lmstudio(self, host_info, model_name: str) -> str:
        """Load model for LM Studio using GET /v1/models?model={model_name}"""
        import requests
        host_url = host_info.get("host_url", "")
        endpoint = host_info.get("endpoint_url", "/v1/models")
        url = f"{host_url}{endpoint}?model={model_name}"
        try:
            logger.info(f"Attempting to load model '{model_name}' on LM Studio host via GET {url}")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"Model load response: {data}")
            self.current_model = model_name
            return data.get("message", "Model loaded successfully")
        except Exception as e:
            logger.error(f"Model load failed for {model_name} on LM Studio: {e}", exc_info=True)
            return f"Model load failed: {str(e)}"

    def _load_model_ollama(self, host_info, model_name: str) -> str:
        """Load model for Ollama (just set current model, no API call needed)"""
        self.current_model = model_name
        logger.info(f"Ollama: set current model to '{model_name}' (no API call required)")
        return f"Ollama model set: {model_name}"

    def test_connection(self, provider_name: str) -> bool:
        """Test connection to a specific provider using sample script logic"""
        import requests
        hosts = self.host_connection.hosts
        host_info = next((h for h in hosts if h.get("name") == provider_name), None)
        if not host_info:
            logger.info("No host info found for provider: %s", provider_name)
            return False
        host_url = host_info.get("host_url", "")
        # Generic connection test using host_url and endpoint_url from config
        try:
            url = host_url + host_info.get("endpoint_url", "")
            logger.debug(f"Connection attempt: GET {url}")
            resp = requests.get(url, timeout=3)
            logger.debug(f"Response status: {resp.status_code}")
            logger.debug(f"Response headers: {resp.headers}")
            logger.debug(f"Response content: {resp.text}")
            resp.raise_for_status()
            data = resp.json()
            # Try common keys for models
            models = []
            if "models" in data:
                models = [m.get("name", m.get("id", "")) for m in data["models"]]
            elif "data" in data:
                models = [m.get("id", m.get("name", "")) for m in data["data"]]
            logger.debug(f"Models found: {models}")
            return bool(models)
        except Exception as e:
            logger.error(f"Connection test failed for {provider_name}: {e}", exc_info=True)
            return False
