"""LLM Management and Integration"""
import logging
from typing import Optional, Dict, Any
from .llm_config import LLMConfigManager, LLMConfig
from .host_connection import HostConnection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class LLMManager:
    """
    Manages multiple LLM providers using LiteLLM and HostConnection.

    Attributes:
        config_manager (LLMConfigManager): Manages LLM provider configurations.
        host_connection (HostConnection): Manages host selection and connection.
        current_provider (str): Currently selected provider name.
        current_model (str): Currently selected model name.
    """

    def __init__(self):
        """
        Initialize LLMManager.
        """
        self.config_manager = LLMConfigManager()
        self.host_connection = HostConnection()
        self.current_provider = ""
        self.current_model = ""
        # self._setup_litellm()  # LiteLLM disabled for testing

    def list_models(self) -> list:
        """
        List available models for the selected host using the correct API endpoint.

        Returns:
            list: List of available model names.
        """
        import requests
        host_name = self.host_connection.selected_host
        hosts = self.host_connection.hosts
        host_info = next((h for h in hosts if h.get("name") == host_name), None)
        if not host_info:
            logger.info("No host selected or host info missing.")
            return []
        host_url = host_info.get("host_url", "")
        host_type = host_info.get("type", "").lower()
        try:
            if "ollama" in host_type or "ollama" in host_name.lower():
                url = f"{host_url}/api/tags"
                logger.debug(f"Ollama model listing: GET {url}")
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                models = [m.get("name", m.get("model", "")) for m in data.get("models", [])]
                logger.info(f"Ollama models found: {models}")
                return models
            else:
                endpoint_url = host_info.get("endpoint_url", "")
                if not host_url or not endpoint_url:
                    logger.info("Host URL or endpoint URL missing.")
                    return []
                url = host_url + endpoint_url
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if "models" in data:
                    models = [m.get("name", m.get("id", "")) for m in data["models"]]
                elif "data" in data:
                    models = [m.get("id", m.get("name", "")) for m in data["data"]]
                else:
                    models = [str(m) for m in data.get("models", [])]
                logger.info(f"Models found: {models}")
                return models
        except Exception as e:
            logger.error(f"Model listing error for host {host_name}: {e}")
            return []

    # def _setup_litellm(self):
    #     """
    #     Configure LiteLLM settings.
    #     """
    #     litellm.set_verbose = False  # Disable verbose logging
    #     litellm.drop_params = True   # Drop unsupported parameters

    def set_host(self, host_name: str) -> bool:
        """
        Switch to a different host.

        Args:
            host_name (str): Name of the host to select.

        Returns:
            bool: True if switched successfully.
        """
        self.host_connection.set_host(host_name)
        self.current_provider = host_name
        logger.info(f"Switched to host: {host_name}")
        logger.debug(f"set_host: current_provider set to {self.current_provider}")
        return True

    def get_current_config(self) -> Optional[LLMConfig]:
        """
        Get the current host/model configuration.

        Returns:
            Optional[LLMConfig]: Configuration object if found, else None.
        """
        # Map common host names to config keys
        host_map = {
            "Local Ollama": "ollama",
            "Ollama": "ollama",
            "LM Studio": "lm_studio",
            "LMStudio": "lm_studio",
            "Local LM Studio": "lm_studio"
        }
        config_key = host_map.get(self.current_provider, self.current_provider.lower().replace(" ", "_"))
        logger.debug(f"get_current_config: current_provider={self.current_provider}, config_key={config_key}")
        return self.config_manager.get_config(config_key)

    def generate_response(self, prompt: str) -> str:
        """
        Generate response from the current LLM provider using direct HTTP requests.

        Args:
            prompt (str): Prompt to send to the model.

        Returns:
            str: Model response.
        """
        import requests
        config = self.get_current_config()
        if not config:
            raise ValueError(f"No configuration for provider: {self.current_provider}")

        logger.debug(f"Selected provider: {self.current_provider}, config: {config}")

        if config.provider == "ollama":
            url = f"{config.base_url}/api/generate"
            payload = {"model": config.model, "prompt": prompt}
            resp = requests.post(url, json=payload, timeout=30, stream=True)
            resp.raise_for_status()
            import json
            response_text = ""
            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        response_text += chunk.get("response", "")
                    except Exception as e:
                        logger.error(f"Ollama response parse error: {e}")
            return response_text
        elif config.provider in ["lm_studio", "LM Studio"]:
            url = f"{config.base_url}/v1/chat/completions"
            payload = {
                "model": config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config.temperature,
                "max_tokens": config.max_tokens
            }
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices and "message" in choices[0]:
                return choices[0]["message"].get("content", "")
            return ""
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    # LiteLLM and streaming logic removed for simplicity

    def load_model(self, model_name: str) -> str:
        """
        Load model using the correct API for the host type.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            str: Status message.
        """
        host_name = self.host_connection.selected_host
        hosts = self.host_connection.hosts
        host_info = next((h for h in hosts if h.get("name") == host_name), None)
        if not host_info:
            logger.error("No host info found for provider: %s", host_name)
            return "Host info missing"
        host_type = host_info.get("type", "").lower()
        if "ollama" in host_type or "ollama" in host_name.lower():
            return self._load_model_ollama(host_info, model_name)
        elif "lmstudio" in host_type or "lmstudio" in host_name.lower():
            return self._load_model_lmstudio(host_info, model_name)
        else:
            logger.warning(f"Unknown host type for {host_name}, defaulting to LM Studio API.")
            return self._load_model_lmstudio(host_info, model_name)

    def _load_model_lmstudio(self, host_info, model_name: str) -> str:
        """
        Load model for LM Studio using GET /v1/models?model={model_name}.

        Args:
            host_info (dict): Host information.
            model_name (str): Name of the model to load.

        Returns:
            str: Status message.
        """
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
        """
        Load model for Ollama by POSTing to /api/generate with an empty prompt.

        Args:
            host_info (dict): Host information.
            model_name (str): Name of the model to load.

        Returns:
            str: Status message.
        """
        import requests
        host_url = host_info.get("host_url", "")
        url = f"{host_url}/api/generate"
        payload = {"model": model_name}
        try:
            logger.info(f"Ollama: loading model '{model_name}' via POST {url}")
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            self.current_model = model_name
            if data.get("done", False):
                logger.info(f"Ollama model '{model_name}' loaded successfully.")
                return f"Ollama model set: {model_name}"
            else:
                logger.warning(f"Ollama model '{model_name}' load did not complete: {data}")
                return f"Ollama model load incomplete: {data}"
        except Exception as e:
            logger.error(f"Ollama model load failed for {model_name}: {e}", exc_info=True)
            return f"Ollama model load failed: {str(e)}"

    def test_connection(self, provider_name: str) -> bool:
        """
        Test connection to a specific provider.

        For Ollama, POST to /api/generate with an empty prompt and check for success.

        Args:
            provider_name (str): Name of the provider to test.

        Returns:
            bool: True if connection is successful, False otherwise.
        """
        import requests
        hosts = self.host_connection.hosts
        host_info = next((h for h in hosts if h.get("name") == provider_name), None)
        if not host_info:
            logger.info("No host info found for provider: %s", provider_name)
            return False
        host_url = host_info.get("host_url", "")
        host_type = host_info.get("type", "").lower()
        try:
            if "ollama" in host_type or "ollama" in provider_name.lower():
                tags_url = f"{host_url}/api/tags"
                try:
                    tags_resp = requests.get(tags_url, timeout=5)
                    tags_resp.raise_for_status()
                    tags_data = tags_resp.json()
                    models = [m.get("name", m.get("model", "")) for m in tags_data.get("models", [])]
                    if not models:
                        logger.error(f"Ollama connection test: No models found at {tags_url}")
                        return False
                    model_name = models[0]
                except Exception as e:
                    logger.error(f"Ollama connection test: Failed to fetch models: {e}")
                    return False
                url = f"{host_url}/api/generate"
                payload = {"model": model_name, "prompt": ""}
                logger.debug(f"Ollama connection test: POST {url} with model {model_name}")
                resp = requests.post(url, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                return data.get("done", False)
            else:
                url = host_url + host_info.get("endpoint_url", "")
                logger.debug(f"Connection attempt: GET {url}")
                resp = requests.get(url, timeout=3)
                logger.debug(f"Response status: {resp.status_code}")
                logger.debug(f"Response headers: {resp.headers}")
                logger.debug(f"Response content: {resp.text}")
                resp.raise_for_status()
                data = resp.json()
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
