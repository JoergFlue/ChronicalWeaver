"""LLM Configuration Management"""
from dataclasses import dataclass
from typing import Dict, Optional, Any
import json
from pathlib import Path

@dataclass
class LLMConfig:
    """
    Configuration for a specific LLM provider.

    Attributes:
        provider (str): Name of the LLM provider.
        model (str): Model name or identifier.
        api_key (Optional[str]): API key for authentication.
        base_url (Optional[str]): Base URL for API requests.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum number of tokens in response.
        timeout (int): Timeout for API requests (seconds).
        stream (bool): Whether to stream responses.
    """
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30
    stream: bool = True

class LLMConfigManager:
    """
    Manages LLM configurations for different providers.

    Attributes:
        config_path (Path): Path to the configuration file.
        configs (Dict[str, LLMConfig]): Loaded configurations.
    """

    def __init__(self, config_path: str = "config/llm_configs.json"):
        """
        Initialize LLMConfigManager.

        Args:
            config_path (str): Path to the configuration file.
        """
        self.config_path = Path(config_path)
        self.configs = self._load_configs()

    def _load_configs(self) -> Dict[str, LLMConfig]:
        """
        Load LLM configurations from file.

        Returns:
            Dict[str, LLMConfig]: Dictionary of provider names to LLMConfig objects.
        """
        if not self.config_path.exists():
            return self._create_default_configs()

        with open(self.config_path, 'r') as f:
            data = json.load(f)

        return {
            name: LLMConfig(**config)
            for name, config in data.items()
        }

    def _create_default_configs(self) -> Dict[str, LLMConfig]:
        """
        Create default LLM configurations and save them.

        Returns:
            Dict[str, LLMConfig]: Default configurations.
        """
        defaults = {
            "openai": LLMConfig(
                provider="openai",
                model="gpt-4",
                temperature=0.7
            ),
            "gemini": LLMConfig(
                provider="gemini",
                model="gemini-pro",
                temperature=0.7
            ),
            "ollama": LLMConfig(
                provider="ollama",
                model="",
                base_url="http://localhost:11434",
                temperature=0.7
            ),
            "lm_studio": LLMConfig(
                provider="lm_studio",
                model="",
                base_url="http://localhost:1234",
                temperature=0.7
            )
        }
        self._save_configs(defaults)
        return defaults

    def get_config(self, provider_name: str) -> Optional[LLMConfig]:
        """
        Get configuration for a specific provider.

        Args:
            provider_name (str): Name of the provider.

        Returns:
            Optional[LLMConfig]: Configuration object if found, else None.
        """
        return self.configs.get(provider_name)

    def update_config(self, provider_name: str, config: LLMConfig):
        """
        Update configuration for a provider and save to file.

        Args:
            provider_name (str): Name of the provider.
            config (LLMConfig): Configuration object.
        """
        self.configs[provider_name] = config
        self._save_configs(self.configs)

    def _save_configs(self, configs: Dict[str, LLMConfig]):
        """
        Save configurations to file.

        Args:
            configs (Dict[str, LLMConfig]): Dictionary of configurations.
        """
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {
            name: {
                "provider": config.provider,
                "model": config.model,
                "api_key": config.api_key,
                "base_url": config.base_url,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "timeout": config.timeout,
                "stream": config.stream
            }
            for name, config in configs.items()
        }

        with open(self.config_path, 'w') as f:
            json.dump(serializable, f, indent=2)
