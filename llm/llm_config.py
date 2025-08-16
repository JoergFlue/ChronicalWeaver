"""LLM Configuration Management"""
from dataclasses import dataclass
from typing import Dict, Optional, Any
import json
from pathlib import Path

@dataclass
class LLMConfig:
    """Configuration for a specific LLM provider"""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30
    stream: bool = True

class LLMConfigManager:
    """Manages LLM configurations for different providers"""

    def __init__(self, config_path: str = "config/llm_configs.json"):
        self.config_path = Path(config_path)
        self.configs = self._load_configs()

    def _load_configs(self) -> Dict[str, LLMConfig]:
        """Load LLM configurations from file"""
        if not self.config_path.exists():
            return self._create_default_configs()

        with open(self.config_path, 'r') as f:
            data = json.load(f)

        return {
            name: LLMConfig(**config)
            for name, config in data.items()
        }

    def _create_default_configs(self) -> Dict[str, LLMConfig]:
        """Create default LLM configurations"""
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
                model="llama2",
                base_url="http://localhost:11434",
                temperature=0.7
            ),
            "lm_studio": LLMConfig(
                provider="openai",
                model="local-model",
                base_url="http://localhost:1234/v1",
                temperature=0.7
            )
        }
        self._save_configs(defaults)
        return defaults

    def get_config(self, provider_name: str) -> Optional[LLMConfig]:
        """Get configuration for a specific provider"""
        return self.configs.get(provider_name)

    def update_config(self, provider_name: str, config: LLMConfig):
        """Update configuration for a provider"""
        self.configs[provider_name] = config
        self._save_configs(self.configs)

    def _save_configs(self, configs: Dict[str, LLMConfig]):
        """Save configurations to file"""
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
