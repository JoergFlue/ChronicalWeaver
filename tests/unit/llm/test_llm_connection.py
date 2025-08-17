import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from llm.llm_manager import LLMManager

def test_llm_provider_connections():
    providers = ["openai", "gemini", "ollama", "lm_studio"]
    results = {}
    successful_models = {}

    for provider in providers:
        manager = LLMManager()
        # Update config for local providers
        config = manager.config_manager.get_config(provider)
        if config:
            if provider == "ollama":
                config.provider = "ollama"
                config.model = "llama2"
                config.base_url = "http://127.0.0.1:11434"
            elif provider == "lm_studio":
                config.provider = "lm_studio"
                config.model = "local-model"
                config.base_url = "http://127.0.0.1:1234"
            manager.config_manager.update_config(provider, config)
        connected = manager.set_provider(provider)
        if connected:
            try:
                models = manager.list_models()
                results[provider] = True if models else False
                if models:
                    successful_models[provider] = models
            except Exception as e:
                print(f"{provider} error: {e}")
                results[provider] = False
        else:
            results[provider] = False

    # Print status for each provider
    for provider, status in results.items():
        print(f"{provider}: {'Connected' if status else 'Failed'}")

    # Test passes only if at least one provider is connected and has models
    assert any(results.values()), "No LLM provider connected successfully."
    assert any(successful_models.values()), "No models retrieved from any provider."
    print("Successful models:", successful_models)
