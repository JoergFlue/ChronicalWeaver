import os
import json

class ConfigManager:
    """Handles reading and writing configuration and settings files."""

    @staticmethod
    def load_user_settings(settings_path="config/user_settings.json"):
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    @staticmethod
    def save_user_settings(settings, settings_path="config/user_settings.json"):
        try:
            with open(settings_path, "w") as f:
                json.dump(settings, f, indent=2)
            return True
        except Exception:
            return False

    @staticmethod
    def load_llm_configs(llm_config_path="config/llm_configs.json"):
        if os.path.exists(llm_config_path):
            try:
                with open(llm_config_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    @staticmethod
    def save_llm_configs(llm_configs, llm_config_path="config/llm_configs.json"):
        try:
            with open(llm_config_path, "w") as f:
                json.dump(llm_configs, f, indent=2)
            return True
        except Exception:
            return False
