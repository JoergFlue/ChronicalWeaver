import json
import os

class HostConnection:
    """Encapsulates host connection data and logic."""

    def __init__(self, config_path="config/host_config.json"):
        self.config_path = config_path
        self.host_url = None
        self.provider = None
        self.model = None
        self.connected = False
        self.load_config()

    def load_config(self):
        """Load host config from file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                data = json.load(f)
            self.host_url = data.get("host_url", "http://localhost:11434")
            self.provider = data.get("provider", "ollama")
            self.model = data.get("model", None)
        else:
            self.host_url = "http://localhost:11434"
            self.provider = "ollama"
            self.model = None

    def save_config(self):
        """Save host config to file."""
        data = {
            "host_url": self.host_url,
            "provider": self.provider,
            "model": self.model
        }
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

    def connect(self):
        """Simulate connection logic. Extend with real checks as needed."""
        # Example: try to reach host_url, set self.connected accordingly
        import requests
        try:
            resp = requests.get(self.host_url, timeout=3)
            self.connected = resp.status_code == 200
        except Exception:
            self.connected = False
        return self.connected

    def set_host(self, host_url, provider=None, model=None):
        self.host_url = host_url
        if provider:
            self.provider = provider
        if model:
            self.model = model
        self.save_config()
        self.connect()

    def set_model(self, model):
        self.model = model
        self.save_config()

    def get_status(self):
        """Return status string for UI."""
        status = "Connected" if self.connected else "Not Connected"
        model_str = self.model if self.model else "No model loaded"
        return f"Status: {status} | Host: {self.host_url} | Provider: {self.provider} | Model: {model_str}"
