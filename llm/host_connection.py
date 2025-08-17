import json
import os

class HostConnection:
    """Encapsulates host connection data and logic."""

    def __init__(self, config_path="config/host_config.json"):
        self.config_path = config_path
        self.hosts = []
        self.selected_host = ""
        self.selected_model = ""
        self.connected = False
        self.load_config()

    def load_config(self):
        """Load host config from file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                data = json.load(f)
            self.hosts = data.get("hosts", [])
            self.selected_host = data.get("selected_host", "")
            self.selected_model = data.get("selected_model", "")
        else:
            self.hosts = []
            self.selected_host = ""
            self.selected_model = ""

    def save_config(self):
        """Save host config to file."""
        data = {
            "hosts": self.hosts,
            "selected_host": self.selected_host,
            "selected_model": self.selected_model
        }
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

    def connect(self):
        """Simulate connection logic. Extend with real checks as needed."""
        # No host connected at start
        self.connected = False
        return self.connected

    def set_host(self, host_name):
        self.selected_host = host_name
        self.save_config()
        self.connect()

    def set_model(self, model_name):
        self.selected_model = model_name
        self.save_config()

    def get_status(self):
        """Return status string for UI."""
        status = "Connected" if self.connected else "Not Connected"
        host_str = self.selected_host if self.selected_host else "No host selected"
        model_str = self.selected_model if self.selected_model else "No model loaded"
        return f"Status: {status} | Host: {host_str} | Model: {model_str}"
