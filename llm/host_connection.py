import json
import os

class HostConnection:
    """
    Encapsulates host connection data and logic.

    Attributes:
        config_path (str): Path to the host configuration file.
        hosts (list): List of available hosts.
        selected_host (str): Currently selected host name.
        selected_model (str): Currently selected model name.
        connected (bool): Connection status.
    """

    def __init__(self, config_path="config/host_config.json"):
        """
        Initialize HostConnection.

        Args:
            config_path (str): Path to the host configuration file.
        """
        self.config_path = config_path
        self.hosts = []
        self.selected_host = ""
        self.selected_model = ""
        self.connected = False
        self.load_config()

    def load_config(self):
        """
        Load host configuration from file.

        Loads hosts, selected host, and selected model from the config file.
        If the file does not exist, initializes with empty values.
        """
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
        """
        Save host configuration to file.

        Persists hosts, selected host, and selected model to the config file.
        """
        data = {
            "hosts": self.hosts,
            "selected_host": self.selected_host,
            "selected_model": self.selected_model
        }
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

    def connect(self):
        """
        Simulate connection logic.

        Returns:
            bool: Connection status (always False in simulation).
        """
        self.connected = False
        return self.connected

    def set_host(self, host_name):
        """
        Set the selected host.

        Args:
            host_name (str): Name of the host to select.
        """
        self.selected_host = host_name
        self.save_config()
        self.connect()

    def set_model(self, model_name):
        """
        Set the selected model.

        Args:
            model_name (str): Name of the model to select.
        """
        self.selected_model = model_name
        self.save_config()

    def get_status(self):
        """
        Get connection status string for UI.

        Returns:
            str: Status string including connection, host, and model info.
        """
        status = "Connected" if self.connected else "Not Connected"
        host_str = self.selected_host if self.selected_host else "No host selected"
        model_str = self.selected_model if self.selected_model else "No model loaded"
        return f"Status: {status} | Host: {host_str} | Model: {model_str}"
