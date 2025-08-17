import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QCheckBox
import logging
from llm.llm_manager import LLMManager

class SettingsTab(QWidget):
    def __init__(self, parent=None, on_settings_saved=None, on_status_update=None):
        super().__init__(parent)
        self.on_settings_saved = on_settings_saved
        self.on_status_update = on_status_update
        self.layout = QVBoxLayout(self)
        self.host_dropdown = QComboBox(self)
        self.model_dropdown = QComboBox(self)
        self.current_model_label = QLabel("No model loaded", self)
        self.save_button = QPushButton("Save Selection", self)
        self.llm_manager = LLMManager()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Memory management checkbox
        from config.config_manager import ConfigManager
        settings = ConfigManager.load_user_settings()
        self.memory_checkbox = QCheckBox("Enable Memory Management", self)
        self.memory_checkbox.setChecked(settings.get("enable_memory_management", True))

        self.layout.addWidget(QLabel("Host:", self))
        self.layout.addWidget(self.host_dropdown)
        self.layout.addWidget(QLabel("Available Models:", self))
        self.layout.addWidget(self.model_dropdown)
        self.layout.addWidget(QLabel("Current Model:", self))
        self.layout.addWidget(self.current_model_label)
        self.layout.addWidget(self.memory_checkbox)
        self.layout.addWidget(self.save_button)

        self._populate_hosts()
        self.current_model_label.setText("No model loaded")

        self.host_dropdown.currentIndexChanged.connect(self._host_changed)
        self.model_dropdown.currentIndexChanged.connect(self._model_changed)
        self.save_button.clicked.connect(self._save_selection)

    def _populate_hosts(self):
        self.host_dropdown.clear()
        hosts = self.llm_manager.host_connection.hosts
        host_names = ["none"] + [h.get("name", "") for h in hosts]
        self.host_dropdown.addItems(host_names)
        # Set current host selection
        current_host = self.llm_manager.host_connection.selected_host or "none"
        idx = self.host_dropdown.findText(current_host)
        self.host_dropdown.setCurrentIndex(idx if idx != -1 else 0)
        # Do not call self._host_changed here to avoid status bar errors during init

    def _populate_models(self, models):
        self.model_dropdown.clear()
        model_names = ["none"] + list(models)
        self.model_dropdown.addItems(model_names)
        # Set current model selection
        current_model = self.llm_manager.current_model or "none"
        idx = self.model_dropdown.findText(current_model)
        self.model_dropdown.setCurrentIndex(idx if idx != -1 else 0)

    def _host_changed(self, idx):
        host_name = self.host_dropdown.itemText(idx)
        if host_name == "none":
            self.logger.info("Disconnecting from host.")
            self.llm_manager.set_host("")
            self._populate_models([])
            self.current_model_label.setText("No model loaded")
            if self.on_status_update:
                self.on_status_update(host_name, "")
            return
        self.logger.info(f"Attempting connection to host: {host_name}")
        self.llm_manager.set_host(host_name)
        connected = self.llm_manager.test_connection(host_name)
        models = self.llm_manager.list_models() if connected else []
        self._populate_models(models)
        if connected:
            self.logger.info(f"Connection to host '{host_name}' successful.")
        else:
            self.logger.warning(f"Connection to host '{host_name}' failed.")
        if self.on_status_update:
            self.on_status_update(host_name, self.llm_manager.current_model)

    def _model_changed(self, idx):
        model = self.model_dropdown.currentText()
        if model == "none":
            self.logger.info("Unloading model from host.")
            self.llm_manager.current_model = ""
            self.current_model_label.setText("No model loaded")
            if self.on_status_update:
                self.on_status_update(self.llm_manager.host_connection.selected_host, "")
            return
        if model:
            self.logger.info(f"Attempting to load model: {model}")
            result = self.llm_manager.load_model(model)
            self.logger.info(f"Model load result: {result}")
            self.current_model_label.setText(result)
            if self.on_status_update:
                self.on_status_update(self.llm_manager.host_connection.selected_host, model)
        else:
            self.current_model_label.setText("No model loaded")

    def _save_selection(self):
        from config.config_manager import ConfigManager
        provider = self.host_dropdown.currentText()
        model = self.model_dropdown.currentText()
        settings = ConfigManager.load_user_settings()
        settings["last_provider"] = provider
        settings["last_model"] = model
        settings["enable_memory_management"] = self.memory_checkbox.isChecked()
        ConfigManager.save_user_settings(settings)
        self.current_model_label.setText(model)
        if self.on_settings_saved:
            self.on_settings_saved()
