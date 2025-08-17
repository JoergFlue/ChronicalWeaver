"""Main application window"""
from PyQt6.QtWidgets import (QMainWindow, QTabWidget, QVBoxLayout, 
                           QWidget, QMenuBar, QStatusBar, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from .roleplay_tab import RoleplayTab
from llm.llm_manager import LLMManager
from agents.main_agent import MainAgent
import logging

logger = logging.getLogger(__name__)

import os
import json

class MainWindow(QMainWindow):
    """Main application window with tabbed interface"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.llm_manager = LLMManager()
        self.main_agent = MainAgent(self.llm_manager)

        self.setWindowTitle("Chronicle Weaver")
        self.setGeometry(100, 100, 1200, 800)

        self._setup_ui()
        self._setup_menu_bar()
        self._setup_status_bar()
        self._load_llm_settings()

        logger.info("Main window initialized")

    def _load_llm_settings(self):
        """Load last used LLM provider/model from settings file"""
        settings_path = os.path.join("config", "user_settings.json")
        llm_config_path = os.path.join("config", "llm_configs.json")
        base_url = "http://localhost:11434"
        model = "llama2"
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r") as f:
                    settings = json.load(f)
                ollama = settings.get("ollama", {})
                base_url = ollama.get("base_url", base_url)
                model = ollama.get("model", model)
            except Exception as e:
                self.status_bar.showMessage(f"Failed to load LLM settings: {e}")
        # Update Ollama config in llm_configs.json
        try:
            if os.path.exists(llm_config_path):
                with open(llm_config_path, "r") as f:
                    llm_configs = json.load(f)
            else:
                llm_configs = {}
            if "ollama" not in llm_configs:
                llm_configs["ollama"] = {}
            llm_configs["ollama"]["provider"] = "ollama"
            llm_configs["ollama"]["model"] = model
            llm_configs["ollama"]["base_url"] = base_url
            with open(llm_config_path, "w") as f:
                json.dump(llm_configs, f, indent=2)
        except Exception as e:
            self.status_bar.showMessage(f"Failed to update Ollama config: {e}")
        # Set provider in LLMManager
        success = self.llm_manager.set_provider("ollama")
        if success:
            self.status_bar.showMessage(f"Connected to Ollama: {model}")
        else:
            self.status_bar.showMessage("Could not connect to Ollama API.")

    def _setup_ui(self):
        """Setup the main UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        self.roleplay_tab = RoleplayTab(self.main_agent, self.llm_manager)
        self.tab_widget.addTab(self.roleplay_tab, "Roleplay")

        self.tab_widget.addTab(QWidget(), "Agents")
        self.tab_widget.addTab(QWidget(), "Library")
        from ui.settings_tab import SettingsTab
        self.tab_widget.addTab(SettingsTab(), "Settings")

    def _setup_menu_bar(self):
        """Setup the menu bar"""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")

        new_action = QAction("New Conversation", self)
        new_action.triggered.connect(self._new_conversation)
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        llm_menu = menubar.addMenu("LLM Host")

        providers = ["openai", "gemini", "ollama", "lm_studio"]
        for provider in providers:
            action = QAction(provider.replace("_", " ").title(), self)
            action.triggered.connect(lambda checked, p=provider: self._switch_llm_provider(p))
            llm_menu.addAction(action)

        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_status_bar(self):
        """Setup the status bar"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        self._update_status_bar()

    def _update_status_bar(self):
        """Update status bar with connection, host, and model info"""
        status_str = self.llm_manager.host_connection.get_status()
        self.status_bar.showMessage(status_str)

    def _new_conversation(self):
        """Start a new conversation"""
        if hasattr(self.roleplay_tab, 'clear_conversation'):
            self.roleplay_tab.clear_conversation()
        logger.info("Started new conversation")

    def _switch_llm_provider(self, provider: str):
        """Switch to a different LLM provider"""
        success = self.llm_manager.set_provider(provider)
        if success:
            models = self.llm_manager.list_models()
            if models:
                self.status_bar.showMessage(f"Connected to {provider}. Models: {', '.join(models)}")
                logger.info(f"Connected to {provider}. Models: {models}")
            else:
                self.status_bar.showMessage(f"Connected to {provider}, but no models found.")
                logger.warning(f"Connected to {provider}, but no models found.")
            self._update_status_bar()
        else:
            self.status_bar.showMessage("Not ready.")
            QMessageBox.critical(
                self, 
                "Provider Connection Failed", 
                f"Could not connect to provider API: {provider}"
            )

    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Chronicle Weaver",
            "Chronicle Weaver v0.1.0\n\n"
            "An AI-driven roleplaying assistant with modular agent systems, "
            "flexible LLM backends, and integrated image generation."
        )
