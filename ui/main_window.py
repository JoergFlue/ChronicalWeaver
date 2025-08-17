"""Main application window"""
from PyQt6.QtWidgets import (QMainWindow, QTabWidget, QVBoxLayout, 
                           QWidget, QMenuBar, QStatusBar, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from .roleplay_tab import RoleplayTab
from llm.llm_manager import LLMManager
from agents.main_agent import MainAgent
import logging
from config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

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
        """Load last used LLM provider/model from settings file (generic)"""
        settings = ConfigManager.load_user_settings()
        provider = settings.get("last_provider")
        model = settings.get("last_model")
        base_url = settings.get("last_base_url")

        # Fallback: if not present, try to get any provider section
        if not provider:
            for key in settings:
                if isinstance(settings[key], dict):
                    provider = key
                    model = settings[key].get("model")
                    base_url = settings[key].get("base_url")
                    break

        # Update provider config in llm_configs.json
        llm_configs = ConfigManager.load_llm_configs()
        if provider:
            if provider not in llm_configs:
                llm_configs[provider] = {}
            llm_configs[provider]["provider"] = provider
            if model:
                llm_configs[provider]["model"] = model
            if base_url:
                llm_configs[provider]["base_url"] = base_url
            ConfigManager.save_llm_configs(llm_configs)
        # Set provider in LLMManager
        if provider:
            success = self.llm_manager.set_host(provider)
            if success:
                msg = f"Connected to {provider}"
                if model:
                    msg += f": {model}"
                self.status_bar.showMessage(msg)
            else:
                self.status_bar.showMessage(f"Could not connect to {provider} API.")
        else:
            self.status_bar.showMessage("No provider configured.")

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
        self.tab_widget.addTab(SettingsTab(
            on_settings_saved=self._load_llm_settings,
            on_status_update=self._handle_status_update
        ), "Settings")

        from ui.logging_tab import LoggingTab
        self.logging_tab = LoggingTab()
        self.tab_widget.addTab(self.logging_tab, "Logging")

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

    def _handle_status_update(self, host, model):
        """Update status bar when host/model changes from SettingsTab"""
        msg = "Ready"
        if host and host != "none":
            msg = f"Host: {host}"
            if model and model != "none":
                msg += f" | Model: {model}"
        self.status_bar.showMessage(msg)

    def _new_conversation(self):
        """Start a new conversation"""
        logger.debug("Menu action: New Conversation triggered")
        if hasattr(self.roleplay_tab, 'clear_conversation'):
            self.roleplay_tab.clear_conversation()
        logger.info("Started new conversation")

    def _switch_llm_provider(self, provider: str):
        """Switch to a different LLM provider"""
        logger.debug(f"Menu action: Switch LLM Provider triggered ({provider})")
        success = self.llm_manager.set_host(provider)
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
        logger.debug("Menu action: About triggered")
        QMessageBox.about(
            self,
            "About Chronicle Weaver",
            "Chronicle Weaver v0.1.0\n\n"
            "An AI-driven roleplaying assistant with modular agent systems, "
            "flexible LLM backends, and integrated image generation."
        )

    # Simulation methods for integration testing
    def simulate_new_conversation(self):
        """Simulate clicking 'New Conversation' menu action"""
        self._new_conversation()

    def simulate_switch_llm_provider(self, provider: str):
        """Simulate switching LLM provider via menu"""
        self._switch_llm_provider(provider)

    def simulate_show_about(self):
        """Simulate clicking 'About' menu action"""
        self._show_about()
