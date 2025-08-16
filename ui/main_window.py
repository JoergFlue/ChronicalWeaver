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

        logger.info("Main window initialized")

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
        self.tab_widget.addTab(QWidget(), "Settings")

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

        llm_menu = menubar.addMenu("LLM")

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
        """Update status bar with current LLM provider"""
        provider = self.llm_manager.current_provider
        self.status_bar.showMessage(f"Ready - LLM: {provider.replace('_', ' ').title()}")

    def _new_conversation(self):
        """Start a new conversation"""
        if hasattr(self.roleplay_tab, 'clear_conversation'):
            self.roleplay_tab.clear_conversation()
        logger.info("Started new conversation")

    def _switch_llm_provider(self, provider: str):
        """Switch to a different LLM provider"""
        success = self.llm_manager.set_provider(provider)
        if success:
            self._update_status_bar()
            logger.info(f"Switched to LLM provider: {provider}")
        else:
            QMessageBox.warning(
                self, 
                "Provider Switch Failed", 
                f"Could not switch to provider: {provider}"
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
