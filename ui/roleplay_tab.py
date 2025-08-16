"""Roleplay conversation tab"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                           QTextEdit, QPushButton, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
from agents.main_agent import MainAgent
from llm.llm_manager import LLMManager
import logging

logger = logging.getLogger(__name__)

class MessageGenerationWorker(QThread):
    """Worker thread for generating AI responses"""
    chunk_received = pyqtSignal(str)
    generation_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, agent: MainAgent, message: str):
        super().__init__()
        self.agent = agent
        self.message = message

    def run(self):
        """Run the generation in thread"""
        import asyncio
        try:
            response = asyncio.run(self.agent.process_message(self.message))
            self.chunk_received.emit(response.content)
            self.generation_complete.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))

class RoleplayTab(QWidget):
    """Tab for roleplay conversations"""

    def __init__(self, main_agent: MainAgent, llm_manager: LLMManager):
        super().__init__()
        self.main_agent = main_agent
        self.llm_manager = llm_manager
        self.generation_worker = None

        self._setup_ui()
        logger.info("Roleplay tab initialized")

    def _setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        self.conversation_area = QTextEdit()
        self.conversation_area.setReadOnly(True)
        self.conversation_area.setFont(QFont("Segoe UI", 10))
        splitter.addWidget(self.conversation_area)

        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)

        self.message_input = QTextEdit()
        self.message_input.setMaximumHeight(100)
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.setFont(QFont("Segoe UI", 10))
        input_layout.addWidget(self.message_input)

        button_layout = QHBoxLayout()
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._send_message)
        self.send_button.setDefault(True)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_conversation)

        button_layout.addStretch()
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.send_button)
        input_layout.addLayout(button_layout)

        splitter.addWidget(input_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.message_input.keyPressEvent = self._handle_key_press
        self._add_system_message("Welcome to Chronicle Weaver! Start a conversation to begin your roleplaying adventure.")

    def _handle_key_press(self, event):
        """Handle key press events in message input"""
        if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self._send_message()
        else:
            QTextEdit.keyPressEvent(self.message_input, event)

    def _send_message(self):
        """Send user message and get AI response"""
        message = self.message_input.toPlainText().strip()
        if not message:
            return

        self._add_user_message(message)
        self.message_input.clear()
        self.send_button.setEnabled(False)
        self.send_button.setText("Generating...")
        self._generate_ai_response(message)

    def _generate_ai_response(self, message: str):
        """Generate AI response in background thread"""
        if self.generation_worker and self.generation_worker.isRunning():
            return

        self.generation_worker = MessageGenerationWorker(self.main_agent, message)
        self.generation_worker.chunk_received.connect(self._on_response_chunk)
        self.generation_worker.generation_complete.connect(self._on_generation_complete)
        self.generation_worker.error_occurred.connect(self._on_generation_error)
        self.generation_worker.start()
        self._add_ai_message_start()

    def _on_response_chunk(self, chunk: str):
        """Handle received response chunk"""
        self._append_to_current_ai_message(chunk)

    def _on_generation_complete(self):
        """Handle generation completion"""
        self._finalize_ai_message()
        self.send_button.setEnabled(True)
        self.send_button.setText("Send")
        logger.info("AI response generation completed")

    def _on_generation_error(self, error: str):
        """Handle generation error"""
        self._add_system_message(f"Error generating response: {error}")
        self.send_button.setEnabled(True)
        self.send_button.setText("Send")
        logger.error(f"AI response generation error: {error}")

    def _add_user_message(self, message: str):
        """Add user message to conversation"""
        self.conversation_area.append(f"<div style='margin-bottom: 10px;'>"
                                    f"<strong style='color: #0066cc;'>You:</strong><br>"
                                    f"<span style='margin-left: 20px;'>{self._escape_html(message)}</span>"
                                    f"</div>")

    def _add_ai_message_start(self):
        """Start a new AI message"""
        self.conversation_area.append(f"<div style='margin-bottom: 10px;'>"
                                    f"<strong style='color: #cc6600;'>Assistant:</strong><br>"
                                    f"<span id='current-ai-message' style='margin-left: 20px;'>")
        self._current_ai_message = ""

    def _append_to_current_ai_message(self, text: str):
        """Append text to current AI message"""
        self._current_ai_message += text
        cursor = self.conversation_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.conversation_area.setTextCursor(cursor)
        self.conversation_area.ensureCursorVisible()

    def _finalize_ai_message(self):
        """Finalize the current AI message"""
        self.conversation_area.append("</span></div>")

    def _add_system_message(self, message: str):
        """Add system message to conversation"""
        self.conversation_area.append(f"<div style='margin-bottom: 10px; font-style: italic; color: #666;'>"
                                    f"{self._escape_html(message)}"
                                    f"</div>")

    def _escape_html(self, text: str) -> str:
        """Escape HTML characters in text"""
        return (text.replace('&', '&')
                   .replace('<', '<')
                   .replace('>', '>')
                   .replace('"', '"')
                   .replace("'", '&#x27;'))

    def clear_conversation(self):
        """Clear the conversation history"""
        self.conversation_area.clear()
        self.main_agent.clear_history()
        self._add_system_message("Conversation cleared. Start a new conversation!")
        logger.info("Conversation cleared")
