from PyQt6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QTextEdit, QLabel
from PyQt6.QtCore import Qt

import logging

class LoggingTab(QWidget):
    """Tab for viewing and filtering internal logs."""

    def __init__(self, log_levels=None, parent=None):
        super().__init__(parent)
        self.log_levels = log_levels or ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        self._setup_ui()
        self._setup_logging_handler()
        self.filter_dropdown.currentTextChanged.connect(self._on_filter_changed)
        self.current_level = logging.DEBUG

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Dropdown for log filter
        filter_label = QLabel("Logging Filter:")
        layout.addWidget(filter_label)

        self.filter_dropdown = QComboBox()
        self.filter_dropdown.addItems(self.log_levels)
        layout.addWidget(self.filter_dropdown)

        # Text field for logs
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

    def set_logs(self, logs: str):
        self.log_text.setPlainText(logs)

    def append_log(self, log: str):
        self.log_text.append(log)

    def _setup_logging_handler(self):
        class QtLogHandler(logging.Handler):
            def __init__(self, tab):
                super().__init__()
                self.tab = tab

            def emit(self, record):
                msg = self.format(record)
                if record.levelno >= self.tab.current_level:
                    self.tab.append_log(msg)

        self.qt_handler = QtLogHandler(self)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        self.qt_handler.setFormatter(formatter)
        logging.getLogger().addHandler(self.qt_handler)

    def _on_filter_changed(self, level_name):
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        self.current_level = level_map.get(level_name, logging.DEBUG)
        self.log_text.clear()
