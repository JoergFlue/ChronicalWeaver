from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton

class SettingsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.provider_dropdown = QComboBox(self)
        self.provider_dropdown.addItems(["ollama", "lm_studio", "openai", "claude", "gemini"])
        self.model_dropdown = QComboBox(self)
        self.current_model_label = QLabel(self)
        self.save_button = QPushButton("Save Selection", self)

        self.layout.addWidget(QLabel("Provider:", self))
        self.layout.addWidget(self.provider_dropdown)
        self.layout.addWidget(QLabel("Available Models:", self))
        self.layout.addWidget(self.model_dropdown)
        self.layout.addWidget(QLabel("Current Model:", self))
        self.layout.addWidget(self.current_model_label)
        self.layout.addWidget(self.save_button)

        # UI only: signals to notify MainWindow of changes
        self.provider_dropdown.currentIndexChanged.connect(self._provider_changed)
        self.model_dropdown.currentIndexChanged.connect(self._model_changed)
        self.save_button.clicked.connect(self._save_selection)

    def _provider_changed(self, idx):
        # Notify MainWindow to update provider and models
        pass

    def _model_changed(self, idx):
        # Notify MainWindow to update model selection
        pass

    def _save_selection(self):
        # Notify MainWindow to save selection
        pass
