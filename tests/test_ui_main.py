import pytest
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

@pytest.fixture(scope="module")
def app():
    import sys
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

def test_main_window_launch(app):
    """Test that MainWindow launches without error."""
    window = MainWindow()
    window.show()
    assert window.isVisible()
    window.close()
