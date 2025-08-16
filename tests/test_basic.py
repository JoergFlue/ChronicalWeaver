"""
Basic tests to verify the Chronicle Weaver setup and imports.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_chronicle_weaver_import():
    """Test that the main package can be imported."""
    import chronicle_weaver

    assert chronicle_weaver.__version__ == "0.1.0"
    assert chronicle_weaver.__author__ == "Chronicle Weaver Team"


def test_module_imports():
    """Test that all module packages can be imported."""
    # Test UI module
    import ui

    assert ui is not None

    # Test agents module
    import agents

    assert agents is not None

    # Test memory module
    import memory

    assert memory is not None

    # Test LLM module
    import llm

    assert llm is not None

    # Test images module
    import images

    assert images is not None


def test_dependencies():
    """Test that core dependencies are available."""
    # Test PyQt6
    try:
        from PyQt6.QtWidgets import QApplication

        assert QApplication is not None
    except ImportError:
        pytest.skip("PyQt6 not available in test environment")

    # Test LangChain
    import langchain

    assert langchain is not None

    # Test LiteLLM
    import litellm

    assert litellm is not None

    # Test OpenAI
    import openai

    assert openai is not None

    # Test requests
    import requests

    assert requests is not None


def test_project_structure():
    """Test that the expected project structure exists."""
    project_root = Path(__file__).parent.parent

    # Check main directories exist
    assert (project_root / "ui").is_dir()
    assert (project_root / "agents").is_dir()
    assert (project_root / "memory").is_dir()
    assert (project_root / "llm").is_dir()
    assert (project_root / "images").is_dir()
    assert (project_root / "tests").is_dir()
    assert (project_root / "docs").is_dir()

    # Check key files exist
    assert (project_root / "README.md").is_file()
    assert (project_root / "pyproject.toml").is_file()
    assert (project_root / "chronicle_weaver" / "__init__.py").is_file()


if __name__ == "__main__":
    pytest.main([__file__])
