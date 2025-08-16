# Chronicle Weaver - Phase 0: Planning & Foundation

**Duration**: 2-4 Days  
**Implementation Confidence**: 95% - Low Risk  
**Dependencies**: None  
**Next Phase**: Phase 1 (Core LLM & Main Agent System)

## Overview
Establish the development environment, project structure, and foundational documentation required for all subsequent phases. This phase focuses on creating a solid foundation that enables rapid development in later phases.

## Key Risk Factors
- **Dependency conflicts** between packages
- **Team environment variations** across different development setups
- **Version compatibility** issues with Python 3.11+ and PyQt6
- **Development tool configuration** inconsistencies

## Acceptance Criteria
- [ ] Development environment is set up with Python 3.11+, PyQt6, and all dependencies
- [ ] Git repository is initialized with proper .gitignore and README
- [ ] Project structure follows modular design principles
- [ ] Basic CI/CD pipeline is configured
- [ ] Architecture documentation is complete and reviewed
- [ ] All team members can run the development environment

## Detailed Implementation Steps

### 1. Environment Setup (Day 1)

#### 1.1 Python Environment
```bash
# Install Python 3.11+ (if not present)
python --version  # Verify 3.11+

# Create virtual environment
python -m venv chronicle_weaver_env
# Windows activation
chronicle_weaver_env\Scripts\activate
```

#### 1.2 Core Dependencies Installation
```bash
pip install --upgrade pip
pip install PyQt6==6.6.0
pip install crewai==0.22.5
pip install langchain==0.1.0
pip install litellm==1.28.0
pip install sqlalchemy==2.0.25
pip install requests==2.31.0
pip install pytest==7.4.4
pip install black==23.12.1
pip install flake8==7.0.0
```

#### 1.3 Development Tools
```bash
# Install additional development tools
pip install pytest-qt==4.3.1  # PyQt testing
pip install pytest-cov==4.0.0  # Coverage reporting
pip install sphinx==7.2.6  # Documentation
pip install pre-commit==3.6.0  # Git hooks
```

### 2. Project Structure Creation (Day 1-2)

#### 2.1 Directory Structure
```
chronicle_weaver/
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   └── agent_configs/
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── llm_manager.py
│   │   └── providers/
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── migrations/
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── components/
│   │   └── resources/
│   ├── image_gen/
│   │   ├── __init__.py
│   │   └── providers/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logging.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── ui/
├── docs/
│   ├── api/
│   ├── user/
│   └── developer/
├── config/
│   ├── default_config.yaml
│   ├── llm_configs.json
│   └── agent_templates.json
├── assets/
│   ├── icons/
│   └── images/
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── pytest.ini
├── .gitignore
├── .pre-commit-config.yaml
└── README.md
```

#### 2.2 Essential Configuration Files

**requirements.txt**
```
PyQt6>=6.6.0
crewai>=0.22.5
langchain>=0.1.0
litellm>=1.28.0
sqlalchemy>=2.0.25
requests>=2.31.0
pyyaml>=6.0.1
```

**requirements-dev.txt**
```
-r requirements.txt
pytest>=7.4.4
pytest-qt>=4.3.1
pytest-cov>=4.0.0
black>=23.12.1
flake8>=7.0.0
sphinx>=7.2.6
pre-commit>=3.6.0
```

**pytest.ini**
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = --verbose --cov=src --cov-report=html --cov-report=term
```

**.gitignore**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
chronicle_weaver_env/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Application specific
config/user_config.yaml
logs/
*.db
*.sqlite
images/generated/
temp/
```

### 3. Architecture Documentation (Day 2-3)

#### 3.1 README.md
```markdown
# Chronicle Weaver

An AI-driven roleplaying assistant for Windows 11 featuring modular agent systems, flexible LLM backends, robust memory management, and integrated image generation.

## Features
- Multiple LLM backend support (OpenAI, Gemini, Ollama, LM Studio)
- Modular agent system with customizable personalities
- Persistent memory and conversation tracking
- Integrated image generation
- Props and clothing library management

## Quick Start
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run application: `python src/main.py`

## Development Setup
See [Developer Guide](docs/developer/setup.md)
```

#### 3.2 Architecture Decision Records (ADRs)
Create `docs/adr/` with:
- **ADR-001**: LiteLLM for LLM abstraction
- **ADR-002**: PyQt6 for desktop UI
- **ADR-003**: SQLite for local data persistence
- **ADR-004**: CrewAI for agent orchestration
- **ADR-005**: Modular plugin architecture

### 4. Basic CI/CD Setup (Day 3-4)

#### 4.1 GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: windows-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
    - name: Run tests
      run: |
        pytest
    - name: Run linting
      run: |
        flake8 src tests
        black --check src tests
```

#### 4.2 Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
-   repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
    -   id: black
-   repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
    -   id: flake8
-   repo: local
    hooks:
    -   id: pytest
        name: pytest
        entry: pytest
        language: system
        types: [python]
        stages: [commit]
```

### 5. Foundation Code Templates (Day 4)

#### 5.1 Basic Application Entry Point
```python
# src/main.py
"""
Chronicle Weaver - Main Application Entry Point
"""
import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow
from utils.logging import setup_logging
from utils.config import load_config

def main():
    """Main application entry point"""
    # Setup logging
    setup_logging()
    
    # Load configuration
    config = load_config()
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Chronicle Weaver")
    app.setApplicationVersion("0.1.0")
    
    # Create main window
    window = MainWindow(config)
    window.show()
    
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

#### 5.2 Configuration Management
```python
# src/utils/config.py
"""Configuration management utilities"""
import yaml
import os
from pathlib import Path

def load_config():
    """Load application configuration"""
    config_path = Path("config/default_config.yaml")
    user_config_path = Path("config/user_config.yaml")
    
    # Load default config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with user config if exists
    if user_config_path.exists():
        with open(user_config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            config.update(user_config)
    
    return config
```

## Testing Strategy

### Unit Tests
- Configuration loading and validation
- Utility function testing
- Module import verification

### Integration Tests
- Development environment setup verification
- Dependency compatibility testing
- Cross-platform compatibility (Windows focus)

### Manual Verification
- [ ] All dependencies install without conflicts
- [ ] Project structure is navigable
- [ ] Documentation is readable and accurate
- [ ] Git repository is properly configured

## Success Metrics
- [ ] Zero dependency installation errors
- [ ] All team members can run `python src/main.py` without errors
- [ ] Documentation passes review
- [ ] CI/CD pipeline executes successfully
- [ ] Project structure supports all planned features

## Deliverables
1. **Functional Development Environment** - Ready for Phase 1 development
2. **Project Structure** - Complete directory hierarchy with placeholder files
3. **Documentation Foundation** - README, architecture docs, and ADRs
4. **CI/CD Pipeline** - Automated testing and quality checks
5. **Configuration System** - Flexible, environment-aware configuration

## Handoff to Phase 1
Upon completion, Phase 1 can immediately begin implementing:
- LLM integration using the established structure
- Main Agent development in `src/agents/`
- Basic UI components in `src/ui/`
- Comprehensive testing using the established framework

The foundation provides clear interfaces and patterns that guide implementation in all subsequent phases.
