# Development Environment Setup

This guide will help you set up a development environment for Chronicle Weaver.

## Prerequisites

### System Requirements
- **Operating System**: Windows 11 (primary target), Windows 10, or Linux/macOS for development
- **Python**: 3.11 or 3.13 (3.12 also supported)
- **Git**: Latest version
- **Memory**: Minimum 8GB RAM (16GB recommended for AI model development)
- **Storage**: At least 5GB free space

### Required Tools
- **Code Editor**: Visual Studio Code (recommended), PyCharm, or similar
- **Terminal**: PowerShell (Windows), bash (Linux/macOS)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/JoergFlue/ChronicalWeaver.git
cd ChronicalWeaver
```

### 2. Install Poetry

Poetry is used for dependency management and virtual environment handling.

**Windows (PowerShell):**
```powershell
# Install Poetry
python -m pip install poetry

# Verify installation
poetry --version
```

**Linux/macOS:**
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH (add to your shell profile)
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
poetry --version
```

### 3. Install Dependencies

```bash
# Install all dependencies (including dev dependencies)
poetry install

# Install only production dependencies
poetry install --no-dev
```

### 4. Activate Virtual Environment

```bash
# Activate the Poetry virtual environment
poetry shell

# Or run commands directly with Poetry
poetry run python --version
```

### 5. Verify Installation

```bash
# Run basic tests to ensure everything is working
poetry run pytest tests/ -v

# Check code formatting
poetry run black --check .

# Run linting
poetry run flake8 .

# Type checking
poetry run mypy . --ignore-missing-imports
```

## Development Workflow

### Running the Application

```bash
# Run the main application
poetry run python main.py

# Run with debug logging
poetry run python main.py --debug
```

### Code Quality Tools

#### Formatting
```bash
# Format code with Black
poetry run black .

# Check formatting without making changes
poetry run black --check .
```

#### Linting
```bash
# Run flake8 linting
poetry run flake8 .

# Run with specific error codes only
poetry run flake8 . --select=E9,F63,F7,F82
```

#### Type Checking
```bash
# Run mypy type checking
poetry run mypy . --ignore-missing-imports

# Run mypy on specific files
poetry run mypy chronicle_weaver/
```

### Testing

#### Unit Tests
```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=chronicle_weaver

# Run specific test file
poetry run pytest tests/test_agents.py

# Run tests with verbose output
poetry run pytest -v
```

#### UI Tests
```bash
# Run PyQt-specific tests
poetry run pytest tests/ui/ --qt-platform=offscreen

# Run with X virtual framebuffer (Linux)
xvfb-run -a poetry run pytest tests/ui/
```

### Database Setup

Chronicle Weaver uses SQLite for local data storage. The database is automatically created when the application first runs.

```bash
# Initialize database with sample data (if available)
poetry run python scripts/init_db.py

# Reset database (development only)
poetry run python scripts/reset_db.py
```

## Development Environment Configuration

### Environment Variables

Create a `.env` file in the project root for development configuration:

```bash
# .env file (DO NOT commit to version control)

# LLM API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Image Generation API Keys
STABILITY_API_KEY=your_stability_ai_key_here
REPLICATE_API_TOKEN=your_replicate_token_here

# Local LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
LM_STUDIO_BASE_URL=http://localhost:1234/v1

# Development Settings
DEBUG=true
LOG_LEVEL=DEBUG
DEV_MODE=true
```

### VS Code Configuration

Recommended VS Code extensions:
- Python
- Pylance
- Black Formatter
- Flake8
- Git Graph
- PyQt6 Snippets

VS Code settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": ".venv/Scripts/python.exe",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".venv": true
    }
}
```

## Building and Packaging

### Development Build
```bash
# Install PyInstaller
poetry run pip install pyinstaller

# Create development build
poetry run pyinstaller --onedir --windowed --name ChronicleWeaver main.py

# Run the built application
./dist/ChronicleWeaver/ChronicleWeaver.exe
```

### Production Build
```bash
# Create optimized production build
poetry run pyinstaller --onefile --windowed --name ChronicleWeaver main.py

# The executable will be in dist/ChronicleWeaver.exe
```

## Troubleshooting

### Common Issues

#### Poetry Installation Issues
```bash
# If Poetry installation fails, try using pip in user mode
python -m pip install --user poetry

# Or use the official installer
curl -sSL https://install.python-poetry.org | python3 -
```

#### PyQt6 Import Errors
```bash
# Install system Qt dependencies (Linux)
sudo apt-get install qt6-base-dev

# Reinstall PyQt6
poetry run pip uninstall PyQt6
poetry install --no-cache
```

#### Permission Errors (Windows)
```powershell
# Run PowerShell as Administrator
# Enable execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Set environment variable
export DEBUG=true

# Or run with debug flag
poetry run python main.py --debug --log-level=DEBUG
```

### Performance Profiling

```bash
# Install profiling tools
poetry add --group dev py-spy cProfile

# Profile application startup
poetry run py-spy record -o profile.svg -- python main.py

# Profile specific function
poetry run python -m cProfile -o profile.prof main.py
```

## Contributing

### Git Workflow

1. Create a feature branch from `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "feat: add your feature description"
```

3. Run quality checks:
```bash
poetry run black .
poetry run flake8 .
poetry run pytest
```

4. Push and create pull request:
```bash
git push origin feature/your-feature-name
# Create PR through GitHub interface
```

### Code Style Guidelines

- Follow PEP 8 style guide
- Use Black for code formatting
- Write docstrings for all public functions and classes
- Add type hints where appropriate
- Maintain test coverage above 80%

### Commit Message Format

Use conventional commits format:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `style:` formatting changes
- `refactor:` code refactoring
- `test:` adding tests
- `chore:` maintenance tasks

## Support

For development questions and issues:
1. Check the documentation in `/docs`
2. Look for similar issues in GitHub Issues
3. Create a new issue with detailed description
4. Join the development Discord (if available)
