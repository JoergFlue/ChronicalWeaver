# Chronicle Weaver - AI-Driven Roleplaying Assistant

Chronicle Weaver is a user-friendly, AI-driven roleplaying assistant for Windows 11, designed to provide rich, continuous roleplay experiences through a modular agent system, flexible Large Language Model (LLM) backends, robust memory management, and integrated image generation.

## Features

- **Modular Agent System**: Specialized AI agents for different roleplay tasks
- **Multiple LLM Support**: Integration with OpenAI, Google Gemini, Ollama, LM Studio via LiteLLM
- **Memory Management**: Dual-layered memory system for short-term conversation and long-term continuity
- **Image Generation**: Support for DALL-E 3, Stability AI, and local generation (Automatic1111/ComfyUI)
- **Intuitive UI**: PyQt6-based desktop application with tabbed interface

## Quick Start

### Prerequisites

- Windows 11
- Python 3.11 or higher
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/JoergFlue/ChronicalWeaver.git
cd ChronicalWeaver
```

2. Install dependencies using Poetry (recommended):
```bash
pip install poetry
poetry install
```

Or using pip:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
poetry run python main.py
```

## Architecture

Chronicle Weaver follows a modular architecture with clearly separated components:

```
├── ui/          # PyQt6 user interface components
├── agents/      # CrewAI agent implementations
├── memory/      # Memory management (LangChain + SQLite)
├── llm/         # LLM integration layer (LiteLLM)
├── images/      # Image generation integrations
├── tests/       # Unit, integration, and E2E tests
└── docs/        # Documentation and planning files
```

## Development

### Core Technology Stack

- **Frontend**: PyQt6
- **Agent Framework**: CrewAI
- **LLM Integration**: LiteLLM
- **Memory**: LangChain + SQLite
- **Image Generation**: Multiple API integrations
- **Language**: Python 3.11+
- **Packaging**: PyInstaller

### Development Principles

- **Modularity**: Independent, reusable components with clear interfaces
- **Scalability**: Easy integration of new LLMs, agents, and features
- **User-Centric Design**: Intuitive UI for agent management and roleplay
- **Robust Testing**: TDD approach with comprehensive test coverage
- **Documentation**: Clear internal and external documentation
- **Iterative Development**: Continuous improvement and feedback integration

### Running Tests

```bash
poetry run pytest
```

### Code Quality

```bash
# Linting
poetry run flake8

# Formatting
poetry run black .
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Documentation

- [Project Overview](docs/PROJECT.md) - Detailed project briefing
- [Development Plans](docs/) - Phase-by-phase implementation plans
- [Architecture Guide](docs/architecture.md) - System architecture details
- [Setup Guide](docs/dev_setup.md) - Development environment setup

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Project Status

🚧 **In Development** - Phase 1: Core LLM & Main Agent System implemented (UI, LLM, agent system, configuration, and tests complete). Now preparing for Phase 2: Memory & Data Persistence.

### Development Phases

- **Phase 0**: Planning & Foundation
- **Phase 1**: Core LLM & Main Agent System *(Current)*
- **Phase 2**: Memory & Data Persistence
- **Phase 3**: Agent Management & Core Sub-Agents
- **Phase 4**: Image Generation & Advanced Features
- **Phase 5**: Polish, Testing & Deployment

## Support

For questions, issues, or contributions, please open an issue on GitHub or contact the development team.
