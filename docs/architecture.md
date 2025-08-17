# Chronicle Weaver - System Architecture

## Overview

Chronicle Weaver is designed as a modular, scalable AI roleplaying assistant with a clear separation of concerns across different system components. The architecture follows modern software engineering principles including modularity, testability, and maintainability.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Chronicle Weaver                         │
├─────────────────────────────────────────────────────────────┤
│  UI Layer (PyQt6)                                          │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │  Roleplay   │   Agents    │   Library   │  Settings   │  │
│  │     Tab     │     Tab     │     Tab     │     Tab     │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Application Layer                                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │             Main Controller                             │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Business Logic Layer                                       │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │   Agent     │   Memory    │  LLM        │   Image     │  │
│  │  Manager    │  Manager    │ Manager     │  Generator  │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │   SQLite    │ LangChain   │  LiteLLM    │   File      │  │
│  │ Database    │   Memory    │   APIs      │  Storage    │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Logging Strategy

Chronicle Weaver implements extensive logging across all layers to ensure traceability, debugging, and operational monitoring.

- **Centralized Logging**: All major subsystems (UI, agents, memory, LLM, images) emit structured logs to a unified logging interface.
- **Log Levels**: Supports DEBUG, INFO, WARNING, ERROR, and CRITICAL levels for granular control.
- **Log Destinations**: Logs are written to local files, with optional integration to external log management systems.
- **UI Integration**: A dedicated Logging Tab in the UI allows users to view, filter, and export logs.
- **Error Reporting**: All exceptions and critical failures are logged with stack traces and context information.
- **Audit Trail**: Key user actions and system events are recorded for auditing and troubleshooting.

## Core Components

### 1. UI Layer (`/ui`)

**Technology**: PyQt6  
**Responsibility**: User interface and interaction handling

- **Main Window**: Container for all application functionality
- **Roleplay Tab**: Primary interaction interface for conversations
- **Agents Tab**: Agent configuration and management
- **Library Tab**: Prop and clothing item management
- **Settings Tab**: Application configuration

### 2. Agent System (`/agents`)

**Technology**: CrewAI (when compatible) + Custom Implementation  
**Responsibility**: AI agent management and coordination

- **Main Agent**: Primary conversation handler and task router
- **Sub-Agents**: Specialized agents for specific tasks
  - Prompt Tracking Agent
  - Continuity Check Agent
  - Search Agent
  - Alternate Sub-Ego Agent
  - Prop Agent

### 3. Memory Management (`/memory`)

**Technology**: LangChain + SQLite  
**Responsibility**: Conversation and context persistence

- **Short-term Memory**: Active conversation context
- **Long-term Memory**: Persistent character states, plot points, relationships
- **Memory Interface**: Abstraction layer for memory operations

### 4. LLM Integration (`/llm`)

**Technology**: LiteLLM  
**Responsibility**: Large Language Model abstraction and management

- **LLM Manager**: Unified interface for multiple LLM providers
- **Provider Adapters**: OpenAI, Google Gemini, Ollama, LM Studio
- **Configuration Management**: API keys, model selection, parameters

### 5. Image Generation (`/images`)

**Technology**: Multiple API integrations  
**Responsibility**: Image generation and management

- **Image Manager**: Unified interface for image generation
- **Provider Adapters**: DALL-E 3, Stability AI, Automatic1111, ComfyUI
- **Image Storage**: Local file management and metadata

## Data Flow

### Conversation Flow
1. User input → UI Layer → Main Controller
2. Main Controller → Agent Manager → Main Agent
3. Main Agent evaluates task and delegates to Sub-Agents if needed
4. Agents access Memory Manager for context
5. Agents call LLM Manager for AI responses
6. Response flows back through the chain to UI

### Memory Flow
1. Conversation events → Memory Manager
2. Short-term memory updates (immediate context)
3. Periodic summarization to long-term memory
4. Context retrieval for future conversations

### Configuration Flow
1. Settings UI → Configuration Manager
2. Configuration updates propagated to relevant managers
3. Persistent storage in SQLite database

## Directory Structure

```
chronicle_weaver/
├── __init__.py
├── main.py                 # Application entry point
├── ui/                     # User interface components
│   ├── __init__.py
│   ├── main_window.py      # Main application window
│   ├── roleplay_tab.py     # Conversation interface
│   ├── agents_tab.py       # Agent management
│   ├── library_tab.py      # Prop/clothing management
│   └── settings_tab.py     # Application settings
├── agents/                 # Agent system
│   ├── __init__.py
│   ├── base_agent.py       # Base agent class
│   ├── main_agent.py       # Primary agent
│   ├── sub_agents/         # Specialized agents
│   └── agent_manager.py    # Agent coordination
├── memory/                 # Memory management
│   ├── __init__.py
│   ├── memory_manager.py   # Main memory interface
│   ├── short_term.py       # Conversation memory
│   └── long_term.py        # Persistent memory
├── llm/                    # LLM integration
│   ├── __init__.py
│   ├── llm_manager.py      # LLM abstraction
│   └── providers/          # LLM provider adapters
└── images/                 # Image generation
    ├── __init__.py
    ├── image_manager.py    # Image generation interface
    └── providers/          # Image provider adapters
```

## Design Patterns

### 1. Manager Pattern
Each major subsystem has a manager class that provides a unified interface:
- `AgentManager`: Coordinates agent lifecycle and communication
- `MemoryManager`: Handles all memory operations
- `LLMManager`: Manages LLM provider selection and calls
- `ImageManager`: Coordinates image generation requests

### 2. Strategy Pattern
Provider adapters implement common interfaces:
- `LLMProvider`: Abstract base for LLM integrations
- `ImageProvider`: Abstract base for image generation
- `MemoryProvider`: Abstract base for memory backends

### 3. Observer Pattern
Event-driven updates between components:
- Configuration changes notify relevant managers
- Memory updates trigger UI refreshes
- Agent status changes update UI indicators

## Configuration Management

### Application Configuration
- LLM provider settings (API keys, model selection)
- Image generation preferences
- UI theme and layout preferences
- Memory retention policies

### Agent Configuration
- Agent personality definitions
- Tool access permissions
- Memory access levels
- Delegation relationships

## Security Considerations

### API Key Management
- Secure storage of API keys using system keyring
- Environment variable fallbacks for development
- No hardcoded credentials in source code

### Data Privacy
- Local storage by default (SQLite)
- Optional cloud sync with user consent
- Clear data retention policies

## Testing Strategy

### Integration Testing

- **Mandatory Integration Tests Before and After Each Change**: Every code change must be preceded and followed by integration tests to ensure system stability and correct interactions between components.
- **Continuous Integration**: Automated pipelines run integration tests on every commit and pull request.
- **Database and API Integration**: All database and external API interactions are covered by integration tests using real or mocked services.

### Unit Tests
- Individual component testing
- Mock external dependencies
- PyQt widget testing with pytest-qt

### End-to-End Tests
- Full workflow testing
- UI automation testing
- Performance testing

### Workability Verification

- **main.py Workability Test**: After each test (unit, integration, or end-to-end), the workability of `main.py` must be verified by launching the application and confirming core functionality.
- **Automated Smoke Tests**: Basic startup and UI checks are automated to quickly detect regressions.

## Development Principles

### Modularity
- Clear separation of concerns
- Minimal coupling between components
- Well-defined interfaces

#### UI / Functionality / Configuration Separation
- UI code (PyQt6) must not contain business logic or direct configuration management.
- All configuration and settings operations are handled by dedicated manager modules (e.g., ConfigManager).
- UI components interact with configuration only via manager interfaces.
- Business logic and data management are strictly separated from UI rendering and event handling.

### Testability
- Dependency injection for external services
- Mock-friendly design
- Comprehensive test coverage

### Extensibility
- Plugin architecture for new LLM providers
- Configurable agent behaviors
- Extensible memory backends

### User Experience
- Responsive UI design
- Clear error handling and user feedback
- Intuitive configuration interfaces
