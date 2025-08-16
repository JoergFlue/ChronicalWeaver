# Chronicle Weaver - Detailed Implementation Plan

## Project Overview
Chronicle Weaver is an AI-driven roleplaying assistant for Windows 11 featuring a modular agent system, flexible LLM backends, robust memory management, and integrated image generation. This plan ensures each component can be developed and tested independently following TDD principles.

## Development Principles
- **Modularity**: Independent components with clear interfaces
- **Test-Driven Development**: Acceptance tests before implementation
- **Iterative Development**: Each phase delivers working functionality
- **Scalability**: Easy integration of new features
- **Documentation**: Comprehensive inline and external docs

---

# Phase 0: Foundation & Setup (2-4 Days)

## Acceptance Tests
```python
def test_development_environment():
    """Verify development environment is properly configured"""
    assert python_version >= "3.11"
    assert all_dependencies_installed()
    assert git_repository_initialized()
    assert project_structure_exists()

def test_project_structure():
    """Verify project follows modular architecture"""
    assert os.path.exists("src/chronicle_weaver/")
    assert os.path.exists("src/chronicle_weaver/agents/")
    assert os.path.exists("src/chronicle_weaver/memory/")
    assert os.path.exists("src/chronicle_weaver/llm/")
    assert os.path.exists("src/chronicle_weaver/ui/")
    assert os.path.exists("tests/")
    assert os.path.exists("docs/")

def test_core_dependencies():
    """Verify all core dependencies are available"""
    import PyQt6
    import crewai
    import langchain
    import litellm
    import sqlite3
    assert True  # If imports succeed
```

## Implementation Steps

### 1. Development Environment Setup
- [ ] Install Python 3.11+
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Install core dependencies:
  ```bash
  pip install PyQt6 crewai langchain litellm sqlite3
  pip install pytest pytest-qt pytest-asyncio
  pip install black flake8 mypy
  ```

### 2. Project Structure Creation
```
chronicle_weaver/
├── src/
│   └── chronicle_weaver/
│       ├── __init__.py
│       ├── main.py
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base_agent.py
│       │   └── main_agent.py
│       ├── llm/
│       │   ├── __init__.py
│       │   └── llm_manager.py
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── short_term.py
│       │   └── long_term.py
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── main_window.py
│       │   └── tabs/
│       │       ├── __init__.py
│       │       ├── roleplay_tab.py
│       │       ├── agents_tab.py
│       │       ├── library_tab.py
│       │       └── settings_tab.py
│       ├── image_generation/
│       │   ├── __init__.py
│       │   └── image_manager.py
│       └── config/
│           ├── __init__.py
│           └── settings.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
├── requirements.txt
├── pyproject.toml
└── README.md
```

### 3. Base Configuration System
- [ ] Create `src/chronicle_weaver/config/settings.py`:
  ```python
  from dataclasses import dataclass
  from typing import Dict, Any
  
  @dataclass
  class AppSettings:
      data_dir: str
      log_level: str
      theme: str = "light"
      
  @dataclass
  class LLMSettings:
      default_provider: str
      api_keys: Dict[str, str]
      default_temperature: float = 0.7
      max_tokens: int = 500
  ```

### 4. Testing Framework Setup
- [ ] Configure pytest with PyQt6 support
- [ ] Create test fixtures for database, UI components
- [ ] Set up CI/CD pipeline basics

### 5. Documentation Framework
- [ ] Create README.md with project overview
- [ ] Set up inline documentation standards
- [ ] Create developer guide template

## Testing Strategy
- **Unit Tests**: Individual component functionality
- **Integration Tests**: Component interaction
- **Configuration Tests**: Settings loading/saving
- **Environment Tests**: Dependency verification

## Success Criteria
- ✅ Clean project structure following modular design
- ✅ All dependencies properly installed and importable
- ✅ Basic test framework operational
- ✅ Git repository with initial commit
- ✅ Documentation framework in place

---

# Phase 1: Core LLM & Main Agent System (2-3 Weeks)

## Acceptance Tests
```python
def test_llm_manager_initialization():
    """Verify LLM Manager can initialize with different providers"""
    manager = LLMManager()
    assert manager.initialize_provider("openai", {"api_key": "test"})
    assert manager.initialize_provider("ollama", {"base_url": "localhost:11434"})

def test_llm_manager_chat_completion():
    """Verify LLM Manager can generate responses"""
    manager = LLMManager()
    response = manager.chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        provider="mock"
    )
    assert response.content is not None
    assert len(response.content) > 0

def test_main_agent_creation():
    """Verify Main Agent can be created and configured"""
    agent = MainAgent(
        name="TestAgent",
        role="Test Role", 
        backstory="Test backstory"
    )
    assert agent.name == "TestAgent"
    assert agent.role == "Test Role"

def test_main_agent_task_execution():
    """Verify Main Agent can execute basic tasks"""
    agent = MainAgent()
    response = agent.execute_task("Say hello")
    assert response is not None

def test_basic_ui_initialization():
    """Verify basic PyQt6 UI can be created"""
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window is not None
    assert hasattr(window, 'roleplay_tab')
```

## Implementation Steps

### 1. LLM Manager Implementation
- [ ] Create `src/chronicle_weaver/llm/llm_manager.py`:
  ```python
  from typing import Dict, Any, Optional, List
  import litellm
  from dataclasses import dataclass
  
  @dataclass
  class LLMResponse:
      content: str
      provider: str
      model: str
      usage: Dict[str, int]
  
  class LLMManager:
      def __init__(self):
          self.providers: Dict[str, Dict[str, Any]] = {}
          self.default_provider: Optional[str] = None
      
      def initialize_provider(self, name: str, config: Dict[str, Any]) -> bool:
          """Initialize LLM provider with configuration"""
          pass
      
      def chat_completion(self, messages: List[Dict], provider: Optional[str] = None) -> LLMResponse:
          """Generate chat completion using specified or default provider"""
          pass
      
      def list_available_models(self, provider: str) -> List[str]:
          """List available models for a provider"""
          pass
  ```

### 2. Base Agent System
- [ ] Create `src/chronicle_weaver/agents/base_agent.py`:
  ```python
  from abc import ABC, abstractmethod
  from typing import Dict, Any, Optional
  from crewai import Agent
  
  class BaseAgent(ABC):
      def __init__(self, name: str, role: str, backstory: str, **kwargs):
          self.name = name
          self.role = role
          self.backstory = backstory
          self.llm_manager = kwargs.get('llm_manager')
          self.agent = self._create_crew_agent()
      
      def _create_crew_agent(self) -> Agent:
          """Create CrewAI agent with current configuration"""
          pass
      
      @abstractmethod
      def execute_task(self, task_description: str) -> str:
          """Execute a task and return response"""
          pass
  ```

### 3. Main Agent Implementation
- [ ] Create `src/chronicle_weaver/agents/main_agent.py`:
  ```python
  from .base_agent import BaseAgent
  from typing import List, Dict, Any
  
  class MainAgent(BaseAgent):
      def __init__(self, **kwargs):
          super().__init__(
              name="Main Roleplay Agent",
              role="Primary roleplay facilitator",
              backstory="Experienced storyteller and roleplay guide",
              **kwargs
          )
          self.sub_agents: List[BaseAgent] = []
      
      def execute_task(self, task_description: str) -> str:
          """Execute roleplay task, potentially delegating to sub-agents"""
          pass
      
      def add_sub_agent(self, agent: BaseAgent):
          """Add a sub-agent for delegation"""
          pass
      
      def delegate_task(self, task: str, agent_name: str) -> str:
          """Delegate specific task to named sub-agent"""
          pass
  ```

### 4. Basic PyQt6 UI Framework
- [ ] Create `src/chronicle_weaver/ui/main_window.py`:
  ```python
  from PyQt6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget
  from .tabs.roleplay_tab import RoleplayTab
  
  class MainWindow(QMainWindow):
      def __init__(self):
          super().__init__()
          self.setWindowTitle("Chronicle Weaver")
          self.setMinimumSize(1000, 700)
          
          self.setup_ui()
          self.setup_connections()
      
      def setup_ui(self):
          """Initialize UI components"""
          central_widget = QWidget()
          self.setCentralWidget(central_widget)
          
          layout = QVBoxLayout(central_widget)
          
          self.tab_widget = QTabWidget()
          layout.addWidget(self.tab_widget)
          
          # Initialize tabs
          self.roleplay_tab = RoleplayTab()
          self.tab_widget.addTab(self.roleplay_tab, "Roleplay")
      
      def setup_connections(self):
          """Setup signal/slot connections"""
          pass
  ```

### 5. Roleplay Tab Implementation
- [ ] Create `src/chronicle_weaver/ui/tabs/roleplay_tab.py`:
  ```python
  from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                              QTextEdit, QLineEdit, QPushButton, QLabel)
  
  class RoleplayTab(QWidget):
      def __init__(self):
          super().__init__()
          self.setup_ui()
          self.setup_connections()
      
      def setup_ui(self):
          """Setup roleplay tab UI"""
          layout = QVBoxLayout(self)
          
          # Conversation display
          self.conversation_display = QTextEdit()
          self.conversation_display.setReadOnly(True)
          layout.addWidget(self.conversation_display)
          
          # Input area
          input_layout = QHBoxLayout()
          
          self.input_field = QLineEdit()
          self.send_button = QPushButton("Send")
          self.image_button = QPushButton("Generate Image")
          
          input_layout.addWidget(self.input_field)
          input_layout.addWidget(self.send_button)
          input_layout.addWidget(self.image_button)
          
          layout.addLayout(input_layout)
          
          # Status bar
          self.status_label = QLabel("Ready")
          layout.addWidget(self.status_label)
      
      def setup_connections(self):
          """Setup signal connections"""
          self.send_button.clicked.connect(self.send_message)
          self.input_field.returnPressed.connect(self.send_message)
      
      def send_message(self):
          """Handle sending user message"""
          pass
      
      def display_message(self, sender: str, message: str):
          """Display message in conversation"""
          pass
  ```

### 6. Application Entry Point
- [ ] Create `src/chronicle_weaver/main.py`:
  ```python
  import sys
  from PyQt6.QtWidgets import QApplication
  from ui.main_window import MainWindow
  from llm.llm_manager import LLMManager
  from agents.main_agent import MainAgent
  
  class ChronicleWeaverApp:
      def __init__(self):
          self.app = QApplication(sys.argv)
          self.llm_manager = LLMManager()
          self.main_agent = MainAgent(llm_manager=self.llm_manager)
          self.main_window = MainWindow()
      
      def run(self):
          """Start the application"""
          self.main_window.show()
          return self.app.exec()
  
  def main():
      app = ChronicleWeaverApp()
      sys.exit(app.run())
  
  if __name__ == "__main__":
      main()
  ```

## Testing Strategy
- **Unit Tests**: LLM Manager, Agent classes, UI components
- **Integration Tests**: Agent-LLM communication, UI-Agent interaction
- **Mock Testing**: External LLM APIs for reliable testing
- **UI Tests**: PyQt6 widget functionality using pytest-qt

## Success Criteria
- ✅ LLM Manager can connect to multiple providers (with mocks)
- ✅ Main Agent can process basic roleplay requests
- ✅ Basic PyQt6 UI displays and accepts input
- ✅ Agent can generate responses through LLM Manager
- ✅ UI updates display conversation history
- ✅ All components have comprehensive unit tests

---

# Phase 2: Memory & Data Persistence (2-3 Weeks)

## Acceptance Tests
```python
def test_short_term_memory_initialization():
    """Verify short-term memory system initializes correctly"""
    memory = ShortTermMemory()
    assert memory.buffer_size > 0
    assert memory.conversation_buffer is not None

def test_short_term_memory_operations():
    """Verify short-term memory can store and retrieve messages"""
    memory = ShortTermMemory()
    memory.add_message("user", "Hello")
    memory.add_message("assistant", "Hi there!")
    
    messages = memory.get_recent_messages(count=2)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"

def test_long_term_memory_database():
    """Verify long-term memory database operations"""
    db = LongTermMemory(":memory:")  # In-memory for testing
    
    # Test conversation summaries
    summary_id = db.store_conversation_summary("Test summary", {"key": "value"})
    assert summary_id is not None
    
    summary = db.get_conversation_summary(summary_id)
    assert summary["content"] == "Test summary"

def test_agent_configuration_crud():
    """Verify agent configuration CRUD operations"""
    db = LongTermMemory(":memory:")
    
    config = {
        "name": "Test Agent",
        "role": "Tester", 
        "backstory": "A test agent",
        "system_prompt": "You are a test agent"
    }
    
    agent_id = db.create_agent_config(config)
    assert agent_id is not None
    
    retrieved = db.get_agent_config(agent_id)
    assert retrieved["name"] == "Test Agent"
    
    # Test update
    config["role"] = "Updated Tester"
    db.update_agent_config(agent_id, config)
    
    updated = db.get_agent_config(agent_id)
    assert updated["role"] == "Updated Tester"

def test_prop_library_operations():
    """Verify prop/clothing library CRUD operations"""
    db = LongTermMemory(":memory:")
    
    item = {
        "name": "Magic Sword",
        "category": "weapon",
        "description": "A glowing sword",
        "image_path": "/path/to/image.jpg"
    }
    
    item_id = db.create_library_item(item)
    assert item_id is not None
    
    retrieved = db.get_library_item(item_id)
    assert retrieved["name"] == "Magic Sword"

def test_memory_integration_with_agent():
    """Verify agent can use both memory systems"""
    short_memory = ShortTermMemory()
    long_memory = LongTermMemory(":memory:")
    
    agent = MainAgent(
        short_term_memory=short_memory,
        long_term_memory=long_memory
    )
    
    # Agent should be able to access memory
    assert agent.short_term_memory is not None
    assert agent.long_term_memory is not None
```

## Implementation Steps

### 1. Short-Term Memory System
- [ ] Create `src/chronicle_weaver/memory/short_term.py`:
  ```python
  from langchain.memory import ConversationBufferWindowMemory
  from typing import List, Dict, Any, Optional
  
  class ShortTermMemory:
      def __init__(self, buffer_size: int = 10):
          self.buffer_size = buffer_size
          self.conversation_buffer = ConversationBufferWindowMemory(
              k=buffer_size,
              return_messages=True
          )
      
      def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
          """Add message to short-term memory"""
          pass
      
      def get_recent_messages(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
          """Get recent messages from buffer"""
          pass
      
      def clear_buffer(self):
          """Clear the conversation buffer"""
          pass
      
      def get_buffer_size(self) -> int:
          """Get current buffer size"""
          pass
      
      def set_buffer_size(self, size: int):
          """Update buffer size"""
          pass
  ```

### 2. Long-Term Memory Database Schema
- [ ] Create `src/chronicle_weaver/memory/database_schema.sql`:
  ```sql
  -- Conversation summaries table
  CREATE TABLE IF NOT EXISTS conversation_summaries (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      summary_text TEXT NOT NULL,
      metadata TEXT,  -- JSON metadata
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  
  -- Agent configurations table
  CREATE TABLE IF NOT EXISTS agent_configs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT UNIQUE NOT NULL,
      role TEXT NOT NULL,
      backstory TEXT,
      system_prompt TEXT,
      llm_model TEXT,
      temperature REAL DEFAULT 0.7,
      max_tokens INTEGER DEFAULT 500,
      enabled BOOLEAN DEFAULT 1,
      is_main_agent BOOLEAN DEFAULT 0,
      tools_enabled TEXT,  -- JSON list of enabled tools
      memory_access_level TEXT DEFAULT 'full',
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  
  -- Library items (props/clothing) table
  CREATE TABLE IF NOT EXISTS library_items (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      category TEXT NOT NULL,
      description TEXT,
      image_path TEXT,
      metadata TEXT,  -- JSON metadata
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  
  -- Character states table
  CREATE TABLE IF NOT EXISTS character_states (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      traits TEXT,  -- JSON traits
      current_clothing TEXT,  -- JSON clothing items
      relationships TEXT,  -- JSON relationships
      location TEXT,
      metadata TEXT,  -- Additional JSON metadata
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  
  -- Session history table
  CREATE TABLE IF NOT EXISTS session_history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_name TEXT,
      agent_configs_snapshot TEXT,  -- JSON snapshot of agent configs
      conversation_summary TEXT,
      started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      ended_at TIMESTAMP
  );
  ```

### 3. Long-Term Memory Implementation
- [ ] Create `src/chronicle_weaver/memory/long_term.py`:
  ```python
  import sqlite3
  import json
  from typing import Dict, Any, List, Optional
  from pathlib import Path
  
  class LongTermMemory:
      def __init__(self, db_path: str = "chronicle_weaver.db"):
          self.db_path = db_path
          self.init_database()
      
      def init_database(self):
          """Initialize database with schema"""
          pass
      
      def store_conversation_summary(self, summary: str, metadata: Dict[str, Any]) -> int:
          """Store conversation summary with metadata"""
          pass
      
      def get_conversation_summary(self, summary_id: int) -> Optional[Dict[str, Any]]:
          """Retrieve conversation summary by ID"""
          pass
      
      def search_conversation_summaries(self, query: str) -> List[Dict[str, Any]]:
          """Search conversation summaries by content"""
          pass
      
      # Agent Configuration CRUD
      def create_agent_config(self, config: Dict[str, Any]) -> int:
          """Create new agent configuration"""
          pass
      
      def get_agent_config(self, agent_id: int) -> Optional[Dict[str, Any]]:
          """Get agent configuration by ID"""
          pass
      
      def get_all_agent_configs(self) -> List[Dict[str, Any]]:
          """Get all agent configurations"""
          pass
      
      def update_agent_config(self, agent_id: int, config: Dict[str, Any]) -> bool:
          """Update agent configuration"""
          pass
      
      def delete_agent_config(self, agent_id: int) -> bool:
          """Delete agent configuration"""
          pass
      
      # Library Item CRUD
      def create_library_item(self, item: Dict[str, Any]) -> int:
          """Create new library item"""
          pass
      
      def get_library_item(self, item_id: int) -> Optional[Dict[str, Any]]:
          """Get library item by ID"""
          pass
      
      def get_library_items_by_category(self, category: str) -> List[Dict[str, Any]]:
          """Get library items by category"""
          pass
      
      def search_library_items(self, query: str) -> List[Dict[str, Any]]:
          """Search library items by name or description"""
          pass
      
      def update_library_item(self, item_id: int, item: Dict[str, Any]) -> bool:
          """Update library item"""
          pass
      
      def delete_library_item(self, item_id: int) -> bool:
          """Delete library item"""
          pass
  ```

### 4. Memory Integration Layer
- [ ] Create `src/chronicle_weaver/memory/__init__.py`:
  ```python
  from .short_term import ShortTermMemory
  from .long_term import LongTermMemory
  
  class MemoryManager:
      def __init__(self, db_path: str = "chronicle_weaver.db", buffer_size: int = 10):
          self.short_term = ShortTermMemory(buffer_size)
          self.long_term = LongTermMemory(db_path)
      
      def add_conversation_turn(self, user_message: str, agent_response: str, metadata: Dict[str, Any] = None):
          """Add complete conversation turn to short-term memory"""
          pass
      
      def summarize_and_store_session(self, session_name: str = None) -> int:
          """Summarize current session and store in long-term memory"""
          pass
      
      def get_conversation_context(self, include_summaries: bool = True) -> List[Dict[str, Any]]:
          """Get full conversation context for agent"""
          pass
      
      def clear_short_term_memory(self):
          """Clear short-term memory buffer"""
          pass
  ```

### 5. Agent Memory Integration
- [ ] Update `src/chronicle_weaver/agents/main_agent.py`:
  ```python
  # Add to MainAgent class
  def __init__(self, **kwargs):
      super().__init__(**kwargs)
      self.memory_manager = kwargs.get('memory_manager')
      self.use_memory = kwargs.get('use_memory', True)
  
  def execute_task(self, task_description: str) -> str:
      """Execute task with memory context"""
      if self.use_memory and self.memory_manager:
          # Get conversation context
          context = self.memory_manager.get_conversation_context()
          
          # Include context in task execution
          response = self._execute_with_context(task_description, context)
          
          # Store conversation turn
          self.memory_manager.add_conversation_turn(task_description, response)
          
          return response
      else:
          return self._execute_without_memory(task_description)
  ```

### 6. Configuration Management
- [ ] Update `src/chronicle_weaver/config/settings.py`:
  ```python
  @dataclass
  class MemorySettings:
      database_path: str = "chronicle_weaver.db"
      short_term_buffer_size: int = 10
      auto_summarize_threshold: int = 50  # messages before auto-summarize
      enable_long_term_memory: bool = True
      backup_interval_hours: int = 24
  ```

## Testing Strategy
- **Unit Tests**: Individual memory operations, CRUD functions
- **Integration Tests**: Memory-Agent interaction, database consistency
- **Performance Tests**: Large conversation handling, query performance
- **Migration Tests**: Database schema updates and data migration

## Success Criteria
- ✅ Short-term memory maintains conversation context
- ✅ Long-term memory persists data across sessions
- ✅ Agent configurations can be saved/loaded
- ✅ Library items can be managed through CRUD operations
- ✅ Memory systems integrate seamlessly with agents
- ✅ Database performs well with realistic data volumes
- ✅ All memory operations have comprehensive test coverage

---

# Phase 3: Agent Management & Core Sub-Agents (3-4 Weeks)

## Acceptance Tests
```python
def test_agent_configuration_ui():
    """Verify agent configuration UI functionality"""
    app = QApplication.instance() or QApplication([])
    agents_tab = AgentsTab()
    
    # Test agent list display
    assert agents_tab.agent_list_widget is not None
    
    # Test configuration form
    assert agents_tab.config_form is not None
    assert hasattr(agents_tab.config_form, 'name_field')
    assert hasattr(agents_tab.config_form, 'role_field')
    assert hasattr(agents_tab.config_form, 'system_prompt_field')

def test_agent_creation_workflow():
    """Verify complete agent creation workflow"""
    memory = LongTermMemory(":memory:")
    agent_manager = AgentManager(memory)
    
    config = {
        "name": "Test Agent",
        "role": "Tester",
        "backstory": "Test backstory",
        "system_prompt": "You are a test agent",
        "enabled": True
    }
    
    # Create agent
    agent = agent_manager.create_agent(config)
    assert agent is not None
    assert agent.name == "Test Agent"
    
    # Verify persistence
    saved_config = memory.get_agent_config(agent.id)
    assert saved_config["name"] == "Test Agent"

def test_sub_agent_delegation():
    """Verify main agent can delegate to sub-agents"""
    main_agent = MainAgent()
    sub_agent = PromptTrackingAgent()
    
    main_agent.add_sub_agent(sub_agent)
    
    # Test delegation
    result = main_agent.delegate_task(
        "Track this prompt: Hello world", 
        "PromptTrackingAgent"
    )
    
    assert result is not None
    assert "tracked" in result.lower()

def test_prompt_tracking_agent():
    """Verify prompt tracking agent functionality"""
    agent = PromptTrackingAgent()
    memory = LongTermMemory(":memory:")
    agent.memory = memory
    
    # Test prompt logging
    response = agent.execute_task("Log this prompt: Test message")
    assert response is not None
    
    # Verify logged in database
    logs = memory.get_prompt_logs()
    assert len(logs) > 0

def test_continuity_check_agent():
    """Verify continuity check agent functionality"""
    agent = ContinuityCheckAgent()
    memory = LongTermMemory(":memory:")
    agent.memory = memory
    
    # Setup some conversation history
    memory.store_conversation_summary("Character has blue eyes", {})
    
    # Test continuity check
    response = agent.execute_task(
        "Check if 'Character now has green eyes' is consistent"
    )
    
    assert response is not None
    assert "inconsistent" in response.lower()

def test_agent_manager_delegation_routing():
    """Verify agent manager routes tasks correctly"""
    manager = AgentManager()
    
    # Setup agents
    main_agent = manager.create_main_agent()
    manager.create_sub_agent("PromptTrackingAgent", {})
    
    # Test routing
    response = manager.process_user_input("Hello, track this message")
    assert response is not None
```

## Implementation Steps

### 1. Agent Management System
- [ ] Create `src/chronicle_weaver/agents/agent_manager.py`:
  ```python
  from typing import Dict, List, Optional, Any
  from .main_agent import MainAgent
  from .sub_agents.prompt_tracking_agent import PromptTrackingAgent
  from .sub_agents.continuity_check_agent import ContinuityCheckAgent
  from ..memory import MemoryManager
  
  class AgentManager:
      def __init__(self, memory_manager: MemoryManager = None):
          self.memory_manager = memory_manager
          self.main_agent: Optional[MainAgent] = None
          self.sub_agents: Dict[str, BaseAgent] = {}
          self.agent_configs: Dict[str, Dict[str, Any]] = {}
      
      def create_main_agent(self, config: Dict[str, Any] = None) -> MainAgent:
          """Create and configure main agent"""
          pass
      
      def create_sub_agent(self, agent_type: str, config: Dict[str, Any]) -> BaseAgent:
          """Create sub-agent of specified type"""
          pass
      
      def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
          """Get agent by name"""
          pass
      
      def list_agents(self) -> List[Dict[str, Any]]:
          """List all configured agents"""
          pass
      
      def enable_agent(self, agent_name: str, enabled: bool = True):
          """Enable or disable an agent"""
          pass
      
      def process_user_input(self, user_input: str) -> str:
          """Process user input through agent system"""
          pass
      
      def save_agent_configs(self):
          """Save all agent configurations to long-term memory"""
          pass
      
      def load_agent_configs(self):
          """Load agent configurations from long-term memory"""
          pass
  ```

### 2. Agents Tab UI Implementation
- [ ] Create `src/chronicle_weaver/ui/tabs/agents_tab.py`:
  ```python
  from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QListWidget, 
                              QFormLayout, QLineEdit, QTextEdit, QComboBox,
                              QCheckBox, QSlider, QSpinBox, QPushButton,
                              QGroupBox, QLabel, QSplitter)
  from PyQt6.QtCore import pyqtSignal
  
  class AgentsTab(QWidget):
      agent_config_changed = pyqtSignal(str, dict)  # agent_name, config
      
      def __init__(self):
          super().__init__()
          self.current_agent_name = None
          self.setup_ui()
          self.setup_connections()
      
      def setup_ui(self):
          """Setup agents tab UI"""
          layout = QHBoxLayout(self)
          
          # Create splitter
          splitter = QSplitter()
          layout.addWidget(splitter)
          
          # Left panel - Agent list
          self.setup_agent_list_panel(splitter)
          
          # Right panel - Agent configuration
          self.setup_agent_config_panel(splitter)
      
      def setup_agent_list_panel(self, parent):
          """Setup left panel with agent list"""
          panel = QWidget()
          layout = QVBoxLayout(panel)
          
          # Agent list
          layout.addWidget(QLabel("Agents"))
          self.agent_list_widget = QListWidget()
          layout.addWidget(self.agent_list_widget)
          
          # Control buttons
          button_layout = QHBoxLayout()
          self.new_agent_button = QPushButton("New Agent")
          self.delete_agent_button = QPushButton("Delete")
          self.clone_agent_button = QPushButton("Clone")
          
          button_layout.addWidget(self.new_agent_button)
          button_layout.addWidget(self.delete_agent_button)
          button_layout.addWidget(self.clone_agent_button)
          layout.addLayout(button_layout)
          
          parent.addWidget(panel)
      
      def setup_agent_config_panel(self, parent):
          """Setup right panel with agent configuration"""
          panel = QWidget()
          layout = QVBoxLayout(panel)
          
          # Basic Information Group
          self.setup_basic_info_group(layout)
          
          # AI Configuration Group
          self.setup_ai_config_group(layout)
          
          # Tools & Capabilities Group
          self.setup_tools_group(layout)
          
          # Status & Control Group
          self.setup_status_group(layout)
          
          # Save/Reset buttons
          button_layout = QHBoxLayout()
          self.save_button = QPushButton("Save Configuration")
          self.reset_button = QPushButton("Reset to Saved")
          
          button_layout.addWidget(self.save_button)
          button_layout.addWidget(self.reset_button)
          layout.addLayout(button_layout)
          
          parent.addWidget(panel)
      
      def setup_basic_info_group(self, parent_layout):
          """Setup basic information group"""
          group = QGroupBox("Basic Information")
          layout = QFormLayout()
          
          self.name_field = QLineEdit()
          self.role_field = QLineEdit()
          self.backstory_field = QTextEdit()
          self.backstory_field.setMaximumHeight(100)
          
          layout.addRow("Name:", self.name_field)
          layout.addRow("Role:", self.role_field)
          layout.addRow("Backstory:", self.backstory_field)
          
          group.setLayout(layout)
          parent_layout.addWidget(group)
      
      def setup_ai_config_group(self, parent_layout):
          """Setup AI configuration group"""
          group = QGroupBox("AI Configuration")
          layout = QFormLayout()
          
          self.system_prompt_field = QTextEdit()
          self.system_prompt_field.setMaximumHeight(120)
          
          self.llm_model_combo = QComboBox()
          self.llm_model_combo.addItems(["Global Default", "GPT-4", "Claude-3", "Llama3"])
          
          self.temperature_slider = QSlider()
          self.temperature_slider.setOrientation(1)  # Horizontal
          self.temperature_slider.setRange(0, 100)
          self.temperature_slider.setValue(70)
          
          self.max_tokens_spin = QSpinBox()
          self.max_tokens_spin.setRange(50, 4000)
          self.max_tokens_spin.setValue(500)
          
          layout.addRow("System Prompt:", self.system_prompt_field)
          layout.addRow("LLM Model:", self.llm_model_combo)
          layout.addRow("Temperature:", self.temperature_slider)
          layout.addRow("Max Tokens:", self.max_tokens_spin)
          
          group.setLayout(layout)
          parent_layout.addWidget(group)
      
      def setup_tools_group(self, parent_layout):
          """Setup tools and capabilities group"""
          group = QGroupBox("Tools & Capabilities")
          layout = QVBoxLayout()
          
          self.web_search_check = QCheckBox("Web Search")
          self.image_gen_check = QCheckBox("Image Generation")
          self.prop_library_check = QCheckBox("Access Prop/Clothing Library")
          self.memory_access_check = QCheckBox("Read/Write Long-Term Memory")
          self.calculation_check = QCheckBox("Calculate")
          
          layout.addWidget(self.web_search_check)
          layout.addWidget(self.image_gen_check)
          layout.addWidget(self.prop_library_check)
          layout.addWidget(self.memory_access_check)
          layout.addWidget(self.calculation_check)
          
          group.setLayout(layout)
          parent_layout.addWidget(group)
      
      def setup_status_group(self, parent_layout):
          """Setup status and control group"""
          group = QGroupBox("Status & Control")
          layout = QFormLayout()
          
          self.enabled_check = QCheckBox()
          self.main_agent_check = QCheckBox()
          
          self.activity_level_combo = QComboBox()
          self.activity_level_combo.addItems(["High", "Medium", "Low", "Passive"])
          
          layout.addRow("Enabled:", self.enabled_check)
          layout.addRow("Main Interface Agent:", self.main_agent_check)
          layout.addRow("Activity Level:", self.activity_level_combo)
          
          group.setLayout(layout)
          parent_layout.addWidget(group)
      
      def setup_connections(self):
          """Setup signal connections"""
          self.agent_list_widget.currentTextChanged.connect(self.load_agent_config)
          self.save_button.clicked.connect(self.save_current_config)
          self.new_agent_button.clicked.connect(self.create_new_agent)
          self.delete_agent_button.clicked.connect(self.delete_current_agent)
      
      def load_agent_config(self, agent_name: str):
          """Load configuration for selected agent"""
          pass
      
      def save_current_config(self):
          """Save current agent configuration"""
          pass
      
      def create_new_agent(self):
          """Create new agent"""
          pass
      
      def delete_current_agent(self):
          """Delete currently selected agent"""
          pass
      
      def get_current_config(self) -> Dict[str, Any]:
          """Get configuration from UI fields"""
          pass
      
      def set_config_fields(self, config: Dict[str, Any]):
          """Set UI fields from configuration"""
          pass
  ```

### 3. Sub-Agent Implementations

#### 3.1 Prompt Tracking Agent
- [ ] Create `src/chronicle_weaver/agents/sub_agents/prompt_tracking_agent.py`:
  ```python
  from ..base_agent import BaseAgent
  from typing import Dict, Any
  import json
  from datetime import datetime
  
  class PromptTrackingAgent(BaseAgent):
      def __init__(self, **kwargs):
          super().__init__(
              name="Prompt Tracking Agent",
              role="Logs and tracks user prompts and agent responses",
              backstory="A meticulous recorder that maintains detailed logs of all interactions",
              **kwargs
          )
      
      def execute_task(self, task_description: str) -> str:
          """Log prompt and return confirmation"""
          try:
              # Extract prompt from task description
              prompt_data = self._extract_prompt_data(task_description)
              
              # Store in memory
              if self.memory:
                  self.memory.store_prompt_log(prompt_data)
              
              return f"Logged prompt: {prompt_data['type']} from {prompt_data['source']}"
              
          except Exception as e:
              return f"Error logging prompt: {str(e)}"
      
      def _extract_prompt_data(self, task_description: str) -> Dict[str, Any]:
          """Extract structured prompt data from task description"""
          return {
              "timestamp": datetime.now().isoformat(),
              "content": task_description,
              "type": "user_prompt",  # or "agent_response"
              "source": "user",
              "metadata": {}
          }
      
      def get_recent_logs(self, count: int = 10) -> List[Dict[str, Any]]:
          """Get recent prompt logs"""
          if self.memory:
              return self.memory.get_recent_prompt_logs(count)
          return []
  ```

#### 3.2 Continuity Check Agent
- [ ] Create `src/chronicle_weaver/agents/sub_agents/continuity_check_agent.py`:
  ```python
  from ..base_agent import BaseAgent
  from typing import List, Dict, Any
  
  class ContinuityCheckAgent(BaseAgent):
      def __init__(self, **kwargs):
          super().__init__(
              name="Continuity Check Agent",
              role="Maintains narrative consistency by checking for contradictions",
              backstory="A detail-oriented guardian of story consistency with excellent memory",
              **kwargs
          )
      
      def execute_task(self, task_description: str) -> str:
          """Check for continuity issues"""
          try:
              # Get recent conversation context
              recent_context = self._get_recent_context()
              
              # Get long-term memory summaries
              long_term_context = self._get_long_term_context()
              
              # Analyze for inconsistencies
              inconsistencies = self._check_consistency(
                  task_description, 
                  recent_context, 
                  long_term_context
              )
              
              if inconsistencies:
                  return self._format_inconsistency_report(inconsistencies)
              else:
                  return "No continuity issues detected."
                  
          except Exception as e:
              return f"Error checking continuity: {str(e)}"
      
      def _get_recent_context(self) -> List[str]:
          """Get recent conversation messages"""
          if self.memory:
              messages = self.memory.get_recent_messages(10)
              return [msg["content"] for msg in messages]
          return []
      
      def _get_long_term_context(self) -> List[str]:
          """Get relevant long-term memory summaries"""
          if self.memory:
              summaries = self.memory.get_recent_summaries(5)
              return [summary["content"] for summary in summaries]
          return []
      
      def _check_consistency(self, new_content: str, recent: List[str], long_term: List[str]) -> List[Dict[str, Any]]:
          """Check for inconsistencies in content"""
          # This would use LLM to analyze for inconsistencies
          # For now, return a placeholder
          return []
      
      def _format_inconsistency_report(self, inconsistencies: List[Dict[str, Any]]) -> str:
          """Format inconsistency findings into readable report"""
          report = "Continuity issues detected:\n"
          for issue in inconsistencies:
              report += f"- {issue['description']}\n"
          return report
  ```

### 4. Enhanced Main Agent with Sub-Agent Integration
- [ ] Update `src/chronicle_weaver/agents/main_agent.py`:
  ```python
  class MainAgent(BaseAgent):
      def __init__(self, **kwargs):
          super().__init__(
              name="Main Roleplay Agent",
              role="Primary roleplay facilitator and task coordinator",
              backstory="Experienced storyteller and coordinator who manages specialized assistants",
              **kwargs
          )
          self.sub_agents: Dict[str, BaseAgent] = {}
          self.delegation_rules: Dict[str, List[str]] = {}
          
      def add_sub_agent(self, agent: BaseAgent):
          """Add a sub-agent for delegation"""
          self.sub_agents[agent.name] = agent
      
      def execute_task(self, task_description: str) -> str:
          """Execute task, potentially delegating to sub-agents"""
          try:
              # Check if task should be delegated
              delegation_target = self._should_delegate(task_description)
              
              if delegation_target:
                  return self.delegate_task(task_description, delegation_target)
              
              # Execute task directly
              response = self._execute_direct_task(task_description)
              
              # Notify relevant sub-agents
              self._notify_sub_agents(task_description, response)
              
              return response
              
          except Exception as e:
              return f"Error executing task: {str(e)}"
      
      def delegate_task(self, task: str, agent_name: str) -> str:
          """Delegate specific task to named sub-agent"""
          if agent_name in self.sub_agents:
              agent = self.sub_agents[agent_name]
              return agent.execute_task(task)
          else:
              return f"Sub-agent '{agent_name}' not found"
      
      def _should_delegate(self, task: str) -> Optional[str]:
          """Determine if task should be delegated and to which agent"""
          # Simple keyword-based delegation logic
          task_lower = task.lower()
          
          if "track" in task_lower or "log" in task_lower:
              return "Prompt Tracking Agent"
          elif "consistent" in task_lower or "continuity" in task_lower:
              return "Continuity Check Agent"
          
          return None
      
      def _notify_sub_agents(self, task: str, response: str):
          """Notify relevant sub-agents of completed task"""
          # Automatically log all interactions
          if "Prompt Tracking Agent" in self.sub_agents:
              self.delegate_task(f"Log interaction: {task} -> {response}", "Prompt Tracking Agent")
  ```

### 5. Database Extensions for Agent Management
- [ ] Update database schema in `src/chronicle_weaver/memory/long_term.py`:
  ```sql
  -- Add to existing schema
  CREATE TABLE IF NOT EXISTS prompt_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      content TEXT NOT NULL,
      type TEXT NOT NULL,  -- 'user_prompt', 'agent_response'
      source TEXT NOT NULL,
      metadata TEXT,  -- JSON metadata
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  
  CREATE TABLE IF NOT EXISTS agent_delegation_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      main_agent_id INTEGER,
      sub_agent_id INTEGER,
      task_description TEXT,
      result TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  ```

## Testing Strategy
- **Unit Tests**: Individual agent functionality, UI components
- **Integration Tests**: Agent delegation, memory integration
- **UI Tests**: Agent configuration workflow using pytest-qt
- **E2E Tests**: Complete agent creation and usage scenarios

## Success Criteria
- ✅ Comprehensive agent configuration UI is functional
- ✅ Main agent can create and manage sub-agents
- ✅ Sub-agents can be delegated tasks appropriately
- ✅ Prompt tracking agent logs all interactions
- ✅ Continuity check agent identifies inconsistencies
- ✅ Agent configurations persist across sessions
- ✅ UI properly loads and saves agent configurations
- ✅ All agent management features have test coverage

---

# Phase 4: Image Generation & Advanced Features (3-4 Weeks)

## Acceptance Tests
```python
def test_image_generation_manager():
    """Verify image generation manager functionality"""
    manager = ImageGenerationManager()
    
    # Test provider initialization
    assert manager.initialize_provider("mock", {"api_key": "test"})
    
    # Test image generation
    result = manager.generate_image(
        prompt="A fantasy castle",
        provider="mock"
    )
    
    assert result.success
    assert result.image_path is not None
    assert result.metadata["provider"] == "mock"

def test_dall_e_integration():
    """Verify DALL-E integration (with mock)"""
    provider = DallEProvider(api_key="mock_key")
    
    result = provider.generate_image(
        prompt="A dragon",
        size="1024x1024",
        quality="standard"
    )
    
    assert result is not None

def test_local_generation_integration():
    """Verify local generation API integration"""
    provider = Automatic1111Provider(base_url="http://localhost:7860")
    
    # This should work with mock/test setup
    if provider.is_available():
        result = provider.generate_image("A test image")
        assert result is not None

def test_search_agent_functionality():
    """Verify search agent can retrieve information"""
    agent = SearchAgent(api_key="mock_key")
    
    result = agent.execute_task("Search for information about dragons in mythology")
    
    assert result is not None
    assert len(result) > 0

def test_alternate_subego_agent():
    """Verify alternate sub-ego agent personality switching"""
    agent = AlternateSubEgoAgent()
    
    # Add personality
    agent.add_personality("cheerful", {
        "system_prompt": "You are cheerful and optimistic",
        "temperature": 0.8
    })
    
    # Switch personality
    agent.set_active_personality("cheerful")
    
    response = agent.execute_task("How are you feeling?")
    assert "cheerful" in response.lower() or "happy" in response.lower()

def test_prop_agent_library_access():
    """Verify prop agent can access and use library items"""
    memory = LongTermMemory(":memory:")
    agent = PropAgent(memory=memory)
    
    # Add test item to library
    item_id = memory.create_library_item({
        "name": "Magic Sword",
        "category": "weapon",
        "description": "A glowing magic sword"
    })
    
    # Agent should be able to access it
    result = agent.execute_task("Equip magic sword")
    assert "magic sword" in result.lower()

def test_library_tab_ui():
    """Verify library tab UI functionality"""
    app = QApplication.instance() or QApplication([])
    library_tab = LibraryTab()
    
    assert library_tab.item_list_widget is not None
    assert library_tab.item_details_panel is not None
    assert library_tab.add_item_button is not None

def test_image_integration_in_roleplay():
    """Verify image generation integrates with roleplay tab"""
    app = QApplication.instance() or QApplication([])
    roleplay_tab = RoleplayTab()
    
    # Mock image generation
    roleplay_tab.image_manager = MockImageManager()
    
    # Test image generation trigger
    roleplay_tab.generate_image_for_context("A fantasy scene")
    
    # Verify image appears in conversation
    assert roleplay_tab.conversation_display.toPlainText().contains("Image:")
```

## Implementation Steps

### 1. Image Generation Manager
- [ ] Create `src/chronicle_weaver/image_generation/image_manager.py`:
  ```python
  from typing import Dict, Any, Optional, List
  from dataclasses import dataclass
  from pathlib import Path
  import uuid
  from datetime import datetime
  
  @dataclass
  class ImageGenerationResult:
      success: bool
      image_path: Optional[str] = None
      error_message: Optional[str] = None
      metadata: Dict[str, Any] = None
  
  class ImageGenerationManager:
      def __init__(self, storage_dir: str = "generated_images"):
          self.storage_dir = Path(storage_dir)
          self.storage_dir.mkdir(exist_ok=True)
          self.providers: Dict[str, 'BaseImageProvider'] = {}
          self.default_provider: Optional[str] = None
      
      def register_provider(self, name: str, provider: 'BaseImageProvider'):
          """Register an image generation provider"""
          self.providers[name] = provider
      
      def initialize_provider(self, name: str, config: Dict[str, Any]) -> bool:
          """Initialize provider with configuration"""
          try:
              if name == "dalle":
                  from .providers.dalle_provider import DallEProvider
                  provider = DallEProvider(**config)
              elif name == "stability":
                  from .providers.stability_provider import StabilityProvider
                  provider = StabilityProvider(**config)
              elif name == "automatic1111":
                  from .providers.automatic1111_provider import Automatic1111Provider
                  provider = Automatic1111Provider(**config)
              else:
                  return False
              
              self.register_provider(name, provider)
              return True
          except Exception:
              return False
      
      def generate_image(self, prompt: str, provider: Optional[str] = None, **kwargs) -> ImageGenerationResult:
          """Generate image using specified or default provider"""
          provider_name = provider or self.default_provider
          
          if not provider_name or provider_name not in self.providers:
              return ImageGenerationResult(
                  success=False,
                  error_message="No provider available"
              )
          
          try:
              provider_instance = self.providers[provider_name]
              image_data = provider_instance.generate_image(prompt, **kwargs)
              
              # Save image to storage
              image_path = self._save_image(image_data, prompt, provider_name)
              
              return ImageGenerationResult(
                  success=True,
                  image_path=str(image_path),
                  metadata={
                      "provider": provider_name,
                      "prompt": prompt,
                      "timestamp": datetime.now().isoformat(),
                      "filename": image_path.name
                  }
              )
              
          except Exception as e:
              return ImageGenerationResult(
                  success=False,
                  error_message=str(e)
              )
      
      def _save_image(self, image_data: bytes, prompt: str, provider: str) -> Path:
          """Save generated image to storage directory"""
          filename = f"{uuid.uuid4()}_{provider}.png"
          image_path = self.storage_dir / filename
          
          with open(image_path, 'wb') as f:
              f.write(image_data)
          
          return image_path
      
      def list_generated_images(self) -> List[Dict[str, Any]]:
          """List all generated images with metadata"""
          images = []
          for image_file in self.storage_dir.glob("*.png"):
              images.append({
                  "path": str(image_file),
                  "filename": image_file.name,
                  "created": datetime.fromtimestamp(image_file.stat().st_ctime)
              })
          return sorted(images, key=lambda x: x["created"], reverse=True)
  ```

### 2. Image Provider Base Class and Implementations
- [ ] Create `src/chronicle_weaver/image_generation/providers/base_provider.py`:
  ```python
  from abc import ABC, abstractmethod
  from typing import Dict, Any
  
  class BaseImageProvider(ABC):
      def __init__(self, **config):
          self.config = config
      
      @abstractmethod
      def generate_image(self, prompt: str, **kwargs) -> bytes:
          """Generate image and return as bytes"""
          pass
      
      @abstractmethod
      def is_available(self) -> bool:
          """Check if provider is available/configured"""
          pass
      
      def validate_config(self) -> bool:
          """Validate provider configuration"""
          return True
  ```

- [ ] Create `src/chronicle_weaver/image_generation/providers/dalle_provider.py`:
  ```python
  from .base_provider import BaseImageProvider
  import openai
  import requests
  from typing import Dict, Any
  
  class DallEProvider(BaseImageProvider):
      def __init__(self, api_key: str, **config):
          super().__init__(api_key=api_key, **config)
          self.client = openai.OpenAI(api_key=api_key)
      
      def generate_image(self, prompt: str, **kwargs) -> bytes:
          """Generate image using DALL-E"""
          size = kwargs.get('size', '1024x1024')
          quality = kwargs.get('quality', 'standard')
          
          response = self.client.images.generate(
              model="dall-e-3",
              prompt=prompt,
              size=size,
              quality=quality,
              n=1,
          )
          
          image_url = response.data[0].url
          
          # Download image
          img_response = requests.get(image_url)
          img_response.raise_for_status()
          
          return img_response.content
      
      def is_available(self) -> bool:
          """Check if DALL-E API is available"""
          try:
              # Test API connection
              self.client.models.list()
              return True
          except Exception:
              return False
  ```

- [ ] Create `src/chronicle_weaver/image_generation/providers/automatic1111_provider.py`:
  ```python
  from .base_provider import BaseImageProvider
  import requests
  import base64
  from typing import Dict, Any
  
  class Automatic1111Provider(BaseImageProvider):
      def __init__(self, base_url: str = "http://localhost:7860", **config):
          super().__init__(base_url=base_url, **config)
          self.base_url = base_url.rstrip('/')
      
      def generate_image(self, prompt: str, **kwargs) -> bytes:
          """Generate image using Automatic1111 API"""
          payload = {
              "prompt": prompt,
              "negative_prompt": kwargs.get('negative_prompt', ''),
              "steps": kwargs.get('steps', 20),
              "sampler_index": kwargs.get('sampler', 'Euler'),
              "cfg_scale": kwargs.get('cfg_scale', 7),
              "width": kwargs.get('width', 512),
              "height": kwargs.get('height', 512),
              "batch_size": 1,
              "n_iter": 1,
          }
          
          response = requests.post(
              f"{self.base_url}/sdapi/v1/txt2img",
              json=payload
          )
          response.raise_for_status()
          
          result = response.json()
          
          # Decode base64 image
          image_data = base64.b64decode(result['images'][0])
          return image_data
      
      def is_available(self) -> bool:
          """Check if Automatic1111 API is available"""
          try:
              response = requests.get(f"{self.base_url}/sdapi/v1/options", timeout=5)
              return response.status_code == 200
          except Exception:
              return False
  ```

### 3. Advanced Sub-Agents

#### 3.1 Search Agent
- [ ] Create `src/chronicle_weaver/agents/sub_agents/search_agent.py`:
  ```python
  from ..base_agent import BaseAgent
  import requests
  from typing import Dict, Any, List
  
  class SearchAgent(BaseAgent):
      def __init__(self, **kwargs):
          super().__init__(
              name="Search Agent",
              role="Retrieves factual information from web searches",
              backstory="A knowledgeable researcher with access to current information",
              **kwargs
          )
          self.api_key = kwargs.get('search_api_key')
          self.search_engine = kwargs.get('search_engine', 'serpapi')
      
      def execute_task(self, task_description: str) -> str:
          """Execute search task and return results"""
          try:
              # Extract search query from task
              query = self._extract_search_query(task_description)
              
              # Perform search
              results = self._perform_search(query)
              
              # Format results
              return self._format_search_results(results)
              
          except Exception as e:
              return f"Search error: {str(e)}"
      
      def _extract_search_query(self, task: str) -> str:
          """Extract search query from task description"""
          # Simple extraction - could be enhanced with NLP
          if "search for" in task.lower():
              return task.split("search for", 1)[1].strip()
          return task
      
      def _perform_search(self, query: str) -> List[Dict[str, Any]]:
          """Perform web search"""
          if self.search_engine == 'serpapi' and self.api_key:
              return self._serpapi_search(query)
          else:
              # Fallback to mock results for testing
              return self._mock_search_results(query)
      
      def _serpapi_search(self, query: str) -> List[Dict[str, Any]]:
          """Search using SerpAPI"""
          url = "https://serpapi.com/search"
          params = {
              "q": query,
              "api_key": self.api_key,
              "engine": "google",
              "num": 5
          }
          
          response = requests.get(url, params=params)
          response.raise_for_status()
          
          data = response.json()
          return data.get('organic_results', [])
      
      def _mock_search_results(self, query: str) -> List[Dict[str, Any]]:
          """Return mock search results for testing"""
