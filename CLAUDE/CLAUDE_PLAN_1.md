# Chronicle Weaver - Phase 1: Core LLM & Main Agent System

**Duration**: 2-3 Weeks  
**Implementation Confidence**: 85% - Medium Risk  
**Dependencies**: Phase 0 (Planning & Foundation)  
**Next Phase**: Phase 2 (Memory & Data Persistence)

## Overview
Implement the core LLM integration layer and Main Agent system that handles direct user interactions. This phase establishes the fundamental conversation flow and multi-LLM backend support that forms the foundation for all AI interactions in the application.

## Key Risk Factors
- **LiteLLM API compatibility** - Different providers may have varying response formats
- **PyQt6 threading issues** - UI blocking during API calls
- **CrewAI integration complexity** - Agent framework learning curve
- **Error handling scope** - Graceful degradation for API failures
- **Response streaming** - Real-time message display challenges

## Acceptance Criteria
- [ ] LiteLLM wrapper successfully connects to OpenAI, Gemini, Ollama, and LM Studio
- [ ] Main Agent can process user input and generate responses
- [ ] Basic PyQt6 UI displays conversation history
- [ ] User can send messages and receive AI responses
- [ ] LLM backend can be switched dynamically
- [ ] Error handling works for API failures and invalid inputs
- [ ] Basic logging system captures all interactions

## Detailed Implementation Steps

### Week 1: LLM Integration Foundation

#### 1.1 LLM Configuration System (`src/llm/llm_config.py`)

```python
"""LLM Configuration Management"""
from dataclasses import dataclass
from typing import Dict, Optional, Any
import json
from pathlib import Path

@dataclass
class LLMConfig:
    """Configuration for a specific LLM provider"""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30
    stream: bool = True
    
class LLMConfigManager:
    """Manages LLM configurations for different providers"""
    
    def __init__(self, config_path: str = "config/llm_configs.json"):
        self.config_path = Path(config_path)
        self.configs = self._load_configs()
    
    def _load_configs(self) -> Dict[str, LLMConfig]:
        """Load LLM configurations from file"""
        if not self.config_path.exists():
            return self._create_default_configs()
        
        with open(self.config_path, 'r') as f:
            data = json.load(f)
        
        return {
            name: LLMConfig(**config) 
            for name, config in data.items()
        }
    
    def _create_default_configs(self) -> Dict[str, LLMConfig]:
        """Create default LLM configurations"""
        defaults = {
            "openai": LLMConfig(
                provider="openai",
                model="gpt-4",
                temperature=0.7
            ),
            "gemini": LLMConfig(
                provider="gemini",
                model="gemini-pro",
                temperature=0.7
            ),
            "ollama": LLMConfig(
                provider="ollama",
                model="llama2",
                base_url="http://localhost:11434",
                temperature=0.7
            ),
            "lm_studio": LLMConfig(
                provider="openai",
                model="local-model",
                base_url="http://localhost:1234/v1",
                temperature=0.7
            )
        }
        self._save_configs(defaults)
        return defaults
    
    def get_config(self, provider_name: str) -> Optional[LLMConfig]:
        """Get configuration for a specific provider"""
        return self.configs.get(provider_name)
    
    def update_config(self, provider_name: str, config: LLMConfig):
        """Update configuration for a provider"""
        self.configs[provider_name] = config
        self._save_configs(self.configs)
    
    def _save_configs(self, configs: Dict[str, LLMConfig]):
        """Save configurations to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        serializable = {
            name: {
                "provider": config.provider,
                "model": config.model,
                "api_key": config.api_key,
                "base_url": config.base_url,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "timeout": config.timeout,
                "stream": config.stream
            }
            for name, config in configs.items()
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(serializable, f, indent=2)
```

#### 1.2 LLM Manager Implementation (`src/llm/llm_manager.py`)

```python
"""LLM Management and Integration"""
import asyncio
import logging
from typing import AsyncGenerator, Optional, Dict, Any
import litellm
from .llm_config import LLMConfigManager, LLMConfig

logger = logging.getLogger(__name__)

class LLMManager:
    """Manages multiple LLM providers using LiteLLM"""
    
    def __init__(self):
        self.config_manager = LLMConfigManager()
        self.current_provider = "openai"  # Default
        self._setup_litellm()
    
    def _setup_litellm(self):
        """Configure LiteLLM settings"""
        litellm.set_verbose = False  # Disable verbose logging
        litellm.drop_params = True   # Drop unsupported parameters
    
    def set_provider(self, provider_name: str) -> bool:
        """Switch to a different LLM provider"""
        if provider_name not in self.config_manager.configs:
            logger.error(f"Provider {provider_name} not configured")
            return False
        
        self.current_provider = provider_name
        logger.info(f"Switched to provider: {provider_name}")
        return True
    
    def get_current_config(self) -> Optional[LLMConfig]:
        """Get the current provider configuration"""
        return self.config_manager.get_config(self.current_provider)
    
    async def generate_response(
        self, 
        messages: list, 
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Generate response from current LLM provider"""
        config = self.get_current_config()
        if not config:
            raise ValueError(f"No configuration for provider: {self.current_provider}")
        
        try:
            # Prepare LiteLLM parameters
            llm_params = self._prepare_llm_params(config, messages, stream)
            
            if stream:
                async for chunk in self._stream_response(llm_params):
                    yield chunk
            else:
                response = await self._get_complete_response(llm_params)
                yield response
                
        except Exception as e:
            logger.error(f"LLM generation error: {str(e)}")
            raise
    
    def _prepare_llm_params(self, config: LLMConfig, messages: list, stream: bool) -> Dict[str, Any]:
        """Prepare parameters for LiteLLM call"""
        params = {
            "model": f"{config.provider}/{config.model}",
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": stream
        }
        
        # Add provider-specific parameters
        if config.api_key:
            params["api_key"] = config.api_key
        if config.base_url:
            params["api_base"] = config.base_url
            
        return params
    
    async def _stream_response(self, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream response from LLM"""
        try:
            response = await litellm.acompletion(**params)
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            raise
    
    async def _get_complete_response(self, params: Dict[str, Any]) -> str:
        """Get complete response from LLM"""
        params["stream"] = False
        response = await litellm.acompletion(**params)
        return response.choices[0].message.content
    
    def test_connection(self, provider_name: str) -> bool:
        """Test connection to a specific provider"""
        try:
            config = self.config_manager.get_config(provider_name)
            if not config:
                return False
            
            # Simple test message
            test_messages = [{"role": "user", "content": "Hello"}]
            params = self._prepare_llm_params(config, test_messages, False)
            
            # Synchronous test call
            response = litellm.completion(**params)
            return bool(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Connection test failed for {provider_name}: {str(e)}")
            return False
```

#### 1.3 Base Agent Framework (`src/agents/base_agent.py`)

```python
"""Base Agent Class and Interface"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class AgentMessage:
    """Represents a message in agent communication"""
    content: str
    role: str = "assistant"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentCapability:
    """Defines an agent capability"""
    name: str
    description: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, 
                 name: str, 
                 system_prompt: str,
                 capabilities: List[AgentCapability] = None):
        self.name = name
        self.system_prompt = system_prompt
        self.capabilities = capabilities or []
        self.enabled = True
        self.conversation_history = []
        self.metadata = {}
        
        logger.info(f"Initialized agent: {self.name}")
    
    @abstractmethod
    async def process_message(self, message: str, context: Dict[str, Any] = None) -> AgentMessage:
        """Process a user message and return agent response"""
        pass
    
    def add_capability(self, capability: AgentCapability):
        """Add a capability to the agent"""
        self.capabilities.append(capability)
        logger.info(f"Added capability '{capability.name}' to agent '{self.name}'")
    
    def remove_capability(self, capability_name: str):
        """Remove a capability from the agent"""
        self.capabilities = [c for c in self.capabilities if c.name != capability_name]
        logger.info(f"Removed capability '{capability_name}' from agent '{self.name}'")
    
    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability"""
        return any(c.name == capability_name and c.enabled for c in self.capabilities)
    
    def update_system_prompt(self, new_prompt: str):
        """Update the agent's system prompt"""
        self.system_prompt = new_prompt
        logger.info(f"Updated system prompt for agent: {self.name}")
    
    def get_conversation_context(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history for context"""
        recent_history = self.conversation_history[-max_messages:]
        return [
            {"role": msg.role, "content": msg.content}
            for msg in recent_history
        ]
    
    def add_to_history(self, message: AgentMessage):
        """Add message to conversation history"""
        self.conversation_history.append(message)
        
        # Keep history size manageable
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-50:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info(f"Cleared history for agent: {self.name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent to dictionary"""
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "enabled": self.enabled,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "enabled": cap.enabled,
                    "config": cap.config
                }
                for cap in self.capabilities
            ],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseAgent':
        """Deserialize agent from dictionary"""
        capabilities = [
            AgentCapability(
                name=cap["name"],
                description=cap["description"],
                enabled=cap["enabled"],
                config=cap["config"]
            )
            for cap in data.get("capabilities", [])
        ]
        
        # Note: This creates a BaseAgent instance, subclasses should override
        agent = cls(
            name=data["name"],
            system_prompt=data["system_prompt"],
            capabilities=capabilities
        )
        agent.enabled = data.get("enabled", True)
        agent.metadata = data.get("metadata", {})
        
        return agent
```

### Week 2: Main Agent Implementation

#### 2.1 Main Agent Class (`src/agents/main_agent.py`)

```python
"""Main Agent - Primary user interaction handler"""
from typing import Dict, Any, Optional
from .base_agent import BaseAgent, AgentMessage, AgentCapability
from ..llm.llm_manager import LLMManager
import logging

logger = logging.getLogger(__name__)

class MainAgent(BaseAgent):
    """Main agent that handles direct user interactions"""
    
    def __init__(self, llm_manager: LLMManager):
        super().__init__(
            name="Main Agent",
            system_prompt=self._get_default_system_prompt(),
            capabilities=[
                AgentCapability(
                    name="conversation",
                    description="Handle general conversation"
                ),
                AgentCapability(
                    name="task_delegation",
                    description="Delegate tasks to sub-agents"
                )
            ]
        )
        self.llm_manager = llm_manager
        self.sub_agents = {}
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the main agent"""
        return """You are the Main Agent for Chronicle Weaver, an AI-driven roleplaying assistant.

Your responsibilities:
1. Handle direct user interactions with warmth and helpfulness
2. Maintain conversation context and flow
3. Delegate specialized tasks to appropriate sub-agents when available
4. Provide roleplaying assistance and creative writing support
5. Remember character details and story continuity

Guidelines:
- Be conversational and engaging
- Ask clarifying questions when needed
- Offer suggestions for roleplay scenarios
- Help develop characters and storylines
- Maintain consistency with established facts

Always prioritize user experience and creative collaboration."""
    
    async def process_message(self, message: str, context: Dict[str, Any] = None) -> AgentMessage:
        """Process user message and generate response"""
        try:
            # Add user message to history
            user_msg = AgentMessage(content=message, role="user")
            self.add_to_history(user_msg)
            
            # Prepare conversation context
            conversation_context = self._prepare_conversation_context(context)
            
            # Generate response using LLM
            response_content = ""
            async for chunk in self.llm_manager.generate_response(
                messages=conversation_context,
                stream=True
            ):
                response_content += chunk
            
            # Create response message
            response = AgentMessage(
                content=response_content.strip(),
                role="assistant",
                metadata={
                    "provider": self.llm_manager.current_provider,
                    "agent": self.name
                }
            )
            
            # Add to history
            self.add_to_history(response)
            
            logger.info(f"Main Agent processed message, response length: {len(response_content)}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            error_response = AgentMessage(
                content="I'm sorry, I encountered an error processing your message. Please try again.",
                role="assistant",
                metadata={"error": str(e)}
            )
            return error_response
    
    def _prepare_conversation_context(self, additional_context: Dict[str, Any] = None) -> list:
        """Prepare conversation context for LLM"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add recent conversation history
        messages.extend(self.get_conversation_context(max_messages=20))
        
        # Add any additional context
        if additional_context:
            context_str = self._format_additional_context(additional_context)
            if context_str:
                messages.append({
                    "role": "system", 
                    "content": f"Additional context: {context_str}"
                })
        
        return messages
    
    def _format_additional_context(self, context: Dict[str, Any]) -> str:
        """Format additional context into a string"""
        formatted_parts = []
        
        if "character_info" in context:
            formatted_parts.append(f"Character: {context['character_info']}")
        
        if "scene_setting" in context:
            formatted_parts.append(f"Setting: {context['scene_setting']}")
        
        if "recent_events" in context:
            formatted_parts.append(f"Recent events: {context['recent_events']}")
        
        return " | ".join(formatted_parts)
    
    def register_sub_agent(self, agent_name: str, agent: BaseAgent):
        """Register a sub-agent for task delegation"""
        self.sub_agents[agent_name] = agent
        logger.info(f"Registered sub-agent: {agent_name}")
    
    def unregister_sub_agent(self, agent_name: str):
        """Unregister a sub-agent"""
        if agent_name in self.sub_agents:
            del self.sub_agents[agent_name]
            logger.info(f"Unregistered sub-agent: {agent_name}")
    
    async def delegate_to_sub_agent(self, agent_name: str, message: str, context: Dict[str, Any] = None) -> Optional[AgentMessage]:
        """Delegate a task to a specific sub-agent"""
        if agent_name not in self.sub_agents:
            logger.warning(f"Sub-agent {agent_name} not found")
            return None
        
        try:
            sub_agent = self.sub_agents[agent_name]
            if not sub_agent.enabled:
                logger.warning(f"Sub-agent {agent_name} is disabled")
                return None
            
            response = await sub_agent.process_message(message, context)
            logger.info(f"Delegated task to {agent_name}, received response")
            return response
            
        except Exception as e:
            logger.error(f"Error delegating to sub-agent {agent_name}: {str(e)}")
            return None
```

### Week 2-3: Basic UI Implementation

#### 2.2 Main Window (`src/ui/main_window.py`)

```python
"""Main application window"""
from PyQt6.QtWidgets import (QMainWindow, QTabWidget, QVBoxLayout, 
                           QWidget, QMenuBar, QStatusBar, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from .roleplay_tab import RoleplayTab
from ..llm.llm_manager import LLMManager
from ..agents.main_agent import MainAgent
import logging

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """Main application window with tabbed interface"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.llm_manager = LLMManager()
        self.main_agent = MainAgent(self.llm_manager)
        
        self.setWindowTitle("Chronicle Weaver")
        self.setGeometry(100, 100, 1200, 800)
        
        self._setup_ui()
        self._setup_menu_bar()
        self._setup_status_bar()
        
        logger.info("Main window initialized")
    
    def _setup_ui(self):
        """Setup the main UI components"""
        # Create central widget with tab system
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Add roleplay tab
        self.roleplay_tab = RoleplayTab(self.main_agent, self.llm_manager)
        self.tab_widget.addTab(self.roleplay_tab, "Roleplay")
        
        # Placeholder for future tabs
        self.tab_widget.addTab(QWidget(), "Agents")
        self.tab_widget.addTab(QWidget(), "Library")
        self.tab_widget.addTab(QWidget(), "Settings")
    
    def _setup_menu_bar(self):
        """Setup the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Conversation", self)
        new_action.triggered.connect(self._new_conversation)
        file_menu.addAction(new_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # LLM menu
        llm_menu = menubar.addMenu("LLM")
        
        providers = ["openai", "gemini", "ollama", "lm_studio"]
        for provider in providers:
            action = QAction(provider.replace("_", " ").title(), self)
            action.triggered.connect(lambda checked, p=provider: self._switch_llm_provider(p))
            llm_menu.addAction(action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_status_bar(self):
        """Setup the status bar"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Show current LLM provider
        self._update_status_bar()
    
    def _update_status_bar(self):
        """Update status bar with current LLM provider"""
        provider = self.llm_manager.current_provider
        self.status_bar.showMessage(f"Ready - LLM: {provider.replace('_', ' ').title()}")
    
    def _new_conversation(self):
        """Start a new conversation"""
        if hasattr(self.roleplay_tab, 'clear_conversation'):
            self.roleplay_tab.clear_conversation()
        logger.info("Started new conversation")
    
    def _switch_llm_provider(self, provider: str):
        """Switch to a different LLM provider"""
        success = self.llm_manager.set_provider(provider)
        if success:
            self._update_status_bar()
            logger.info(f"Switched to LLM provider: {provider}")
        else:
            QMessageBox.warning(
                self, 
                "Provider Switch Failed", 
                f"Could not switch to provider: {provider}"
            )
    
    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Chronicle Weaver",
            "Chronicle Weaver v0.1.0\n\n"
            "An AI-driven roleplaying assistant with modular agent systems, "
            "flexible LLM backends, and integrated image generation."
        )
```

#### 2.3 Roleplay Tab (`src/ui/roleplay_tab.py`)

```python
"""Roleplay conversation tab"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                           QTextEdit, QLineEdit, QPushButton, 
                           QSplitter, QScrollArea, QLabel)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor
from ..agents.main_agent import MainAgent
from ..llm.llm_manager import LLMManager
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
    
    async def _generate_response(self):
        """Generate response asynchronously"""
        try:
            response = await self.agent.process_message(self.message)
            # Emit the complete response
            self.chunk_received.emit(response.content)
            self.generation_complete.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def run(self):
        """Run the generation in thread"""
        import asyncio
        asyncio.run(self._generate_response())

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
        
        # Create splitter for conversation area and input
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)
        
        # Conversation display area
        self.conversation_area = QTextEdit()
        self.conversation_area.setReadOnly(True)
        self.conversation_area.setFont(QFont("Segoe UI", 10))
        splitter.addWidget(self.conversation_area)
        
        # Input area
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        
        # Message input
        self.message_input = QTextEdit()
        self.message_input.setMaximumHeight(100)
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.setFont(QFont("Segoe UI", 10))
        input_layout.addWidget(self.message_input)
        
        # Button layout
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
        
        # Set splitter proportions
        splitter.setStretchFactor(0, 3)  # Conversation area takes more space
        splitter.setStretchFactor(1, 1)  # Input area takes less space
        
        # Connect Enter key to send message
        self.message_input.keyPressEvent = self._handle_key_press
        
        # Add welcome message
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
        
        # Add user message to conversation
        self._add_user_message(message)
        
        # Clear input
        self.message_input.clear()
        
        # Disable send button during generation
        self.send_button.setEnabled(False)
        self.send_button.setText("Generating...")
        
        # Start AI response generation
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
        
        # Add placeholder for AI response
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
        # Update the display (simplified - in practice might need more sophisticated handling)
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
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#x27;'))
    
    def clear_conversation(self):
        """Clear the conversation history"""
        self.conversation_area.clear()
        self.main_agent.clear_history()
        self._add_system_message("Conversation cleared. Start a new conversation!")
        logger.info("Conversation cleared")
```

## Testing Strategy

### Unit Tests (`tests/unit/`)
- **LLM Manager Tests**: Test provider switching, configuration loading, connection testing
- **Agent Tests**: Test message processing, capability management, history tracking
- **Configuration Tests**: Test config loading, validation, serialization

### Integration Tests (`tests/integration/`)
- **Agent-LLM Integration**: Test agent responses with different LLM providers
- **UI-Agent Integration**: Test UI message flow with agents
- **Error Handling**: Test graceful degradation scenarios

### UI Tests (`tests/ui/`)
- **Main Window Tests**: Test tab switching, menu actions
- **Roleplay Tab Tests**: Test message sending, conversation display
- **Threading Tests**: Test UI responsiveness during AI generation

## Performance Requirements
- **UI Responsiveness**: No blocking during AI generation (background threads)
- **Memory Usage**: < 200MB for basic conversation (excluding LLM model memory)
- **Response Time**: UI updates within 100ms of receiving text chunks

## Error Handling Strategy
- **API Failures**: Graceful error messages, provider fallback options
- **Network Issues**: Retry logic with exponential backoff
- **Threading Errors**: Proper cleanup and error propagation to UI
- **Invalid Input**: Input validation and user feedback

## Success Metrics
- [ ] All LLM providers connect successfully
- [ ] User can send messages and receive responses
- [ ] UI remains responsive during generation
- [ ] Error handling prevents application crashes
- [ ] Provider switching works without restart
- [ ] Conversation history persists during session

## Deliverables
1. **LLM Integration Layer** - Multi-provider support with LiteLLM
2. **Agent Framework** - Base classes and Main Agent implementation
3. **Basic UI** - Functional conversation interface
4. **Threading System** - Non-blocking AI response generation
5. **Configuration System** - LLM provider management
6. **Test Suite** - Comprehensive testing for all components

## Handoff to Phase 2
Phase 1 establishes the core conversation flow and provides:
- Working agent interface for memory integration
- Message flow patterns for persistence
- UI framework for additional tabs
- Error handling patterns for robust operation

Phase 2 can build upon this foundation to add persistent memory and data storage capabilities.
