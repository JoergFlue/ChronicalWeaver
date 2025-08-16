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
