# Chronicle Weaver - Phase 3: Agent Management & Core Sub-Agents

**Duration**: 3-4 Weeks  
**Implementation Confidence**: 70% - High Risk  
**Dependencies**: Phase 2 (Memory & Data Persistence)  
**Next Phase**: Phase 4 (Image Generation & Advanced Features)

## Overview
Implement the comprehensive agent management system with a sophisticated UI for creating, configuring, and managing agents. Develop core sub-agents that provide specialized functionality including prompt tracking, continuity checking, and task delegation. This phase establishes the multi-agent architecture that enables complex roleplaying scenarios.

## Key Risk Factors
- **CrewAI agent orchestration complexity** - Managing multiple agent interactions and task delegation
- **Inter-agent communication protocols** - Ensuring reliable message passing and state synchronization
- **UI complexity for agent configuration** - Building intuitive interfaces for complex agent settings
- **Agent lifecycle management** - Proper creation, startup, shutdown, and cleanup procedures
- **Performance optimization** - Preventing agent conflicts and resource contention

## Acceptance Criteria
- [ ] Agents tab UI allows creating, editing, and deleting agents
- [ ] Agent configurations persist and reload correctly
- [ ] Main Agent can delegate tasks to active sub-agents
- [ ] Prompt Tracking Agent logs all interactions
- [ ] Continuity Check Agent identifies inconsistencies
- [ ] Agents can be enabled/disabled dynamically
- [ ] Agent system prompts generate correctly from configurations
- [ ] Sub-agent delegation works without conflicts
- [ ] Agent performance monitoring works

## Detailed Implementation Steps

### Week 1: Agent Management Infrastructure

#### 1.1 Agent Registry System (`src/agents/agent_registry.py`)

```python
"""Agent registry and lifecycle management"""
from typing import Dict, List, Optional, Type, Any
from .base_agent import BaseAgent, AgentCapability
from ..memory.memory_manager import MemoryManager
from ..llm.llm_manager import LLMManager
import logging
import asyncio
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent status enumeration"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

class AgentInstance:
    """Wrapper for agent instances with metadata"""
    
    def __init__(self, agent: BaseAgent, agent_class: Type[BaseAgent]):
        self.agent = agent
        self.agent_class = agent_class
        self.status = AgentStatus.STOPPED
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.error_count = 0
        self.last_error = None
        self.performance_metrics = {
            "messages_processed": 0,
            "average_response_time": 0.0,
            "success_rate": 1.0
        }

class AgentRegistry:
    """Central registry for all agent types and instances"""
    
    def __init__(self, memory_manager: MemoryManager, llm_manager: LLMManager):
        self.memory_manager = memory_manager
        self.llm_manager = llm_manager
        self.agent_classes = {}  # name -> class
        self.agent_instances = {}  # name -> AgentInstance
        self.agent_dependencies = {}  # name -> [required_agents]
        
        # Register built-in agent types
        self._register_builtin_agents()
        
        logger.info("Agent registry initialized")
    
    def _register_builtin_agents(self):
        """Register built-in agent types"""
        from .prompt_tracking_agent import PromptTrackingAgent
        from .continuity_check_agent import ContinuityCheckAgent
        from .main_agent import MainAgent
        
        self.register_agent_class("main_agent", MainAgent)
        self.register_agent_class("prompt_tracking", PromptTrackingAgent)
        self.register_agent_class("continuity_check", ContinuityCheckAgent)
    
    def register_agent_class(self, name: str, agent_class: Type[BaseAgent], 
                           dependencies: List[str] = None):
        """Register an agent class"""
        self.agent_classes[name] = agent_class
        self.agent_dependencies[name] = dependencies or []
        logger.info(f"Registered agent class: {name}")
    
    def create_agent(self, name: str, agent_type: str, config: Dict[str, Any]) -> bool:
        """Create a new agent instance"""
        try:
            if agent_type not in self.agent_classes:
                logger.error(f"Unknown agent type: {agent_type}")
                return False
            
            if name in self.agent_instances:
                logger.error(f"Agent already exists: {name}")
                return False
            
            # Check dependencies
            dependencies = self.agent_dependencies.get(agent_type, [])
            for dep in dependencies:
                if dep not in self.agent_instances:
                    logger.error(f"Missing dependency {dep} for agent {name}")
                    return False
            
            # Create agent instance
            agent_class = self.agent_classes[agent_type]
            
            # Prepare constructor arguments based on agent type
            if agent_type == "main_agent":
                agent = agent_class(self.llm_manager)
            else:
                agent = agent_class(
                    memory_manager=self.memory_manager,
                    llm_manager=self.llm_manager
                )
            
            # Apply configuration
            self._apply_config(agent, config)
            
            # Wrap in instance container
            instance = AgentInstance(agent, agent_class)
            self.agent_instances[name] = instance
            
            # Save configuration to persistent memory
            config_data = {
                "name": name,
                "display_name": config.get("display_name", name),
                "system_prompt": agent.system_prompt,
                "enabled": True,
                "llm_provider": config.get("llm_provider", "openai"),
                "temperature": config.get("temperature", 0.7),
                "max_tokens": config.get("max_tokens", 2048),
                "capabilities": [cap.name for cap in agent.capabilities],
                "metadata": {"agent_type": agent_type}
            }
            self.memory_manager.save_agent_configuration(config_data)
            
            logger.info(f"Created agent: {name} (type: {agent_type})")
            return True
            
        except Exception as e:
            logger.error(f"Error creating agent {name}: {str(e)}")
            return False
    
    def _apply_config(self, agent: BaseAgent, config: Dict[str, Any]):
        """Apply configuration to agent"""
        if "system_prompt" in config:
            agent.update_system_prompt(config["system_prompt"])
        
        if "capabilities" in config:
            # Clear existing capabilities
            agent.capabilities.clear()
            
            # Add configured capabilities
            for cap_config in config["capabilities"]:
                capability = AgentCapability(
                    name=cap_config["name"],
                    description=cap_config.get("description", ""),
                    enabled=cap_config.get("enabled", True),
                    config=cap_config.get("config", {})
                )
                agent.add_capability(capability)
        
        # Apply other configuration options
        for key, value in config.items():
            if hasattr(agent, key) and key not in ["capabilities", "system_prompt"]:
                setattr(agent, key, value)
    
    def start_agent(self, name: str) -> bool:
        """Start an agent"""
        try:
            if name not in self.agent_instances:
                logger.error(f"Agent not found: {name}")
                return False
            
            instance = self.agent_instances[name]
            
            if instance.status == AgentStatus.RUNNING:
                logger.warning(f"Agent already running: {name}")
                return True
            
            instance.status = AgentStatus.STARTING
            
            # Start the agent (could involve async initialization)
            if hasattr(instance.agent, 'start'):
                result = instance.agent.start()
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            
            instance.status = AgentStatus.RUNNING
            instance.started_at = datetime.utcnow()
            
            logger.info(f"Started agent: {name}")
            return True
            
        except Exception as e:
            if name in self.agent_instances:
                self.agent_instances[name].status = AgentStatus.ERROR
                self.agent_instances[name].last_error = str(e)
                self.agent_instances[name].error_count += 1
            
            logger.error(f"Error starting agent {name}: {str(e)}")
            return False
    
    def stop_agent(self, name: str) -> bool:
        """Stop an agent"""
        try:
            if name not in self.agent_instances:
                logger.error(f"Agent not found: {name}")
                return False
            
            instance = self.agent_instances[name]
            
            if instance.status == AgentStatus.STOPPED:
                logger.warning(f"Agent already stopped: {name}")
                return True
            
            instance.status = AgentStatus.STOPPING
            
            # Stop the agent
            if hasattr(instance.agent, 'stop'):
                result = instance.agent.stop()
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            
            instance.status = AgentStatus.STOPPED
            
            logger.info(f"Stopped agent: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping agent {name}: {str(e)}")
            return False
    
    def remove_agent(self, name: str) -> bool:
        """Remove an agent instance"""
        try:
            if name not in self.agent_instances:
                logger.error(f"Agent not found: {name}")
                return False
            
            # Stop the agent first
            self.stop_agent(name)
            
            # Remove from registry
            del self.agent_instances[name]
            
            logger.info(f"Removed agent: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing agent {name}: {str(e)}")
            return False
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent instance by name"""
        instance = self.agent_instances.get(name)
        return instance.agent if instance else None
    
    def get_agent_status(self, name: str) -> Optional[AgentStatus]:
        """Get agent status by name"""
        instance = self.agent_instances.get(name)
        return instance.status if instance else None
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agent instances with status"""
        return [
            {
                "name": name,
                "type": instance.agent.__class__.__name__,
                "status": instance.status.value,
                "enabled": instance.agent.enabled,
                "created_at": instance.created_at,
                "started_at": instance.started_at,
                "error_count": instance.error_count,
                "performance": instance.performance_metrics
            }
            for name, instance in self.agent_instances.items()
        ]
    
    def get_running_agents(self) -> List[BaseAgent]:
        """Get all running agent instances"""
        return [
            instance.agent 
            for instance in self.agent_instances.values()
            if instance.status == AgentStatus.RUNNING and instance.agent.enabled
        ]
    
    def update_agent_performance(self, name: str, response_time: float, success: bool):
        """Update agent performance metrics"""
        if name in self.agent_instances:
            instance = self.agent_instances[name]
            metrics = instance.performance_metrics
            
            # Update message count
            metrics["messages_processed"] += 1
            
            # Update average response time
            current_avg = metrics["average_response_time"]
            count = metrics["messages_processed"]
            metrics["average_response_time"] = ((current_avg * (count - 1)) + response_time) / count
            
            # Update success rate
            if success:
                metrics["success_rate"] = (metrics["success_rate"] * (count - 1) + 1.0) / count
            else:
                metrics["success_rate"] = (metrics["success_rate"] * (count - 1)) / count
                instance.error_count += 1
    
    def load_agents_from_config(self) -> bool:
        """Load agent configurations from persistent memory"""
        try:
            configs = self.memory_manager.get_agent_configurations()
            
            for config in configs:
                agent_type = config["metadata"].get("agent_type", "main_agent")
                name = config["name"]
                
                # Skip if already exists
                if name in self.agent_instances:
                    continue
                
                # Create agent
                if self.create_agent(name, agent_type, config):
                    if config["enabled"]:
                        self.start_agent(name)
            
            logger.info(f"Loaded {len(configs)} agent configurations")
            return True
            
        except Exception as e:
            logger.error(f"Error loading agent configurations: {str(e)}")
            return False
```

#### 1.2 Agent Communication System (`src/agents/agent_communication.py`)

```python
"""Inter-agent communication and task delegation"""
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import uuid
import logging
from .base_agent import BaseAgent, AgentMessage

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AgentTask:
    """Represents a task for agent execution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requester_agent: str = ""
    target_agent: str = ""
    task_type: str = ""
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[AgentMessage] = None
    error: Optional[str] = None
    timeout_seconds: int = 30

class AgentCommunicationHub:
    """Central hub for agent communication and task delegation"""
    
    def __init__(self):
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}  # task_id -> AgentTask
        self.completed_tasks = {}  # task_id -> AgentTask
        self.agent_registry = None  # Set by agent manager
        self.message_handlers = {}  # agent_name -> List[Callable]
        self.is_running = False
        self._worker_task = None
        
        logger.info("Agent communication hub initialized")
    
    def set_agent_registry(self, registry):
        """Set reference to agent registry"""
        self.agent_registry = registry
    
    async def start(self):
        """Start the communication hub"""
        if self.is_running:
            return
        
        self.is_running = True
        self._worker_task = asyncio.create_task(self._process_tasks())
        logger.info("Agent communication hub started")
    
    async def stop(self):
        """Stop the communication hub"""
        self.is_running = False
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Agent communication hub stopped")
    
    async def delegate_task(self, 
                          requester_agent: str,
                          target_agent: str,
                          task_type: str,
                          message: str,
                          context: Dict[str, Any] = None,
                          priority: TaskPriority = TaskPriority.NORMAL,
                          timeout_seconds: int = 30) -> str:
        """Delegate a task to another agent"""
        
        task = AgentTask(
            requester_agent=requester_agent,
            target_agent=target_agent,
            task_type=task_type,
            message=message,
            context=context or {},
            priority=priority,
            timeout_seconds=timeout_seconds
        )
        
        # Add to queue with priority handling
        await self.task_queue.put((priority.value, task))
        
        logger.info(f"Task delegated: {task.id} from {requester_agent} to {target_agent}")
        return task.id
    
    async def get_task_result(self, task_id: str, timeout: float = None) -> Optional[AgentMessage]:
        """Wait for and return task result"""
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return task.result
        
        # Wait for completion
        start_time = datetime.utcnow()
        while task_id not in self.completed_tasks:
            if timeout and (datetime.utcnow() - start_time).total_seconds() > timeout:
                return None
            
            await asyncio.sleep(0.1)
        
        task = self.completed_tasks[task_id]
        return task.result
    
    async def _process_tasks(self):
        """Process tasks from the queue"""
        while self.is_running:
            try:
                # Get task with priority (higher priority value = higher priority)
                priority, task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Move to active tasks
                self.active_tasks[task.id] = task
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.utcnow()
                
                # Execute task
                await self._execute_task(task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing task: {str(e)}")
    
    async def _execute_task(self, task: AgentTask):
        """Execute a single task"""
        try:
            # Get target agent
            if not self.agent_registry:
                raise Exception("Agent registry not set")
            
            target_agent = self.agent_registry.get_agent(task.target_agent)
            if not target_agent:
                raise Exception(f"Target agent not found: {task.target_agent}")
            
            if not target_agent.enabled:
                raise Exception(f"Target agent disabled: {task.target_agent}")
            
            # Execute task with timeout
            result = await asyncio.wait_for(
                target_agent.process_message(task.message, task.context),
                timeout=task.timeout_seconds
            )
            
            # Task completed successfully
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
            # Update performance metrics
            response_time = (task.completed_at - task.started_at).total_seconds()
            self.agent_registry.update_agent_performance(task.target_agent, response_time, True)
            
            logger.info(f"Task completed: {task.id}")
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = "Task timeout"
            task.completed_at = datetime.utcnow()
            
            # Update performance metrics
            if task.started_at:
                response_time = (task.completed_at - task.started_at).total_seconds()
                self.agent_registry.update_agent_performance(task.target_agent, response_time, False)
            
            logger.warning(f"Task timeout: {task.id}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            
            # Update performance metrics
            if task.started_at:
                response_time = (task.completed_at - task.started_at).total_seconds()
                self.agent_registry.update_agent_performance(task.target_agent, response_time, False)
            
            logger.error(f"Task failed: {task.id} - {str(e)}")
        
        finally:
            # Move to completed tasks
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            self.completed_tasks[task.id] = task
            
            # Cleanup old completed tasks (keep last 1000)
            if len(self.completed_tasks) > 1000:
                oldest_tasks = sorted(
                    self.completed_tasks.items(),
                    key=lambda x: x[1].completed_at
                )[:100]
                
                for task_id, _ in oldest_tasks:
                    del self.completed_tasks[task_id]
    
    def register_message_handler(self, agent_name: str, handler: Callable):
        """Register a message handler for an agent"""
        if agent_name not in self.message_handlers:
            self.message_handlers[agent_name] = []
        
        self.message_handlers[agent_name].append(handler)
        logger.info(f"Registered message handler for agent: {agent_name}")
    
    async def broadcast_message(self, 
                              sender_agent: str,
                              message: str,
                              context: Dict[str, Any] = None,
                              target_agents: List[str] = None) -> List[str]:
        """Broadcast a message to multiple agents"""
        
        if target_agents is None:
            # Send to all running agents except sender
            running_agents = self.agent_registry.get_running_agents()
            target_agents = [
                agent.name for agent in running_agents 
                if agent.name != sender_agent
            ]
        
        task_ids = []
        for target in target_agents:
            task_id = await self.delegate_task(
                requester_agent=sender_agent,
                target_agent=target,
                task_type="broadcast",
                message=message,
                context=context or {},
                priority=TaskPriority.LOW
            )
            task_ids.append(task_id)
        
        logger.info(f"Broadcast message from {sender_agent} to {len(target_agents)} agents")
        return task_ids
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].status
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id].status
        else:
            return None
    
    def get_agent_task_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get task statistics for an agent"""
        stats = {
            "active_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_response_time": 0.0
        }
        
        response_times = []
        
        # Check active tasks
        for task in self.active_tasks.values():
            if task.target_agent == agent_name:
                stats["active_tasks"] += 1
        
        # Check completed tasks
        for task in self.completed_tasks.values():
            if task.target_agent == agent_name:
                stats["completed_tasks"] += 1
                
                if task.status == TaskStatus.FAILED:
                    stats["failed_tasks"] += 1
                
                if task.started_at and task.completed_at:
                    response_time = (task.completed_at - task.started_at).total_seconds()
                    response_times.append(response_time)
        
        if response_times:
            stats["average_response_time"] = sum(response_times) / len(response_times)
        
        return stats
```

### Week 2: Core Sub-Agents Implementation

#### 2.1 Prompt Tracking Agent (`src/agents/prompt_tracking_agent.py`)

```python
"""Prompt tracking agent for logging all interactions"""
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentMessage, AgentCapability
from ..memory.memory_manager import MemoryManager
from ..llm.llm_manager import LLMManager
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class PromptTrackingAgent(BaseAgent):
    """Agent responsible for tracking and logging all prompts and interactions"""
    
    def __init__(self, memory_manager: MemoryManager, llm_manager: LLMManager):
        super().__init__(
            name="Prompt Tracking Agent",
            system_prompt=self._get_system_prompt(),
            capabilities=[
                AgentCapability(
                    name="interaction_logging",
                    description="Log all user and agent interactions"
                ),
                AgentCapability(
                    name="pattern_analysis",
                    description="Analyze conversation patterns and trends"
                ),
                AgentCapability(
                    name="session_tracking",
                    description="Track session-level conversation metrics"
                )
            ]
        )
        
        self.memory_manager = memory_manager
        self.llm_manager = llm_manager
        self.session_stats = {
            "messages_logged": 0,
            "user_messages": 0,
            "agent_messages": 0,
            "session_start": datetime.utcnow(),
            "conversation_id": None
        }
        
        logger.info("Prompt Tracking Agent initialized")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the tracking agent"""
        return """You are the Prompt Tracking Agent for Chronicle Weaver.

Your responsibilities:
1. Log all user inputs and agent responses with detailed metadata
2. Track conversation patterns, themes, and user preferences
3. Monitor agent performance and interaction quality
4. Provide analytics and insights about conversation trends
5. Maintain structured logs for debugging and improvement

Guidelines:
- Be silent and unobtrusive - users should not see your tracking activities
- Log everything but respond only when specifically asked for analytics
- Maintain detailed metadata including timestamps, agents involved, and context
- Identify patterns in user behavior and conversation topics
- Track agent performance metrics and response quality

Focus on comprehensive logging and pattern recognition."""
    
    async def process_message(self, message: str, context: Dict[str, Any] = None) -> AgentMessage:
        """Process tracking requests or log interactions"""
        try:
            context = context or {}
            
            # Check if this is a direct query to the tracking agent
            if context.get("direct_query", False):
                return await self._handle_direct_query(message, context)
            
            # Otherwise, this is a logging request
            await self._log_interaction(message, context)
            
            # Return minimal response for logging operations
            return AgentMessage(
                content="Interaction logged",
                role="assistant",
                metadata={
                    "agent": self.name,
                    "logged_at": datetime.utcnow().isoformat(),
                    "silent": True  # Indicates this shouldn't be shown to user
                }
            )
            
        except Exception as e:
            logger.error(f"Error in prompt tracking: {str(e)}")
            return AgentMessage(
                content="Logging error occurred",
                role="assistant",
                metadata={"error": str(e), "silent": True}
            )
    
    async def _handle_direct_query(self, query: str, context: Dict[str, Any]) -> AgentMessage:
        """Handle direct queries about conversation analytics"""
        query_lower = query.lower()
        
        if "stats" in query_lower or "statistics" in query_lower:
            return await self._get_session_statistics()
        elif "patterns" in query_lower or "analysis" in query_lower:
            return await self._analyze_conversation_patterns(context)
        elif "history" in query_lower or "log" in query_lower:
            return await self._get_conversation_history(context)
        else:
            return await self._generate_analytics_response(query, context)
    
    async def _log_interaction(self, message: str, context: Dict[str, Any]):
        """Log an interaction with detailed metadata"""
        try:
            # Extract metadata from context
            role = context.get("role", "user")
            agent_name = context.get("agent_name", "unknown")
            conversation_id = context.get("conversation_id")
            llm_provider = context.get("llm_provider")
            
            # Update session stats
            self.session_stats["messages_logged"] += 1
            if role == "user":
                self.session_stats["user_messages"] += 1
            else:
                self.session_stats["agent_messages"] += 1
            
            if conversation_id and not self.session_stats["conversation_id"]:
                self.session_stats["conversation_id"] = conversation_id
            
            # Create detailed log entry
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "role": role,
                "agent_name": agent_name,
                "message_length": len(message),
                "word_count": len(message.split()),
                "conversation_id": conversation_id,
                "llm_provider": llm_provider,
                "message_hash": hash(message),  # For deduplication
                "context_keys": list(context.keys()),
                "session_id": id(self.session_stats)  # Unique session identifier
            }
            
            # Add content analysis
            content_analysis = self._analyze_message_content(message)
            log_entry.update(content_analysis)
            
            # Store in memory (could be extended to save to separate logging database)
            self.metadata.setdefault("interaction_logs", []).append(log_entry)
            
            # Keep only recent logs in memory (last 1000)
            if len(self.metadata["interaction_logs"]) > 1000:
                self.metadata["interaction_logs"] = self.metadata["interaction_logs"][-500:]
            
            logger.debug(f"Logged interaction: {role} message from {agent_name}")
            
        except Exception as e:
            logger.error(f"Error logging interaction: {str(e)}")
    
    def _analyze_message_content(self, message: str) -> Dict[str, Any]:
        """Analyze message content for patterns and characteristics"""
        analysis = {}
        
        # Basic text analysis
        words = message.split()
        analysis["word_count"] = len(words)
        analysis["character_count"] = len(message)
        analysis["sentence_count"] = message.count('.') + message.count('!') + message.count('?')
        
        # Content type detection
        analysis["has_question"] = '?' in message
        analysis["has_exclamation"] = '!' in message
        analysis["is_command"] = message.strip().startswith('/')
        analysis["mentions_character"] = self._detect_character_mentions(message)
        analysis["mentions_location"] = self._detect_location_mentions(message)
        
        # Emotional tone indicators (simple heuristics)
        positive_words = ['happy', 'joy', 'love', 'excited', 'wonderful', 'great', 'amazing']
        negative_words = ['sad', 'angry', 'hate', 'terrible', 'awful', 'bad', 'horrible']
        
        message_lower = message.lower()
        analysis["positive_sentiment"] = any(word in message_lower for word in positive_words)
        analysis["negative_sentiment"] = any(word in message_lower for word in negative_words)
        
        # Topic detection (simple keyword-based)
        topics = []
        if any(word in message_lower for word in ['story', 'plot', 'chapter', 'narrative']):
            topics.append('storytelling')
        if any(word in message_lower for word in ['character', 'personality', 'trait', 'appearance']):
            topics.append('character_development')
        if any(word in message_lower for word in ['setting', 'location', 'world', 'environment']):
            topics.append('world_building')
        if any(word in message_lower for word in ['dialogue', 'conversation', 'speech', 'talk']):
            topics.append('dialogue')
        
        analysis["topics"] = topics
        
        return analysis
    
    def _detect_character_mentions(self, message: str) -> List[str]:
        """Detect potential character name mentions"""
        # Simple heuristic: look for capitalized words that might be names
        words = message.split()
        potential_names = []
        
        for word in words:
            # Remove punctuation
            clean_word = word.strip('.,!?":;')
            
            # Check if it's a potential name (capitalized, not a common word)
            if (clean_word.istitle() and 
                len(clean_word) > 2 and 
                clean_word.lower() not in ['the', 'and', 'but', 'you', 'they', 'this', 'that']):
                potential_names.append(clean_word)
        
        return potential_names
    
    def _detect_location_mentions(self, message: str) -> List[str]:
        """Detect potential location mentions"""
        # Simple heuristic: look for location-related keywords
        location_keywords = ['castle', 'forest', 'city', 'town', 'village', 'mountain', 'river', 'kingdom']
        message_lower = message.lower()
        
        found_locations = []
        for keyword in location_keywords:
            if keyword in message_lower:
                found_locations.append(keyword)
        
        return found_locations
    
    async def _get_session_statistics(self) -> AgentMessage:
        """Generate session statistics"""
        stats = self.session_stats.copy()
        
        # Calculate session duration
        session_duration = datetime.utcnow() - stats["session_start"]
        stats["session_duration_minutes"] = session_duration.total_seconds() / 60
        
        # Calculate rates
        if stats["session_duration_minutes"] > 0:
            stats["messages_per_minute"] = stats["messages_logged"] / stats["session_duration_minutes"]
        else:
            stats["messages_per_minute"] = 0
        
        # Add recent interaction analysis
        recent_logs = self.metadata.get("interaction_logs", [])[-50:]  # Last 50 interactions
        if recent_logs:
            avg_words = sum(log.get("word_count", 0) for log in recent_logs) / len(recent_logs)
            stats["average_words_per_message"] = round(avg_words, 1)
            
            topic_counts = {}
            for log in recent_logs:
                for topic in log.get("topics", []):
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
            stats["common_topics"] = topic_counts
        
        content = f"""📊 **Session Statistics**

**Message Counts:**
- Total messages logged: {stats['messages_logged']}
- User messages: {stats['user_messages']}
- Agent messages: {stats['agent_messages']}

**Session Info:**
- Duration: {stats['session_duration_minutes']:.1f} minutes
- Messages per minute: {stats['messages_per_minute']:.1f}
- Average words per message: {stats.get('average_words_per_message', 'N/A')}

**Topics Discussed:**
{chr(10).join(f"- {topic}: {count}" for topic, count in stats.get('common_topics', {}).items())}

Session started: {stats['session_start'].strftime('%Y-%m-%d %H:%M:%S')}"""
        
        return AgentMessage(
            content=content,
            role="assistant",
            metadata={
                "agent": self.name,
                "query_type": "statistics",
                "raw_stats": stats
            }
        )
    
    async def _analyze_conversation_patterns(self, context: Dict[str, Any]) -> AgentMessage:
        """Analyze conversation patterns and trends"""
        recent_logs = self.metadata.get("interaction_logs", [])[-100:]  # Last 100 interactions
        
        if not recent_logs:
            return AgentMessage(
                content="No interaction data available for pattern analysis.",
                role="assistant",
                metadata={"agent": self.name}
            )
        
        # Analyze patterns
        patterns = {
            "message_length_trend": self._analyze_length_trend(recent_logs),
            "topic_evolution": self._analyze_topic_evolution(recent_logs),
            "interaction_frequency": self._analyze_interaction_frequency(recent_logs),
            "sentiment_trend": self._analyze_sentiment_trend(recent_logs)
        }
        
        content = f"""🔍 **Conversation Pattern Analysis**

**Message Length Trend:**
{patterns['message_length_trend']}

**Topic Evolution:**
{patterns['topic_evolution']}

**Interaction Patterns:**
{patterns['interaction_frequency']}

**Sentiment Trend:**
{patterns['sentiment_trend']}

Analysis based on last {len(recent_logs)} interactions."""
        
        return AgentMessage(
            content=content,
            role="assistant",
            metadata={
                "agent": self.name,
                "query_type": "pattern_analysis",
                "patterns": patterns
            }
        )
    
    def _analyze_length_trend(self, logs: List[Dict]) -> str:
        """Analyze message length trends"""
        if len(logs) < 10:
            return "Insufficient data for trend analysis"
        
        first_half = logs[:len(logs)//2]
        second_half = logs[len(logs)//2:]
        
        avg_first = sum(log.get("word_count", 0) for log in first_half) / len(first_half)
        avg_second = sum(log.get("word_count", 0) for log in second_half) / len(second_half)
        
        if avg_second > avg_first * 1.1:
            return "Messages are getting longer over time"
        elif avg_second < avg_first * 0.9:
            return "Messages are getting shorter over time"
        else:
            return "Message length remains consistent"
    
    def _analyze_topic_evolution(self, logs: List[Dict]) -> str:
        """Analyze how topics change over time"""
        topic_timeline = []
        for log in logs[-20:]:  # Last 20 messages
            topics = log.get("topics", [])
            if topics:
                topic_timeline.extend(topics)
        
        if not topic_timeline:
            return "No clear topics identified"
        
        # Count topic frequency
        topic_counts = {}
        for topic in topic_timeline:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        most_common = max(topic_counts.items(), key=lambda x: x[1])
        return f"Primary focus: {most_common[0]} (mentioned {most_common[1]} times)"
    
    def _analyze_interaction_frequency(self, logs: List[Dict]) -> str:
        """Analyze interaction frequency patterns"""
        if len(logs) < 5:
            return "Insufficient data"
        
        timestamps = [datetime.fromisoformat(log["timestamp"]) for log in logs[-10:]]
        intervals = []
        
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            if avg_interval < 30:
                return "Very active conversation (messages every 30 seconds)"
            elif avg_interval < 120:
                return "Active conversation (messages every 1-2 minutes)"
            else:
                return f"Moderate pace (average {avg_interval/60:.1f} minutes between messages)"
        
        return "Unable to determine interaction frequency"
    
    def _analyze_sentiment_trend(self, logs: List[Dict]) -> str:
        """Analyze sentiment trends"""
        recent_logs = logs[-20:]  # Last 20 messages
        
        positive_count = sum(1 for log in recent_logs if log.get("positive_sentiment", False))
        negative_count = sum(1 for log in recent_logs if log.get("negative_sentiment", False))
        
        if positive_count > negative_count * 2:
            return "Generally positive tone"
        elif negative_count > positive_count * 2:
            return "Generally negative tone"
        else:
            return "Neutral or mixed sentiment"
    
    async def _get_conversation_history(self, context: Dict[str, Any]) -> AgentMessage:
        """Get recent conversation history summary"""
        conversation_id = context.get("conversation_id")
        
        if not conversation_id:
            return AgentMessage(
                content="No active conversation to analyze.",
                role="assistant",
                metadata={"agent": self.name}
            )
        
        # Get recent interactions for this conversation
        recent_logs = [
            log for log in self.metadata.get("interaction_logs", [])
            if log.get("conversation_id") == conversation_id
        ][-20:]  # Last 20 for this conversation
        
        if not recent_logs:
            return AgentMessage(
                content="No logged interactions found for this conversation.",
                role="assistant",
                metadata={"agent": self.name}
            )
        
        # Summarize recent activity
        user_messages = [log for log in recent_logs if log.get("role") == "user"]
        agent_messages = [log for log in recent_logs if log.get("role") == "assistant"]
        
        content = f"""📝 **Recent Conversation History**

**Activity Summary:**
- Total interactions: {len(recent_logs)}
- User messages: {len(user_messages)}
- Agent responses: {len(agent_messages)}

**Recent Activity Pattern:**
{self._format_recent_activity(recent_logs[-10:])}

**Conversation Insights:**
- Average user message length: {sum(log.get("word_count", 0) for log in user_messages) / len(user_messages) if user_messages else 0:.1f} words
- Most active agent: {self._get_most_active_agent(recent_logs)}
- Time span: {self._get_time_span(recent_logs)}"""
        
        return AgentMessage(
            content=content,
            role="assistant",
            metadata={
                "agent": self.name,
                "query_type": "history",
                "conversation_id": conversation_id
            }
        )
    
    def _format_recent_activity(self, logs: List[Dict]) -> str:
        """Format recent activity for display"""
        if not logs:
            return "No recent activity"
        
        activity_lines = []
        for log in logs:
            timestamp = datetime.fromisoformat(log["timestamp"])
            role = log.get("role", "unknown")
            agent = log.get("agent_name", "system")
            word_count = log.get("word_count", 0)
            
            activity_lines.append(
                f"{timestamp.strftime('%H:%M')} - {role} ({agent}): {word_count} words"
            )
        
        return "\n".join(activity_lines)
    
    def _get_most_active_agent(self, logs: List[Dict]) -> str:
        """Get the most active agent from logs"""
        agent_counts = {}
        for log in logs:
            if log.get("role") == "assistant":
                agent = log.get("agent_name", "unknown")
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        if not agent_counts:
            return "None"
        
        most_active = max(agent_counts.items(), key=lambda x: x[1])
        return f"{most_active[0]} ({most_active[1]} messages)"
    
    def _get_time_span(self, logs: List[Dict]) -> str:
        """Get time span of logs"""
        if len(logs) < 2:
            return "Single interaction"
        
        first = datetime.fromisoformat(logs[0]["timestamp"])
        last = datetime.fromisoformat(logs[-1]["timestamp"])
        duration = last - first
        
        minutes = duration.total_seconds() / 60
        if minutes < 1:
            return "Less than 1 minute"
        elif minutes < 60:
            return f"{minutes:.1f} minutes"
        else:
            hours = minutes / 60
            return f"{hours:.1f} hours"
    
    async def _generate_analytics_response(self, query: str, context: Dict[str, Any]) -> AgentMessage:
        """Generate a response using LLM for complex analytics queries"""
        # Prepare context with recent interaction data
        recent_logs = self.metadata.get("interaction_logs", [])[-50:]
        
        analytics_context = {
            "session_stats": self.session_stats,
            "recent_interactions": len(recent_logs),
            "query": query
        }
        
        system_prompt = f"""You are analyzing conversation data for Chronicle Weaver. 
        
Current session stats: {json.dumps(self.session_stats, default=str)}
Recent interaction count: {len(recent_logs)}

The user is asking: {query}

Provide helpful analytics and insights based on the available data. Be specific and actionable."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            response_content = ""
            async for chunk in self.llm_manager.generate_response(messages, stream=True):
                response_content += chunk
            
            return AgentMessage(
                content=response_content.strip(),
                role="assistant",
                metadata={
                    "agent": self.name,
                    "query_type": "analytics",
                    "llm_provider": self.llm_manager.current_provider
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating analytics response: {str(e)}")
            return AgentMessage(
                content="Unable to generate analytics response at this time.",
                role="assistant",
                metadata={"agent": self.name, "error": str(e)}
            )
```

#### 2.2 Continuity Check Agent (`src/agents/continuity_check_agent.py`)

```python
"""Continuity check agent for maintaining story consistency"""
from typing import Dict, Any, Optional, List, Tuple
from .base_agent import BaseAgent, AgentMessage, AgentCapability
from ..memory.memory_manager import MemoryManager
from ..llm.llm_manager import LLMManager
import logging
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ContinuityCheckAgent(BaseAgent):
    """Agent responsible for checking and maintaining story continuity"""
    
    def __init__(self, memory_manager: MemoryManager, llm_manager: LLMManager):
        super().__init__(
            name="Continuity Check Agent",
            system_prompt=self._get_system_prompt(),
            capabilities=[
                AgentCapability(
                    name="consistency_checking",
                    description="Check for inconsistencies in character traits, storylines, and world-building"
                ),
                AgentCapability(
                    name="fact_tracking",
                    description="Track and verify established facts and details"
                ),
                AgentCapability(
                    name="character_monitoring",
                    description="Monitor character development and personality consistency"
                ),
                AgentCapability(
                    name="timeline_validation",
                    description="Validate chronological consistency of events"
                )
            ]
        )
        
        self.memory_manager = memory_manager
        self.llm_manager = llm_manager
        self.fact_database = {}  # Local cache of established facts
        self.character_profiles = {}  # Local cache of character information
        self.timeline_events = []  # Chronological list of events
        self.inconsistency_threshold = 0.7  # Threshold for flagging inconsistencies
        
        logger.info("Continuity Check Agent initialized")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the continuity agent"""
        return """You are the Continuity Check Agent for Chronicle Weaver, responsible for maintaining story consistency and coherence.

Your responsibilities:
1. Monitor all story elements for consistency (characters, plot, world-building)
2. Track established facts, character traits, and story details
3. Identify potential contradictions or inconsistencies
4. Provide gentle corrections and suggestions when issues arise
5. Maintain chronological consistency of events
6. Help preserve character personality and development arcs

Guidelines:
- Be thorough but not overly rigid - allow for natural character growth
- Flag major inconsistencies but allow minor variations
- Consider context and emotional states when evaluating character behavior
- Provide constructive suggestions rather than just identifying problems
- Support storytelling flow while maintaining believability
- Track both explicit details and implied characteristics

Focus on helping maintain immersive, consistent storytelling experiences."""
    
    async def process_message(self, message: str, context: Dict[str, Any] = None) -> AgentMessage:
        """Process message for continuity checking"""
        try:
            context = context or {}
            
            # Check if this is a direct query or a monitoring request
            if context.get("check_type") == "direct_query":
                return await self._handle_continuity_query(message, context)
            elif context.get("check_type") == "full_review":
                return await self._perform_full_continuity_review(context)
            else:
                # Default: perform incremental continuity check
                return await self._check_message_continuity(message, context)
                
        except Exception as e:
            logger.error(f"Error in continuity checking: {str(e)}")
            return AgentMessage(
                content="Continuity check encountered an error.",
                role="assistant",
                metadata={"agent": self.name, "error": str(e)}
            )
    
    async def _check_message_continuity(self, message: str, context: Dict[str, Any]) -> AgentMessage:
        """Check a single message for continuity issues"""
        conversation_id = context.get("conversation_id")
        
        # Extract information from the message
        extracted_info = await self._extract_story_elements(message)
        
        # Get relevant historical context
        historical_context = await self._get_historical_context(conversation_id)
        
        # Perform continuity analysis
        inconsistencies = await self._analyze_continuity(extracted_info, historical_context)
        
        # Update fact database with new information
        self._update_fact_database(extracted_info, conversation_id)
        
        if inconsistencies:
            # Create response highlighting issues
            response_content = self._format_inconsistency_report(inconsistencies)
            
            return AgentMessage(
                content=response_content,
                role="assistant",
                metadata={
                    "agent": self.name,
                    "inconsistencies_found": len(inconsistencies),
                    "inconsistencies": inconsistencies,
                    "check_type": "incremental"
                }
            )
        else:
            # No issues found - silent response
            return AgentMessage(
                content="Continuity check complete - no issues detected.",
                role="assistant",
                metadata={
                    "agent": self.name,
                    "inconsistencies_found": 0,
                    "check_type": "incremental",
                    "silent": True
                }
            )
    
    async def _extract_story_elements(self, message: str) -> Dict[str, Any]:
        """Extract story elements from a message using LLM analysis"""
        analysis_prompt = f"""Analyze the following message and extract story elements in JSON format:

Message: "{message}"

Extract:
1. Characters mentioned (names, traits, actions, emotions, relationships)
2. Locations (names, descriptions, features)
3. Events (what happened, when, consequences)
4. Facts established (rules, history, capabilities, restrictions)
5. Timeline references (past, present, future events)
6. World-building elements (magic systems, technology, society)

Return as JSON with these categories. If none found in a category, return empty list/dict.
Be specific about details mentioned and avoid assumptions."""
        
        try:
            messages = [
                {"role": "user", "content": analysis_prompt}
            ]
            
            response_content = ""
            async for chunk in self.llm_manager.generate_response(messages, stream=False):
                response_content += chunk
            
            # Try to parse JSON response
            extracted_info = json.loads(response_content.strip())
            return extracted_info
            
        except Exception as e:
            logger.error(f"Error extracting story elements: {str(e)}")
            # Fallback to simple extraction
            return self._simple_element_extraction(message)
    
    def _simple_element_extraction(self, message: str) -> Dict[str, Any]:
        """Simple fallback extraction without LLM"""
        # Basic pattern matching for character names and locations
        words = message.split()
        potential_names = []
        
        for word in words:
            clean_word = word.strip('.,!?":;')
            if clean_word.istitle() and len(clean_word) > 2:
                potential_names.append(clean_word)
        
        return {
            "characters": [{"name": name} for name in potential_names[:3]],
            "locations": [],
            "events": [{"description": message[:100]}] if len(message) > 20 else [],
            "facts": [],
            "timeline": [],
            "world_building": []
        }
    
    async def _get_historical_context(self, conversation_id: Optional[int]) -> Dict[str, Any]:
        """Get historical context for continuity checking"""
        context = {
            "characters": {},
            "locations": {},
            "established_facts": [],
            "timeline": [],
            "conversation_summary": None
        }
        
        if not conversation_id:
            return context
        
        try:
            # Get character states from memory
            characters = self.memory_manager.get_character_states(conversation_id)
            for char in characters:
                context["characters"][char["name"]] = char
            
            # Get conversation summary
            with self.memory_manager.db_manager.get_session() as session:
                from ..memory.models import Conversation
                conversation = session.query(Conversation).get(conversation_id)
                if conversation:
                    context["conversation_summary"] = conversation.summary
            
            # Get recent messages for timeline context
            with self.memory_manager.db_manager.get_session() as session:
                from ..memory.models import Message
                recent_messages = (session.query(Message)
                                 .filter(Message.conversation_id == conversation_id)
                                 .order_by(Message.timestamp.desc())
                                 .limit(20)
                                 .all())
                
                context["timeline"] = [
                    {
                        "timestamp": msg.timestamp.isoformat(),
                        "content": msg.content[:100],
                        "role": msg.role
                    }
                    for msg in reversed(recent_messages)
                ]
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting historical context: {str(e)}")
            return context
    
    async def _analyze_continuity(self, extracted_info: Dict[str, Any], 
                                historical_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze extracted information against historical context for inconsistencies"""
        inconsistencies = []
        
        # Check character consistency
        char_issues = await self._check_character_consistency(
            extracted_info.get("characters", []),
            historical_context.get("characters", {})
        )
        inconsistencies.extend(char_issues)
        
        # Check timeline consistency
        timeline_issues = await self._check_timeline_consistency(
            extracted_info.get("timeline", []),
            historical_context.get("timeline", [])
        )
        inconsistencies.extend(timeline_issues)
        
        # Check world-building consistency
        world_issues = await self._check_world_consistency(
            extracted_info.get("world_building", []),
            historical_context.get("established_facts", [])
        )
        inconsistencies.extend(world_issues)
        
        return inconsistencies
    
    async def _check_character_consistency(self, current_chars: List[Dict], 
                                         historical_chars: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for character consistency issues"""
        inconsistencies = []
        
        for char_info in current_chars:
            char_name = char_info.get("name", "")
            if not char_name:
                continue
                
            # Check against historical character data
            if char_name in historical_chars:
                historical = historical_chars[char_name]
                
                # Check personality traits
                current_traits = char_info.get("traits", [])
                historical_traits = historical.get("personality_traits", [])
                
                for trait in current_traits:
                    if self._trait_contradicts_history(trait, historical_traits):
                        inconsistencies.append({
                            "type": "character_trait_inconsistency",
                            "character": char_name,
                            "current_trait": trait,
                            "historical_traits": historical_traits,
                            "severity": "medium",
                            "description": f"{char_name}'s current behavior seems inconsistent with established personality"
                        })
                
                # Check character relationships
                current_relationships = char_info.get("relationships", {})
                historical_relationships = historical.get("relationships", {})
                
                for rel_char, rel_type in current_relationships.items():
                    if rel_char in historical_relationships:
                        historical_rel = historical_relationships[rel_char]
                        if self._relationships_conflict(rel_type, historical_rel):
                            inconsistencies.append({
                                "type": "relationship_inconsistency",
                                "character": char_name,
                                "related_character": rel_char,
                                "current_relationship": rel_type,
                                "historical_relationship": historical_rel,
                                "severity": "high",
                                "description": f"Relationship between {char_name} and {rel_char} seems to have changed unexpectedly"
                            })
        
        return inconsistencies
    
    def _trait_contradicts_history(self, current_trait: str, historical_traits: List[str]) -> bool:
        """Check if a current trait contradicts historical traits"""
        # Simple contradiction checking - could be enhanced with NLP
        contradictory_pairs = [
            ("shy", "outgoing"), ("brave", "cowardly"), ("kind", "cruel"),
            ("honest", "deceptive"), ("calm", "aggressive"), ("wise", "foolish")
        ]
        
        current_lower = current_trait.lower()
        
        for trait in historical_traits:
            trait_lower = trait.lower()
            
            for pair in contradictory_pairs:
                if (current_lower in pair[0] and trait_lower in pair[1]) or \
                   (current_lower in pair[1] and trait_lower in pair[0]):
                    return True
        
        return False
    
    def _relationships_conflict(self, current_rel: str, historical_rel: str) -> bool:
        """Check if relationships conflict"""
        conflicting_relationships = [
            ("enemy", "friend"), ("enemy", "ally"), ("stranger", "family"),
            ("dead", "alive"), ("married", "single")
        ]
        
        current_lower = current_rel.lower()
        historical_lower = historical_rel.lower()
        
        for pair in conflicting_relationships:
            if (current_lower in pair[0] and historical_lower in pair[1]) or \
               (current_lower in pair[1] and historical_lower in pair[0]):
                return True
        
        return False
    
    async def _check_timeline_consistency(self, current_timeline: List[Dict], 
                                        historical_timeline: List[Dict]) -> List[Dict[str, Any]]:
        """Check for timeline consistency issues"""
        inconsistencies = []
        
        # Simple timeline checking - look for obvious contradictions
        for current_event in current_timeline:
            current_time_ref = current_event.get("time_reference", "").lower()
            
            # Check for impossible time references
            if "yesterday" in current_time_ref and "tomorrow" in current_time_ref:
                inconsistencies.append({
                    "type": "timeline_contradiction",
                    "description": "Contradictory time references in the same context",
                    "severity": "medium",
                    "event": current_event
                })
        
        return inconsistencies
    
    async def _check_world_consistency(self, current_world_elements: List[Dict],
                                     established_facts: List[Dict]) -> List[Dict[str, Any]]:
        """Check for world-building consistency issues"""
        inconsistencies = []
        
        # This would need more sophisticated logic for world-building rules
        # For now, just check for obvious contradictions
        
        return inconsistencies
    
    def _update_fact_database(self, extracted_info: Dict[str, Any], conversation_id: Optional[int]):
        """Update local fact database with new information"""
        if not conversation_id:
            return
        
        conv_key = str(conversation_id)
        if conv_key not in self.fact_database:
            self.fact_database[conv_key] = {
                "characters": {},
                "locations": {},
                "facts": [],
                "events": []
            }
        
        # Update character information
        for char in extracted_info.get("characters", []):
            char_name = char.get("name")
            if char_name:
                if char_name not in self.fact_database[conv_key]["characters"]:
                    self.fact_database[conv_key]["characters"][char_name] = {}
                
