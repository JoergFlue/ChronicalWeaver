"""Main Agent - Primary user interaction handler"""
from typing import Dict, Any, Optional
from .base_agent import BaseAgent, AgentMessage, AgentCapability
from llm.llm_manager import LLMManager
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
