"""Short-term conversation memory using LangChain"""
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing import List, Dict, Any, Optional
from .database_manager import DatabaseManager
from .models import Conversation, Message
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConversationMemory:
    """Manages short-term conversation memory with database persistence"""
    
    def __init__(self, 
                 database_manager: DatabaseManager,
                 conversation_id: Optional[int] = None,
                 window_size: int = 20):
        self.db_manager = database_manager
        self.conversation_id = conversation_id
        self.window_size = window_size
        
        # Initialize LangChain memory
        self.memory = ConversationBufferWindowMemory(
            k=window_size,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Load existing conversation if provided
        if conversation_id:
            self._load_conversation_history()
    
    def _load_conversation_history(self):
        """Load conversation history from database"""
        try:
            with self.db_manager.get_session() as session:
                messages = (session.query(Message)
                           .filter(Message.conversation_id == self.conversation_id)
                           .order_by(Message.timestamp.desc())
                           .limit(self.window_size)
                           .all())
                
                # Reverse to get chronological order
                messages.reverse()
                
                for msg in messages:
                    if msg.role == "user":
                        self.memory.chat_memory.add_message(HumanMessage(content=msg.content))
                    elif msg.role == "assistant":
                        self.memory.chat_memory.add_message(AIMessage(content=msg.content))
                    elif msg.role == "system":
                        self.memory.chat_memory.add_message(SystemMessage(content=msg.content))
                
                logger.info(f"Loaded {len(messages)} messages from conversation {self.conversation_id}")
                
        except Exception as e:
            logger.error(f"Error loading conversation history: {str(e)}")
    
    def add_user_message(self, content: str, metadata: Dict[str, Any] = None) -> Message:
        """Add user message to memory and database"""
        # Add to LangChain memory
        self.memory.chat_memory.add_message(HumanMessage(content=content))
        
        # Save to database
        return self._save_message("user", content, metadata or {})
    
    def add_assistant_message(self, content: str, agent_name: str = None, 
                            llm_provider: str = None, metadata: Dict[str, Any] = None) -> Message:
        """Add assistant message to memory and database"""
        # Add to LangChain memory
        self.memory.chat_memory.add_message(AIMessage(content=content))
        
        # Prepare metadata
        msg_metadata = metadata or {}
        if agent_name:
            msg_metadata["agent_name"] = agent_name
        if llm_provider:
            msg_metadata["llm_provider"] = llm_provider
        
        # Save to database
        return self._save_message("assistant", content, msg_metadata, agent_name, llm_provider)
    
    def add_system_message(self, content: str, metadata: Dict[str, Any] = None) -> Message:
        """Add system message to memory and database"""
        # Add to LangChain memory
        self.memory.chat_memory.add_message(SystemMessage(content=content))
        
        # Save to database
        return self._save_message("system", content, metadata or {})
    
    def _save_message(self, role: str, content: str, metadata: Dict[str, Any],
                     agent_name: str = None, llm_provider: str = None) -> Message:
        """Save message to database"""
        try:
            with self.db_manager.get_session() as session:
                message = Message(
                    conversation_id=self.conversation_id,
                    role=role,
                    content=content,
                    agent_name=agent_name,
                    llm_provider=llm_provider,
                    metadata=metadata,
                    token_count=len(content.split())  # Rough estimate
                )
                
                session.add(message)
                session.flush()  # Get the ID
                
                logger.debug(f"Saved {role} message (ID: {message.id})")
                return message
                
        except Exception as e:
            logger.error(f"Error saving message: {str(e)}")
            raise
    
    def get_messages(self, limit: int = None) -> List[BaseMessage]:
        """Get recent messages from memory"""
        messages = self.memory.chat_memory.messages
        if limit:
            return messages[-limit:]
        return messages
    
    def get_context_messages(self) -> List[Dict[str, str]]:
        """Get messages formatted for LLM context"""
        messages = self.get_messages()
        return [
            {
                "role": self._get_openai_role(msg),
                "content": msg.content
            }
            for msg in messages
        ]
    
    def _get_openai_role(self, message: BaseMessage) -> str:
        """Convert LangChain message type to OpenAI role"""
        if isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, AIMessage):
            return "assistant"
        elif isinstance(message, SystemMessage):
            return "system"
        else:
            return "user"  # Default fallback
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def create_new_conversation(self, title: str = None) -> int:
        """Create a new conversation and return its ID"""
        try:
            with self.db_manager.get_session() as session:
                conversation = Conversation(
                    title=title or f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                    metadata={}
                )
                
                session.add(conversation)
                session.flush()
                
                self.conversation_id = conversation.id
                logger.info(f"Created new conversation (ID: {conversation.id})")
                return conversation.id
                
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            raise
    
    def get_conversation_summary(self) -> Optional[str]:
        """Get conversation summary from database"""
        if not self.conversation_id:
            return None
        
        try:
            with self.db_manager.get_session() as session:
                conversation = session.query(Conversation).get(self.conversation_id)
                return conversation.summary if conversation else None
                
        except Exception as e:
            logger.error(f"Error getting conversation summary: {str(e)}")
            return None
    
    def update_conversation_summary(self, summary: str):
        """Update conversation summary in database"""
        if not self.conversation_id:
            return
        
        try:
            with self.db_manager.get_session() as session:
                conversation = session.query(Conversation).get(self.conversation_id)
                if conversation:
                    conversation.summary = summary
                    logger.info(f"Updated conversation summary (ID: {self.conversation_id})")
                
        except Exception as e:
            logger.error(f"Error updating conversation summary: {str(e)}")
