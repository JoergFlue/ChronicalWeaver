"""Unified memory management interface"""
from typing import Dict, Any, List, Optional
from .database_manager import DatabaseManager
from .conversation_memory import ConversationMemory
from .persistent_memory import PersistentMemory
from .summarization import ConversationSummarizer
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MemoryManager:
    """Unified interface for all memory operations"""
    
    def __init__(self, database_url: str = None):
        self.db_manager = DatabaseManager(database_url)
        self.persistent_memory = PersistentMemory(self.db_manager)
        self.summarizer = ConversationSummarizer()
        self.active_conversations = {}  # conversation_id -> ConversationMemory
    
    def get_conversation_memory(self, conversation_id: int = None) -> ConversationMemory:
        """Get or create conversation memory for a session"""
        if conversation_id is None:
            # Create new conversation
            memory = ConversationMemory(self.db_manager)
            conversation_id = memory.create_new_conversation()
            self.active_conversations[conversation_id] = memory
            return memory
        
        if conversation_id not in self.active_conversations:
            # Load existing conversation
            memory = ConversationMemory(self.db_manager, conversation_id)
            self.active_conversations[conversation_id] = memory
        
        return self.active_conversations[conversation_id]
    
    def start_new_conversation(self, title: str = None) -> tuple[int, ConversationMemory]:
        """Start a new conversation and return ID and memory object"""
        memory = ConversationMemory(self.db_manager)
        conversation_id = memory.create_new_conversation(title)
        self.active_conversations[conversation_id] = memory
        
        logger.info(f"Started new conversation: {conversation_id}")
        return conversation_id, memory
    
    def summarize_conversation(self, conversation_id: int, force: bool = False) -> Optional[str]:
        """Create or update conversation summary"""
        try:
            # Check if summary already exists and is recent
            if not force:
                existing_summary = self.persistent_memory.get_conversation_summary(conversation_id)
                if existing_summary:
                    return existing_summary
            
            # Get conversation messages
            with self.db_manager.get_session() as session:
                messages = (session.query(Message)
                           .filter(Message.conversation_id == conversation_id)
                           .order_by(Message.timestamp)
                           .all())
                
                if len(messages) < 5:  # Don't summarize very short conversations
                    return None
                
                # Create summary
                message_texts = [f"{msg.role}: {msg.content}" for msg in messages]
                summary = self.summarizer.create_summary(message_texts)
                
                # Save summary
                if conversation_id in self.active_conversations:
                    self.active_conversations[conversation_id].update_conversation_summary(summary)
                
                logger.info(f"Created summary for conversation {conversation_id}")
                return summary
                
        except Exception as e:
            logger.error(f"Error summarizing conversation: {str(e)}")
            return None
    
    def get_relevant_context(self, query: str, conversation_id: int = None, limit: int = 5) -> Dict[str, Any]:
        """Get relevant context from memory for a query"""
        context = {
            "characters": [],
            "props": [],
            "previous_conversations": [],
            "current_conversation_summary": None
        }
        
        try:
            # Get relevant characters
            characters = self.persistent_memory.find_character_by_name(query)
            if characters:
                context["characters"].append(characters)
            
            # Get relevant props
            props = self.persistent_memory.get_prop_items(search_term=query, limit=3)
            context["props"] = props
            
            # Get relevant previous conversations
            prev_convs = self.persistent_memory.search_conversations(query, limit=limit)
            context["previous_conversations"] = prev_convs
            
            # Get current conversation context
            if conversation_id:
                conv_summary = self.persistent_memory.get_conversation_summary(conversation_id)
                context["current_conversation_summary"] = conv_summary
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return context
    
    # Delegate methods to persistent memory
    def save_character_state(self, character_data: Dict[str, Any]) -> int:
        """Save character state to persistent memory"""
        return self.persistent_memory.save_character_state(character_data)
    
    def get_character_states(self, conversation_id: int = None) -> List[Dict[str, Any]]:
        """Get character states from persistent memory"""
        return self.persistent_memory.get_character_states(conversation_id)
    
    def save_prop_item(self, prop_data: Dict[str, Any]) -> int:
        """Save prop item to persistent memory"""
        return self.persistent_memory.save_prop_item(prop_data)
    
    def get_prop_items(self, **kwargs) -> List[Dict[str, Any]]:
        """Get prop items from persistent memory"""
        return self.persistent_memory.get_prop_items(**kwargs)
    
    def save_agent_configuration(self, agent_data: Dict[str, Any]) -> int:
        """Save agent configuration to persistent memory"""
        return self.persistent_memory.save_agent_configuration(agent_data)
    
    def get_agent_configurations(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """Get agent configurations from persistent memory"""
        return self.persistent_memory.get_agent_configurations(enabled_only)
    
    def get_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversations from persistent memory"""
        return self.persistent_memory.get_conversations(limit)
    
    def archive_conversation(self, conversation_id: int) -> bool:
        """Archive conversation in persistent memory"""
        # Remove from active conversations
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
        
        return self.persistent_memory.archive_conversation(conversation_id)
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to manage storage"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            with self.db_manager.get_session() as session:
                # Archive old inactive conversations
                old_conversations = (session.query(Conversation)
                                   .filter(Conversation.is_active == False)
                                   .filter(Conversation.updated_at < cutoff_date)
                                   .all())
                
                for conv in old_conversations:
                    # Create final summary before deletion
                    self.summarize_conversation(conv.id, force=True)
                    
                    # Delete old messages but keep conversation record with summary
                    session.query(Message).filter(Message.conversation_id == conv.id).delete()
                
                logger.info(f"Cleaned up {len(old_conversations)} old conversations")
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
    
    def close(self):
        """Close memory manager and database connections"""
        self.active_conversations.clear()
        self.db_manager.close()
        logger.info("Memory manager closed")
