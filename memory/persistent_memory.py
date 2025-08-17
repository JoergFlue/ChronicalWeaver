"""Long-term memory and data persistence manager"""
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from .database_manager import DatabaseManager
from .models import (
    Conversation, Message, ConversationSummary, 
    CharacterState, PropItem, AgentConfiguration, UserPreference
)
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PersistentMemory:
    """Manages long-term memory and data persistence"""
    
    def __init__(self, database_manager: DatabaseManager):
        self.db_manager = database_manager
    
    # Conversation Management
    def get_conversations(self, limit: int = 50, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """Get list of conversations"""
        try:
            with self.db_manager.get_session() as session:
                query = session.query(Conversation)
                
                if not include_inactive:
                    query = query.filter(Conversation.is_active == True)
                
                conversations = (query.order_by(desc(Conversation.updated_at))
                               .limit(limit)
                               .all())
                
                return [
                    {
                        "id": conv.id,
                        "title": conv.title,
                        "created_at": conv.created_at,
                        "updated_at": conv.updated_at,
                        "is_active": conv.is_active,
                        "summary": conv.summary,
                        "message_count": len(conv.messages)
                    }
                    for conv in conversations
                ]
                
        except Exception as e:
            logger.error(f"Error getting conversations: {str(e)}")
            return []
    
    def archive_conversation(self, conversation_id: int) -> bool:
        """Archive a conversation (mark as inactive)"""
        try:
            with self.db_manager.get_session() as session:
                conversation = session.query(Conversation).get(conversation_id)
                if conversation:
                    conversation.is_active = False
                    logger.info(f"Archived conversation {conversation_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error archiving conversation: {str(e)}")
            return False
    
    def delete_conversation(self, conversation_id: int) -> bool:
        """Permanently delete a conversation and all related data"""
        try:
            with self.db_manager.get_session() as session:
                conversation = session.query(Conversation).get(conversation_id)
                if conversation:
                    session.delete(conversation)
                    logger.info(f"Deleted conversation {conversation_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error deleting conversation: {str(e)}")
            return False
    
    # Character State Management
    def save_character_state(self, character_data: Dict[str, Any]) -> int:
        """Save or update character state"""
        try:
            with self.db_manager.get_session() as session:
                # Check if character exists
                existing = None
                if "id" in character_data:
                    existing = session.query(CharacterState).get(character_data["id"])
                
                if existing:
                    # Update existing character
                    for key, value in character_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    character = existing
                else:
                    # Create new character
                    character = CharacterState(**character_data)
                    session.add(character)
                
                session.flush()
                logger.info(f"Saved character state: {character.name} (ID: {character.id})")
                return character.id
                
        except Exception as e:
            logger.error(f"Error saving character state: {str(e)}")
            raise
    
    def get_character_states(self, conversation_id: int = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get character states"""
        try:
            with self.db_manager.get_session() as session:
                query = session.query(CharacterState)
                
                if conversation_id:
                    query = query.filter(CharacterState.conversation_id == conversation_id)
                
                if active_only:
                    query = query.filter(CharacterState.is_active == True)
                
                characters = query.order_by(CharacterState.last_seen.desc()).all()
                
                return [
                    {
                        "id": char.id,
                        "name": char.name,
                        "description": char.description,
                        "personality_traits": char.personality_traits,
                        "background": char.background,
                        "current_mood": char.current_mood,
                        "current_location": char.current_location,
                        "relationships": char.relationships,
                        "inventory": char.inventory,
                        "stats": char.stats,
                        "last_seen": char.last_seen,
                        "is_active": char.is_active
                    }
                    for char in characters
                ]
                
        except Exception as e:
            logger.error(f"Error getting character states: {str(e)}")
            return []
    
    def find_character_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find character by name"""
        try:
            with self.db_manager.get_session() as session:
                character = (session.query(CharacterState)
                           .filter(CharacterState.name.ilike(f"%{name}%"))
                           .filter(CharacterState.is_active == True)
                           .first())
                
                if character:
                    return {
                        "id": character.id,
                        "name": character.name,
                        "description": character.description,
                        "personality_traits": character.personality_traits,
                        "background": character.background,
                        "current_mood": character.current_mood,
                        "current_location": character.current_location,
                        "relationships": character.relationships,
                        "inventory": character.inventory,
                        "stats": character.stats
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error finding character: {str(e)}")
            return None
    
    # Props and Items Management
    def save_prop_item(self, prop_data: Dict[str, Any]) -> int:
        """Save or update prop item"""
        try:
            with self.db_manager.get_session() as session:
                # Check if prop exists
                existing = None
                if "id" in prop_data:
                    existing = session.query(PropItem).get(prop_data["id"])
                
                if existing:
                    # Update existing prop
                    for key, value in prop_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    prop = existing
                else:
                    # Create new prop
                    prop = PropItem(**prop_data)
                    session.add(prop)
                
                session.flush()
                logger.info(f"Saved prop item: {prop.name} (ID: {prop.id})")
                return prop.id
                
        except Exception as e:
            logger.error(f"Error saving prop item: {str(e)}")
            raise
    
    def get_prop_items(self, category: str = None, search_term: str = None, 
                      favorites_only: bool = False, limit: int = 100) -> List[Dict[str, Any]]:
        """Get prop items with filtering"""
        try:
            with self.db_manager.get_session() as session:
                query = session.query(PropItem)
                
                if category:
                    query = query.filter(PropItem.category == category)
                
                if search_term:
                    search_filter = or_(
                        PropItem.name.ilike(f"%{search_term}%"),
                        PropItem.description.ilike(f"%{search_term}%")
                    )
                    query = query.filter(search_filter)
                
                if favorites_only:
                    query = query.filter(PropItem.is_favorite == True)
                
                props = (query.order_by(desc(PropItem.usage_count), PropItem.name)
                        .limit(limit)
                        .all())
                
                return [
                    {
                        "id": prop.id,
                        "name": prop.name,
                        "category": prop.category,
                        "subcategory": prop.subcategory,
                        "description": prop.description,
                        "image_path": prop.image_path,
                        "tags": prop.tags,
                        "rarity": prop.rarity,
                        "properties": prop.properties,
                        "usage_count": prop.usage_count,
                        "is_favorite": prop.is_favorite
                    }
                    for prop in props
                ]
                
        except Exception as e:
            logger.error(f"Error getting prop items: {str(e)}")
            return []
    
    def increment_prop_usage(self, prop_id: int):
        """Increment usage count for a prop item"""
        try:
            with self.db_manager.get_session() as session:
                prop = session.query(PropItem).get(prop_id)
                if prop:
                    prop.usage_count += 1
                    prop.last_used = datetime.utcnow()
                    
        except Exception as e:
            logger.error(f"Error incrementing prop usage: {str(e)}")
    
    # Agent Configuration Management
    def save_agent_configuration(self, agent_data: Dict[str, Any]) -> int:
        """Save or update agent configuration"""
        try:
            with self.db_manager.get_session() as session:
                # Check if agent exists
                existing = None
                if "id" in agent_data:
                    existing = session.query(AgentConfiguration).get(agent_data["id"])
                elif "name" in agent_data:
                    existing = (session.query(AgentConfiguration)
                              .filter(AgentConfiguration.name == agent_data["name"])
                              .first())
                
                if existing:
                    # Update existing agent
                    for key, value in agent_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    agent = existing
                else:
                    # Create new agent
                    agent = AgentConfiguration(**agent_data)
                    session.add(agent)
                
                session.flush()
                logger.info(f"Saved agent configuration: {agent.name} (ID: {agent.id})")
                return agent.id
                
        except Exception as e:
            logger.error(f"Error saving agent configuration: {str(e)}")
            raise
    
    def get_agent_configurations(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """Get agent configurations"""
        try:
            with self.db_manager.get_session() as session:
                query = session.query(AgentConfiguration)
                
                if enabled_only:
                    query = query.filter(AgentConfiguration.enabled == True)
                
                agents = query.order_by(AgentConfiguration.name).all()
                
                return [
                    {
                        "id": agent.id,
                        "name": agent.name,
                        "display_name": agent.display_name,
                        "system_prompt": agent.system_prompt,
                        "enabled": agent.enabled,
                        "llm_provider": agent.llm_provider,
                        "temperature": agent.temperature,
                        "max_tokens": agent.max_tokens,
                        "capabilities": agent.capabilities,
                        "metadata": agent.metadata
                    }
                    for agent in agents
                ]
                
        except Exception as e:
            logger.error(f"Error getting agent configurations: {str(e)}")
            return []
    
    # Memory Search and Retrieval
    def search_conversations(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search conversations by content"""
        try:
            with self.db_manager.get_session() as session:
                # Search in conversation summaries and message content
                results = (session.query(Conversation)
                          .join(Message, Conversation.id == Message.conversation_id)
                          .filter(
                              or_(
                                  Conversation.title.ilike(f"%{query}%"),
                                  Conversation.summary.ilike(f"%{query}%"),
                                  Message.content.ilike(f"%{query}%")
                              )
                          )
                          .distinct()
                          .order_by(desc(Conversation.updated_at))
                          .limit(limit)
                          .all())
                
                return [
                    {
                        "id": conv.id,
                        "title": conv.title,
                        "summary": conv.summary,
                        "updated_at": conv.updated_at
                    }
                    for conv in results
                ]
                
        except Exception as e:
            logger.error(f"Error searching conversations: {str(e)}")
            return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            with self.db_manager.get_session() as session:
                stats = {
                    "conversations": session.query(Conversation).count(),
                    "active_conversations": session.query(Conversation).filter(Conversation.is_active == True).count(),
                    "total_messages": session.query(Message).count(),
                    "characters": session.query(CharacterState).count(),
                    "active_characters": session.query(CharacterState).filter(CharacterState.is_active == True).count(),
                    "prop_items": session.query(PropItem).count(),
                    "agent_configs": session.query(AgentConfiguration).count(),
                    "enabled_agents": session.query(AgentConfiguration).filter(AgentConfiguration.enabled == True).count()
                }
                
                # Get recent activity
                week_ago = datetime.utcnow() - timedelta(days=7)
                stats["messages_this_week"] = (session.query(Message)
                                             .filter(Message.timestamp >= week_ago)
                                             .count())
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            return {}
