# Chronicle Weaver - Phase 2: Memory & Data Persistence

**Duration**: 2-3 Weeks  
**Implementation Confidence**: 80% - Medium Risk  
**Dependencies**: Phase 1 (Core LLM & Main Agent System)  
**Next Phase**: Phase 3 (Agent Management & Core Sub-Agents)

## Overview
Implement robust memory and data persistence systems that enable Chronicle Weaver to maintain conversation context, character information, and user preferences across sessions. This phase creates the data foundation that supports advanced agent capabilities and long-term storytelling continuity.

## Key Risk Factors
- **LangChain memory integration complexity** - Advanced memory patterns may require custom implementation
- **SQLite performance with large datasets** - Conversation history growth and query optimization
- **Data migration strategies** - Schema changes during development and deployment
- **Memory consolidation algorithms** - Balancing detail retention with performance
- **Concurrent access patterns** - Multiple agents accessing shared memory safely

## Acceptance Criteria
- [ ] Short-term memory maintains conversation context within session
- [ ] Long-term memory persists data across application restarts
- [ ] SQLite database stores agent configurations, props, and interaction summaries
- [ ] Memory systems integrate seamlessly with Main Agent
- [ ] CRUD operations work for all data types
- [ ] Memory can be queried and filtered effectively
- [ ] Database migrations work for schema updates
- [ ] Memory performance is acceptable for large conversation histories
- [ ] Integration testing is performed to verify memory-agent and database interactions before and after each change

## Detailed Implementation Steps

### Week 1: Database Foundation & Short-Term Memory

#### 1.1 Database Schema Design (`src/memory/models.py`)

```python
"""Database models for Chronicle Weaver"""
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, 
    Float, Boolean, JSON, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
from typing import Dict, Any, Optional
import json

Base = declarative_base()

class Conversation(Base):
    """Conversation session tracking"""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    summary = Column(Text)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    character_states = relationship("CharacterState", back_populates="conversation")
    
    __table_args__ = (
        Index('idx_conversation_created', 'created_at'),
        Index('idx_conversation_active', 'is_active'),
    )

class Message(Base):
    """Individual messages in conversations"""
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    agent_name = Column(String(100))
    llm_provider = Column(String(50))
    token_count = Column(Integer)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    __table_args__ = (
        Index('idx_message_conversation', 'conversation_id'),
        Index('idx_message_timestamp', 'timestamp'),
        Index('idx_message_role', 'role'),
    )

class AgentConfiguration(Base):
    """Stored agent configurations"""
    __tablename__ = 'agent_configurations'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    display_name = Column(String(200))
    system_prompt = Column(Text, nullable=False)
    enabled = Column(Boolean, default=True)
    llm_provider = Column(String(50))
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=2048)
    capabilities = Column(JSON, default=list)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_agent_name', 'name'),
        Index('idx_agent_enabled', 'enabled'),
    )

class CharacterState(Base):
    """Character information and state tracking"""
    __tablename__ = 'character_states'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    description = Column(Text)
    personality_traits = Column(JSON, default=list)
    background = Column(Text)
    current_mood = Column(String(50))
    current_location = Column(String(200))
    relationships = Column(JSON, default=dict)
    inventory = Column(JSON, default=list)
    stats = Column(JSON, default=dict)
    last_seen = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="character_states")
    
    __table_args__ = (
        Index('idx_character_name', 'name'),
        Index('idx_character_conversation', 'conversation_id'),
        Index('idx_character_active', 'is_active'),
    )

class PropItem(Base):
    """Props and items library"""
    __tablename__ = 'prop_items'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)
    subcategory = Column(String(50))
    description = Column(Text)
    image_path = Column(String(500))
    tags = Column(JSON, default=list)
    rarity = Column(String(20), default='common')
    properties = Column(JSON, default=dict)
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_favorite = Column(Boolean, default=False)
    
    __table_args__ = (
        Index('idx_prop_category', 'category'),
        Index('idx_prop_name', 'name'),
        Index('idx_prop_tags', 'tags'),
        Index('idx_prop_favorite', 'is_favorite'),
    )

class ConversationSummary(Base):
    """Summarized conversation history for long-term memory"""
    __tablename__ = 'conversation_summaries'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    summary_text = Column(Text, nullable=False)
    key_events = Column(JSON, default=list)
    character_developments = Column(JSON, default=dict)
    plot_points = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    message_range_start = Column(Integer)  # First message ID in summary
    message_range_end = Column(Integer)    # Last message ID in summary
    
    __table_args__ = (
        Index('idx_summary_conversation', 'conversation_id'),
        Index('idx_summary_created', 'created_at'),
    )

class UserPreference(Base):
    """User preferences and settings"""
    __tablename__ = 'user_preferences'
    
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(JSON, nullable=False)
    category = Column(String(50), default='general')
    description = Column(String(500))
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_preference_key', 'key'),
        Index('idx_preference_category', 'category'),
    )
```

#### 1.2 Database Manager (`src/memory/database_manager.py`)

```python
"""Database connection and management"""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from pathlib import Path
import logging
from .models import Base
from typing import Generator, Optional

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, database_url: str = None):
        if database_url is None:
            # Default to SQLite in user data directory
            db_path = Path("data/chronicle_weaver.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite:///{db_path}"
        
        self.database_url = database_url
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables
        self._init_database()
        
        logger.info(f"Database initialized: {database_url}")
    
    def _create_engine(self):
        """Create database engine with appropriate settings"""
        if self.database_url.startswith("sqlite"):
            # SQLite-specific settings
            engine = create_engine(
                self.database_url,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30
                },
                echo=False  # Set to True for SQL debugging
            )
            
            # Enable foreign keys for SQLite
            @event.listens_for(engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.close()
                
        else:
            # Other database engines
            engine = create_engine(self.database_url, echo=False)
        
        return engine
    
    def _init_database(self):
        """Initialize database tables"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created/verified")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_session_sync(self) -> Session:
        """Get database session for synchronous operations"""
        return self.SessionLocal()
    
    def close(self):
        """Close database connections"""
        self.engine.dispose()
        logger.info("Database connections closed")
    
    def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            if self.database_url.startswith("sqlite"):
                import shutil
                source_path = self.database_url.replace("sqlite:///", "")
                shutil.copy2(source_path, backup_path)
                logger.info(f"Database backed up to: {backup_path}")
                return True
            else:
                logger.warning("Backup not implemented for non-SQLite databases")
                return False
        except Exception as e:
            logger.error(f"Database backup failed: {str(e)}")
            return False
    
    def vacuum_database(self):
        """Vacuum database to reclaim space"""
        try:
            if self.database_url.startswith("sqlite"):
                with self.engine.connect() as conn:
                    conn.execute("VACUUM")
                logger.info("Database vacuumed successfully")
        except Exception as e:
            logger.error(f"Database vacuum failed: {str(e)}")
```

#### 1.3 Short-Term Memory Implementation (`src/memory/conversation_memory.py`)

```python
"""Short-term conversation memory using LangChain"""
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing import List, Dict, Any, Optional
from .database_manager import DatabaseManager
from .models import Conversation, Message
import logging

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
```

### Week 2: Long-Term Memory & Data Operations

#### 2.1 Long-Term Memory Manager (`src/memory/persistent_memory.py`)

```python
"""Long-term memory and data persistence manager"""
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from .database_manager import DatabaseManager
from .models import (
    Conversation, Message, ConversationSummary, 
    CharacterState, PropItem, AgentConfiguration, UserPreference
)
from .conversation_memory import ConversationMemory
import logging
from datetime import datetime, timedelta
import json

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
```

### Week 3: Memory Integration & Advanced Features

#### 2.2 Memory Manager Integration (`src/memory/memory_manager.py`)

```python
"""Unified memory management interface"""
from typing import Dict, Any, List, Optional
from .database_manager import DatabaseManager
from .conversation_memory import ConversationMemory
from .persistent_memory import PersistentMemory
from .summarization import ConversationSummarizer
import logging

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
            characters = self.persistent_memory.search_characters(query, limit=3)
            context["characters"] = characters
            
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
```

#### 2.3 Conversation Summarization (`src/memory/summarization.py`)

```python
"""Conversation summarization utilities"""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConversationSummarizer:
    """Creates summaries of conversation history"""
    
    def __init__(self):
        pass
    
    def create_summary(self, messages: List[str], target_length: int = 200) -> str:
        """Create a summary of conversation messages"""
        try:
            # Simple extractive summarization for now
            # In a full implementation, this would use an LLM or specialized summarization model
            
            if len(messages) <= 3:
                return " | ".join(messages)
            
            # Extract key information
            key_points = []
            character_mentions = set()
            locations = set()
            actions = []
            
            for message in messages:
                # Extract character names (simple heuristic)
                words = message.split()
                for i, word in enumerate(words):
                    if word.istitle() and len(word) > 2:
                        character_mentions.add(word)
                
                # Extract actions (messages with verbs)
                if any(verb in message.lower() for verb in ['went', 'said', 'did', 'came', 'left', 'found']):
                    actions.append(message[:100])  # Truncate long actions
            
            # Build summary
            summary_parts = []
            
            if character_mentions:
                char_list = list(character_mentions)[:3]  # Limit to 3 characters
                summary_parts.append(f"Characters involved: {', '.join(char_list)}")
            
            if actions:
                key_actions = actions[:3]  # Limit to 3 key actions
                summary_parts.append(f"Key events: {' | '.join(key_actions)}")
            
            summary = ". ".join(summary_parts)
            
            # Truncate if too long
            if len(summary) > target_length:
                summary = summary[:target_length-3] + "..."
            
            return summary or "Conversation in progress"
            
        except Exception as e:
            logger.error(f"Error creating summary: {str(e)}")
            return "Summary unavailable"
    
    def extract_key_events(self, messages: List[str]) -> List[str]:
        """Extract key events from conversation"""
        events = []
        
        try:
            for message in messages:
                # Look for action indicators
                if any(indicator in message.lower() for indicator in 
                      ['suddenly', 'then', 'next', 'after', 'before', 'when']):
                    events.append(message[:150])  # Truncate long events
            
            return events[:5]  # Return top 5 events
            
        except Exception as e:
            logger.error(f"Error extracting key events: {str(e)}")
            return []
    
    def extract_character_developments(self, messages: List[str]) -> Dict[str, List[str]]:
        """Extract character developments from conversation"""
        developments = {}
        
        try:
            # Simple implementation - in practice would use NLP
            for message in messages:
                # Look for character development keywords
                if any(keyword in message.lower() for keyword in 
                      ['feels', 'thinks', 'believes', 'remembers', 'learns']):
                    # Extract character name (simplified)
                    words = message.split()
                    for word in words:
                        if word.istitle() and len(word) > 2:
                            if word not in developments:
                                developments[word] = []
                            developments[word].append(message[:100])
                            break
            
            return developments
            
        except Exception as e:
            logger.error(f"Error extracting character developments: {str(e)}")
            return {}
```

## Testing Strategy

### Unit Tests (`tests/unit/memory/`)
- **Database Models**: Test model creation, relationships, constraints
- **Memory Operations**: Test conversation memory, persistent storage
- **Data Validation**: Test data integrity and validation rules

### Integration Tests (`tests/integration/memory/`)
- **Memory-Agent Integration**: Test memory access from agents
- **Database Performance**: Test query performance with large datasets
- **Memory Consolidation**: Test summarization and cleanup processes

#### Integration Testing
- Integration testing is required before and after each change to verify that memory-agent and database interactions work as intended.
- After each integration test, verify the workability of `main.py` to ensure the application entry point remains functional.

### Performance Tests
- **Large Conversation Handling**: Test with 1000+ messages
- **Concurrent Access**: Test multiple agents accessing memory
- **Database Query Optimization**: Benchmark critical queries

## Error Handling Strategy
- **Database Connection Failures**: Graceful degradation with local caching
- **Memory Overflow**: Automatic cleanup and summarization
- **Data Corruption**: Backup and recovery mechanisms
- **Migration Failures**: Rollback strategies for schema changes

## Success Metrics
- [ ] Conversation context maintained across sessions
- [ ] Database queries execute under 100ms for normal operations
- [ ] Memory cleanup processes run without blocking
- [ ] Data integrity maintained through all operations
- [ ] Character and prop data persists and retrieves correctly
- [ ] Memory search returns relevant results within 200ms

## Deliverables
1. **Database Layer** - Complete SQLite schema with optimized queries
2. **Short-Term Memory** - LangChain integration with session persistence
3. **Long-Term Memory** - Persistent storage for all application data
4. **Memory Manager** - Unified interface for all memory operations
5. **Summarization System** - Automatic conversation summarization
6. **Data Migration Tools** - Schema versioning and migration support

## Handoff to Phase 3
Phase 2 provides Phase 3 with:
- Complete data persistence layer for agent configurations
- Memory interfaces for agent communication history
- Character and prop data storage for advanced agents
- Foundation for agent performance tracking and optimization

The memory system enables Phase 3 to implement sophisticated agent behaviors that leverage conversation history, character consistency, and contextual awareness.
