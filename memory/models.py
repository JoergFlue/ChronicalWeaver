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
