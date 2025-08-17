import pytest
from memory.models import (
    Base, Conversation, Message, AgentConfiguration, CharacterState,
    PropItem, ConversationSummary, UserPreference
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="module")
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_conversation_model(db_session):
    conv = Conversation(title="Test Conversation")
    db_session.add(conv)
    db_session.commit()
    assert conv.id is not None
    assert conv.title == "Test Conversation"

def test_message_model(db_session):
    conv = Conversation(title="Msg Conversation")
    db_session.add(conv)
    db_session.commit()
    msg = Message(conversation_id=conv.id, role="user", content="Hello")
    db_session.add(msg)
    db_session.commit()
    assert msg.id is not None
    assert msg.role == "user"
    assert msg.content == "Hello"

def test_agent_configuration_model(db_session):
    agent = AgentConfiguration(
        name="test_agent",
        system_prompt="Test prompt",
        enabled=True
    )
    db_session.add(agent)
    db_session.commit()
    assert agent.id is not None
    assert agent.name == "test_agent"

def test_character_state_model(db_session):
    char = CharacterState(name="Alice", description="Main character")
    db_session.add(char)
    db_session.commit()
    assert char.id is not None
    assert char.name == "Alice"

def test_prop_item_model(db_session):
    prop = PropItem(name="Sword", category="Weapon")
    db_session.add(prop)
    db_session.commit()
    assert prop.id is not None
    assert prop.name == "Sword"

def test_conversation_summary_model(db_session):
    conv = Conversation(title="Summary Conversation")
    db_session.add(conv)
    db_session.commit()
    summary = ConversationSummary(conversation_id=conv.id, summary_text="Summary")
    db_session.add(summary)
    db_session.commit()
    assert summary.id is not None
    assert summary.summary_text == "Summary"

def test_user_preference_model(db_session):
    pref = UserPreference(key="theme", value={"dark": True})
    db_session.add(pref)
    db_session.commit()
    assert pref.id is not None
    assert pref.key == "theme"
