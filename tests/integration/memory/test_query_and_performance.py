import pytest
from memory.database_manager import DatabaseManager
from memory.models import Base, Conversation, Message, AgentConfiguration, CharacterState, PropItem, ConversationSummary, UserPreference
from sqlalchemy import text

@pytest.fixture(scope="module")
def db_manager(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    db_url = f"sqlite:///{db_path}"
    manager = DatabaseManager(db_url)
    Base.metadata.create_all(bind=manager.engine)
    return manager

def test_query_filter_conversation(db_manager):
    with db_manager.get_session() as session:
        conv1 = Conversation(title="Adventure One", is_active=True)
        conv2 = Conversation(title="Adventure Two", is_active=False)
        session.add_all([conv1, conv2])
        session.flush()
        active_convs = session.query(Conversation).filter_by(is_active=True).all()
        assert len(active_convs) == 1
        assert active_convs[0].title == "Adventure One"

def test_query_filter_message(db_manager):
    with db_manager.get_session() as session:
        conv = Conversation(title="MsgTest")
        session.add(conv)
        session.flush()
        msg1 = Message(conversation_id=conv.id, role="user", content="Hello")
        msg2 = Message(conversation_id=conv.id, role="assistant", content="Hi")
        session.add_all([msg1, msg2])
        session.flush()
        user_msgs = session.query(Message).filter_by(role="user").all()
        assert any(m.content == "Hello" for m in user_msgs)

def test_query_filter_agent_configuration(db_manager):
    with db_manager.get_session() as session:
        agent1 = AgentConfiguration(name="AgentA", system_prompt="PromptA", enabled=True)
        agent2 = AgentConfiguration(name="AgentB", system_prompt="PromptB", enabled=False)
        session.add_all([agent1, agent2])
        session.flush()
        enabled_agents = session.query(AgentConfiguration).filter_by(enabled=True).all()
        assert len(enabled_agents) == 1
        assert enabled_agents[0].name == "AgentA"

def test_query_filter_character_state(db_manager):
    with db_manager.get_session() as session:
        char1 = CharacterState(name="Alice", is_active=True)
        char2 = CharacterState(name="Bob", is_active=False)
        session.add_all([char1, char2])
        session.flush()
        active_chars = session.query(CharacterState).filter_by(is_active=True).all()
        assert len(active_chars) == 1
        assert active_chars[0].name == "Alice"

def test_query_filter_prop_item(db_manager):
    with db_manager.get_session() as session:
        prop1 = PropItem(name="Sword", category="Weapon", is_favorite=True)
        prop2 = PropItem(name="Hat", category="Clothing", is_favorite=False)
        session.add_all([prop1, prop2])
        session.flush()
        fav_props = session.query(PropItem).filter_by(is_favorite=True).all()
        assert len(fav_props) == 1
        assert fav_props[0].name == "Sword"

def test_query_filter_user_preference(db_manager):
    with db_manager.get_session() as session:
        pref1 = UserPreference(key="theme", value={"mode": "dark"}, category="ui")
        pref2 = UserPreference(key="lang", value={"code": "en"}, category="general")
        session.add_all([pref1, pref2])
        session.flush()
        ui_prefs = session.query(UserPreference).filter_by(category="ui").all()
        assert len(ui_prefs) == 1
        assert ui_prefs[0].key == "theme"

def test_performance_large_dataset(db_manager):
    import time
    NUM_CONV = 1000
    NUM_MSG = 5000
    # Insert many conversations
    with db_manager.get_session() as session:
        for i in range(NUM_CONV):
            conv = Conversation(title=f"Conv {i}", is_active=(i % 2 == 0))
            session.add(conv)
        session.flush()
        conv_ids = [c.id for c in session.query(Conversation).all()]
        # Insert many messages
        for i in range(NUM_MSG):
            msg = Message(conversation_id=conv_ids[i % NUM_CONV], role="user", content=f"Msg {i}")
            session.add(msg)
        session.flush()
    # Measure query time for active conversations
    start = time.time()
    with db_manager.get_session() as session:
        active_convs = session.query(Conversation).filter_by(is_active=True).all()
        assert len(active_convs) == NUM_CONV // 2
    duration = time.time() - start
    assert duration < 1.5  # Query should complete quickly
    # Measure query time for messages
    start = time.time()
    with db_manager.get_session() as session:
        msgs = session.query(Message).filter_by(role="user").limit(1000).all()
        assert len(msgs) == 1000
    duration = time.time() - start
    assert duration < 1.5

# CRUD tests for persistent data types

def test_crud_conversation(db_manager):
    with db_manager.get_session() as session:
        # Create
        conv = Conversation(title="CRUD Test", is_active=True)
        session.add(conv)
        session.flush()
        conv_id = conv.id
        # Read
        fetched = session.query(Conversation).get(conv_id)
        assert fetched.title == "CRUD Test"
        # Update
        fetched.title = "CRUD Updated"
        session.flush()
        updated = session.query(Conversation).get(conv_id)
        assert updated.title == "CRUD Updated"
        # Delete
        session.delete(updated)
        session.flush()
        deleted = session.query(Conversation).get(conv_id)
        assert deleted is None

def test_crud_message(db_manager):
    with db_manager.get_session() as session:
        conv = Conversation(title="Msg CRUD")
        session.add(conv)
        session.flush()
        # Create
        msg = Message(conversation_id=conv.id, role="user", content="Initial")
        session.add(msg)
        session.flush()
        msg_id = msg.id
        # Read
        fetched = session.query(Message).get(msg_id)
        assert fetched.content == "Initial"
        # Update
        fetched.content = "Updated"
        session.flush()
        updated = session.query(Message).get(msg_id)
        assert updated.content == "Updated"
        # Delete
        session.delete(updated)
        session.flush()
        deleted = session.query(Message).get(msg_id)
        assert deleted is None

def test_crud_agent_configuration(db_manager):
    with db_manager.get_session() as session:
        # Create
        agent = AgentConfiguration(name="CRUDAgent", system_prompt="Prompt", enabled=True)
        session.add(agent)
        session.flush()
        agent_id = agent.id
        # Read
        fetched = session.query(AgentConfiguration).get(agent_id)
        assert fetched.name == "CRUDAgent"
        # Update
        fetched.enabled = False
        session.flush()
        updated = session.query(AgentConfiguration).get(agent_id)
        assert updated.enabled is False
        # Delete
        session.delete(updated)
        session.flush()
        deleted = session.query(AgentConfiguration).get(agent_id)
        assert deleted is None

def test_crud_character_state(db_manager):
    with db_manager.get_session() as session:
        # Create
        char = CharacterState(name="CRUDChar", is_active=True)
        session.add(char)
        session.flush()
        char_id = char.id
        # Read
        fetched = session.query(CharacterState).get(char_id)
        assert fetched.name == "CRUDChar"
        # Update
        fetched.is_active = False
        session.flush()
        updated = session.query(CharacterState).get(char_id)
        assert updated.is_active is False
        # Delete
        session.delete(updated)
        session.flush()
        deleted = session.query(CharacterState).get(char_id)
        assert deleted is None

def test_crud_prop_item(db_manager):
    with db_manager.get_session() as session:
        # Create
        prop = PropItem(name="CRUDProp", category="Test", is_favorite=True)
        session.add(prop)
        session.flush()
        prop_id = prop.id
        # Read
        fetched = session.query(PropItem).get(prop_id)
        assert fetched.name == "CRUDProp"
        # Update
        fetched.is_favorite = False
        session.flush()
        updated = session.query(PropItem).get(prop_id)
        assert updated.is_favorite is False
        # Delete
        session.delete(updated)
        session.flush()
        deleted = session.query(PropItem).get(prop_id)
        assert deleted is None

# Memory search/retrieval accuracy tests

def test_search_conversation_title_partial(db_manager):
    with db_manager.get_session() as session:
        conv1 = Conversation(title="Epic Adventure", is_active=True)
        conv2 = Conversation(title="Epic Quest", is_active=True)
        conv3 = Conversation(title="Side Story", is_active=True)
        session.add_all([conv1, conv2, conv3])
        session.flush()
        results = session.query(Conversation).filter(Conversation.title.like("%Epic%")).all()
        titles = [c.title for c in results]
        assert "Epic Adventure" in titles
        assert "Epic Quest" in titles
        assert "Side Story" not in titles

def test_search_message_content_partial(db_manager):
    with db_manager.get_session() as session:
        conv = Conversation(title="SearchTest")
        session.add(conv)
        session.flush()
        msg1 = Message(conversation_id=conv.id, role="user", content="Find the magic sword")
        msg2 = Message(conversation_id=conv.id, role="assistant", content="Sword found")
        msg3 = Message(conversation_id=conv.id, role="user", content="Look for shield")
        session.add_all([msg1, msg2, msg3])
        session.flush()
        results = session.query(Message).filter(Message.content.like("%sword%")).all()
        contents = [m.content for m in results]
        assert "Find the magic sword" in contents
        assert "Sword found" in contents
        assert "Look for shield" not in contents

def test_search_agent_configuration_by_prompt(db_manager):
    with db_manager.get_session() as session:
        agent1 = AgentConfiguration(name="SearchAgent", system_prompt="Track all prompts", enabled=True)
        agent2 = AgentConfiguration(name="HelperAgent", system_prompt="Assist with tasks", enabled=True)
        session.add_all([agent1, agent2])
        session.flush()
        results = session.query(AgentConfiguration).filter(AgentConfiguration.system_prompt.like("%prompt%")).all()
        names = [a.name for a in results]
        assert "SearchAgent" in names
        assert "HelperAgent" not in names

def test_search_character_state_by_name_partial(db_manager):
    with db_manager.get_session() as session:
        char1 = CharacterState(name="Alicia", is_active=True)
        char2 = CharacterState(name="Bob", is_active=True)
        char3 = CharacterState(name="Al", is_active=True)
        session.add_all([char1, char2, char3])
        session.flush()
        results = session.query(CharacterState).filter(CharacterState.name.like("Ali%")).all()
        names = [c.name for c in results]
        assert "Alicia" in names
        assert "Al" not in names
        assert "Bob" not in names

def test_search_prop_item_by_tag(db_manager):
    with db_manager.get_session() as session:
        prop1 = PropItem(name="Sword", category="Weapon", tags=["sharp", "metal"])
        prop2 = PropItem(name="Hat", category="Clothing", tags=["soft", "fabric"])
        session.add_all([prop1, prop2])
        session.flush()
        results = session.query(PropItem).filter(PropItem.tags.like("%metal%")).all()
        names = [p.name for p in results]
        assert "Sword" in names
        assert "Hat" not in names
