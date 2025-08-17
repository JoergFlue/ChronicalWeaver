import pytest
from memory.database_manager import DatabaseManager
from memory.conversation_memory import ConversationMemory
from memory.persistent_memory import PersistentMemory

@pytest.fixture(scope="module")
def db_manager():
    db = DatabaseManager("sqlite:///:memory:")
    yield db
    db.close()

def test_create_and_load_conversation(db_manager):
    # Create a new conversation
    conv_memory = ConversationMemory(db_manager)
    conv_id = conv_memory.create_new_conversation("Integration Test Conversation")
    assert conv_id is not None

    # Add user and assistant messages
    user_msg = conv_memory.add_user_message("Hello, integration test!")
    assistant_msg = conv_memory.add_assistant_message("Hi, user!", agent_name="TestAgent")
    assert user_msg.id is not None
    assert assistant_msg.id is not None

    # Load conversation history
    conv_memory_loaded = ConversationMemory(db_manager, conversation_id=conv_id)
    messages = conv_memory_loaded.get_messages()
    assert len(messages) == 2
    assert messages[0].content == "Hello, integration test!"
    assert messages[1].content == "Hi, user!"

def test_persistent_memory_conversations(db_manager):
    persistent = PersistentMemory(db_manager)
    # Create and archive a conversation
    conv_memory = ConversationMemory(db_manager)
    conv_id = conv_memory.create_new_conversation("Persistent Conversation")
    assert conv_id is not None
    archived = persistent.archive_conversation(conv_id)
    assert archived is True

    # Retrieve conversations
    conversations = persistent.get_conversations(limit=10, include_inactive=True)
    assert any(conv["id"] == conv_id for conv in conversations)
