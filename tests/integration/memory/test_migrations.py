import pytest
from memory.database_manager import DatabaseManager
from memory.models import Base, Conversation, Message, AgentConfiguration
from sqlalchemy import inspect, text

def test_schema_migration(tmp_path):
    # Setup initial database
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    db_manager = DatabaseManager(db_url)
    Base.metadata.create_all(bind=db_manager.engine)

    # Add sample data
    with db_manager.get_session() as session:
        conv = Conversation(title="Migration Test")
        session.add(conv)
        session.flush()
        msg = Message(conversation_id=conv.id, role="user", content="Hello")
        session.add(msg)
        session.flush()
        agent = AgentConfiguration(name="TestAgent", system_prompt="Prompt", enabled=True)
        session.add(agent)
        session.flush()

    # Simulate migration: add a new column
    with db_manager.engine.connect() as conn:
        conn.execute(text("ALTER TABLE agent_configurations ADD COLUMN test_field TEXT"))

    # Verify new column exists
    inspector = inspect(db_manager.engine)
    columns = [col['name'] for col in inspector.get_columns('agent_configurations')]
    assert "test_field" in columns

    # Verify data integrity after migration
    with db_manager.get_session() as session:
        agents = session.query(AgentConfiguration).all()
        assert len(agents) == 1
        assert agents[0].name == "TestAgent"
