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
