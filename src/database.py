"""
Database management module for RetentionAI.

This module provides a generic DatabaseManager class using SQLAlchemy for
database operations including connection management, table creation, and
CRUD operations for the churn prediction application.
"""

import logging
import pandas as pd
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Float, 
    Boolean, DateTime, text, inspect
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, Session

try:
    from .config import DATABASE_URL, RANDOM_SEED
except ImportError:
    from config import DATABASE_URL, RANDOM_SEED

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Generic database manager for SQLite operations using SQLAlchemy.
    
    Provides connection management, table operations, and data manipulation
    methods for the RetentionAI application.
    """
    
    def __init__(self, database_url: str = DATABASE_URL):
        """
        Initialize DatabaseManager with SQLAlchemy engine.
        
        Args:
            database_url: SQLAlchemy database URL
        """
        self.database_url = database_url
        self.engine: Optional[Engine] = None
        self.session_maker: Optional[sessionmaker] = None
        self.metadata = MetaData()
        
        self._connect()
    
    def _connect(self) -> None:
        """Establish database connection and create session maker."""
        try:
            self.engine = create_engine(
                self.database_url,
                echo=False,  # Set to True for SQL debugging
                pool_pre_ping=True,
                connect_args={"check_same_thread": False}  # For SQLite
            )
            
            self.session_maker = sessionmaker(bind=self.engine)
            logger.info(f"Database connection established: {self.database_url}")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions.
        
        Yields:
            Session: SQLAlchemy session
        """
        session = self.session_maker()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self) -> None:
        """Create all tables defined in metadata."""
        try:
            self.metadata.create_all(self.engine)
            logger.info("All tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            bool: True if table exists, False otherwise
        """
        try:
            inspector = inspect(self.engine)
            return table_name in inspector.get_table_names()
        except SQLAlchemyError as e:
            logger.error(f"Failed to check table existence: {e}")
            return False
    
    def drop_table(self, table_name: str) -> None:
        """
        Drop a table if it exists.
        
        Args:
            table_name: Name of the table to drop
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                conn.commit()
            logger.info(f"Table '{table_name}' dropped successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to drop table '{table_name}': {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as pandas DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            pd.DataFrame: Query results
        """
        try:
            df = pd.read_sql_query(query, self.engine, params=params)
            logger.info(f"Query executed successfully, returned {len(df)} rows")
            return df
        except SQLAlchemyError as e:
            logger.error(f"Failed to execute query: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error executing query: {e}")
            raise
    
    def insert_dataframe(
        self, 
        df: pd.DataFrame, 
        table_name: str, 
        if_exists: str = 'replace',
        index: bool = False,
        method: str = 'multi'
    ) -> None:
        """
        Insert pandas DataFrame into database table.
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            if_exists: How to behave if table exists {'fail', 'replace', 'append'}
            index: Whether to write DataFrame index
            method: Method for insertion {'multi', 'None'}
        """
        try:
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists=if_exists,
                index=index,
                method=method
            )
            logger.info(
                f"DataFrame inserted into '{table_name}': {len(df)} rows, "
                f"mode='{if_exists}'"
            )
        except SQLAlchemyError as e:
            logger.error(f"Failed to insert DataFrame into '{table_name}': {e}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            dict: Table information including columns and row count
        """
        try:
            # Get column information
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name)
            
            # Get row count
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            count_result = self.execute_query(count_query)
            row_count = count_result['count'].iloc[0]
            
            return {
                'table_name': table_name,
                'columns': [col['name'] for col in columns],
                'column_types': {col['name']: str(col['type']) for col in columns},
                'row_count': row_count
            }
        except SQLAlchemyError as e:
            logger.error(f"Failed to get table info for '{table_name}': {e}")
            raise
    
    def get_all_tables(self) -> List[str]:
        """
        Get list of all tables in the database.
        
        Returns:
            List[str]: List of table names
        """
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            logger.info(f"Found {len(tables)} tables in database")
            return tables
        except SQLAlchemyError as e:
            logger.error(f"Failed to get table list: {e}")
            raise
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get database connection information.
        
        Returns:
            dict: Connection information
        """
        return {
            'database_url': self.database_url,
            'engine_info': str(self.engine.url),
            'is_connected': self.engine is not None,
            'tables': self.get_all_tables() if self.engine else []
        }
    
    def close(self) -> None:
        """Close database connection."""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Database connection closed")
        except SQLAlchemyError as e:
            logger.error(f"Error closing database connection: {e}")


# Create a default database manager instance
def get_database_manager() -> DatabaseManager:
    """
    Get a default DatabaseManager instance.
    
    Returns:
        DatabaseManager: Default database manager
    """
    return DatabaseManager()


if __name__ == "__main__":
    # Test database connection and basic operations
    db_manager = get_database_manager()
    
    print("Database Connection Info:")
    info = db_manager.get_connection_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test basic query (will fail if no tables exist yet)
    try:
        tables = db_manager.get_all_tables()
        print(f"\nExisting tables: {tables}")
    except Exception as e:
        print(f"No tables found or error: {e}")
    
    print("\nDatabaseManager initialized successfully!")