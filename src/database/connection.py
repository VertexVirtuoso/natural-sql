"""Database connection management."""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
from datetime import datetime

import mysql.connector
from mysql.connector import Error as MySQLError
from mysql.connector.pooling import MySQLConnectionPool

from ..config.settings import get_settings
from .models import QueryResult, TableSchema, DatabaseSchema

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages database connections and operations."""
    
    def __init__(self, settings=None):
        """Initialize the database connection manager."""
        self.settings = settings or get_settings()
        self._pool: Optional[MySQLConnectionPool] = None
        self._setup_connection_pool()
    
    def _setup_connection_pool(self) -> None:
        """Set up the MySQL connection pool."""
        try:
            config = {
                'host': self.settings.db_host,
                'port': self.settings.db_port,
                'database': self.settings.db_name,
                'user': self.settings.db_user,
                'password': self.settings.db_password,
                'pool_name': 'natural_sql_pool',
                'pool_size': self.settings.db_pool_size,
                'pool_reset_session': True,
                'charset': 'utf8mb4',
                'collation': 'utf8mb4_unicode_ci',
                'autocommit': True
            }
            
            self._pool = MySQLConnectionPool(**config)
            logger.info("Database connection pool created successfully")
            
        except MySQLError as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        connection = None
        try:
            connection = self._pool.get_connection()
            yield connection
        except MySQLError as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()
    
    def execute_query(self, query: str) -> QueryResult:
        """Execute a SQL query and return the results."""
        start_time = time.time()
        
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # Execute the query
                cursor.execute(query)
                
                # Get column names
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                
                # Fetch all rows
                rows = cursor.fetchall() if cursor.description else []
                
                # Convert rows to list of lists
                if rows:
                    rows = [list(row) for row in rows]
                else:
                    rows = []
                
                execution_time = time.time() - start_time
                
                result = QueryResult(
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    execution_time=execution_time,
                    query=query,
                    timestamp=datetime.now()
                )
                
                cursor.close()
                logger.info(f"Query executed successfully in {execution_time:.3f}s")
                return result
                
        except MySQLError as e:
            execution_time = time.time() - start_time
            logger.error(f"Query execution failed: {e}")
            
            return QueryResult(
                columns=[],
                rows=[],
                row_count=0,
                execution_time=execution_time,
                query=query,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                return True
        except MySQLError as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_database_schema(self) -> DatabaseSchema:
        """Get the complete database schema."""
        tables = []
        
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # Get all table names
                cursor.execute("SHOW TABLES")
                table_names = [row[0] for row in cursor.fetchall()]
                
                for table_name in table_names:
                    # Get column information
                    cursor.execute(f"DESCRIBE `{table_name}`")
                    columns_info = cursor.fetchall()
                    
                    columns = []
                    primary_keys = []
                    
                    for col_info in columns_info:
                        column = {
                            'name': col_info[0],
                            'type': col_info[1],
                            'nullable': col_info[2] == 'YES',
                            'key': col_info[3],
                            'default': col_info[4],
                            'extra': col_info[5]
                        }
                        columns.append(column)
                        
                        if col_info[3] == 'PRI':
                            primary_keys.append(col_info[0])
                    
                    # Get foreign key information
                    cursor.execute(f"""
                        SELECT 
                            COLUMN_NAME,
                            REFERENCED_TABLE_NAME,
                            REFERENCED_COLUMN_NAME
                        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                        WHERE TABLE_SCHEMA = %s 
                        AND TABLE_NAME = %s 
                        AND REFERENCED_TABLE_NAME IS NOT NULL
                    """, (self.settings.db_name, table_name))
                    
                    foreign_keys = []
                    for fk_info in cursor.fetchall():
                        foreign_keys.append({
                            'column': fk_info[0],
                            'referenced_table': fk_info[1],
                            'referenced_column': fk_info[2]
                        })
                    
                    table_schema = TableSchema(
                        name=table_name,
                        columns=columns,
                        primary_keys=primary_keys,
                        foreign_keys=foreign_keys
                    )
                    tables.append(table_schema)
                
                cursor.close()
                
        except MySQLError as e:
            logger.error(f"Failed to get database schema: {e}")
            raise
        
        return DatabaseSchema(
            tables=tables,
            database_name=self.settings.db_name
        )
    
    def get_table_names(self) -> List[str]:
        """Get a list of all table names in the database."""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute("SHOW TABLES")
                table_names = [row[0] for row in cursor.fetchall()]
                cursor.close()
                return table_names
        except MySQLError as e:
            logger.error(f"Failed to get table names: {e}")
            return []
    
    def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            logger.info("Closing database connection pool")


# Global database connection instance
_db_connection: Optional[DatabaseConnection] = None


def get_db_connection() -> DatabaseConnection:
    """Get the global database connection instance."""
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
    return _db_connection