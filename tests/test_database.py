"""Tests for database connection module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.database.connection import DatabaseConnection
from src.database.models import QueryResult, TableSchema, DatabaseSchema
from src.config.settings import Settings


class TestDatabaseConnection:
    """Test cases for DatabaseConnection class."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock(spec=Settings)
        settings.db_host = "localhost"
        settings.db_port = 3306
        settings.db_name = "test_db"
        settings.db_user = "test_user"
        settings.db_password = "test_password"
        settings.db_pool_size = 5
        settings.db_max_overflow = 10
        settings.db_pool_timeout = 30
        return settings
    
    @patch('src.database.connection.MySQLConnectionPool')
    def test_setup_connection_pool(self, mock_pool, mock_settings):
        """Test connection pool setup."""
        db = DatabaseConnection(mock_settings)
        
        mock_pool.assert_called_once()
        call_args = mock_pool.call_args[1]
        assert call_args['host'] == 'localhost'
        assert call_args['port'] == 3306
        assert call_args['database'] == 'test_db'
        assert call_args['user'] == 'test_user'
        assert call_args['password'] == 'test_password'
        assert call_args['pool_size'] == 5
    
    @patch('src.database.connection.MySQLConnectionPool')
    def test_execute_query_success(self, mock_pool, mock_settings):
        """Test successful query execution."""
        # Setup mocks
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.description = [('id',), ('name',)]
        mock_cursor.fetchall.return_value = [(1, 'test'), (2, 'test2')]
        
        mock_pool_instance = Mock()
        mock_pool_instance.get_connection.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance
        
        db = DatabaseConnection(mock_settings)
        
        with patch.object(db, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_connection
            
            result = db.execute_query("SELECT * FROM test")
        
        assert result.is_success
        assert result.columns == ['id', 'name']
        assert result.rows == [[1, 'test'], [2, 'test2']]
        assert result.row_count == 2
        assert result.query == "SELECT * FROM test"
    
    @patch('src.database.connection.MySQLConnectionPool')
    def test_execute_query_failure(self, mock_pool, mock_settings):
        """Test query execution failure."""
        from mysql.connector import Error as MySQLError
        
        mock_connection = Mock()
        mock_connection.cursor.side_effect = MySQLError("Connection failed")
        
        mock_pool_instance = Mock()
        mock_pool_instance.get_connection.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance
        
        db = DatabaseConnection(mock_settings)
        
        with patch.object(db, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_connection
            mock_get_conn.return_value.__enter__.side_effect = MySQLError("Connection failed")
            
            result = db.execute_query("SELECT * FROM test")
        
        assert not result.is_success
        assert result.error == "Connection failed"
        assert result.row_count == 0
    
    @patch('src.database.connection.MySQLConnectionPool')
    def test_test_connection_success(self, mock_pool, mock_settings):
        """Test successful connection test."""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (1,)
        
        mock_pool_instance = Mock()
        mock_pool_instance.get_connection.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance
        
        db = DatabaseConnection(mock_settings)
        
        with patch.object(db, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_connection
            
            result = db.test_connection()
        
        assert result is True
        mock_cursor.execute.assert_called_with("SELECT 1")
    
    @patch('src.database.connection.MySQLConnectionPool')
    def test_get_table_names(self, mock_pool, mock_settings):
        """Test getting table names."""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [('users',), ('products',), ('orders',)]
        
        mock_pool_instance = Mock()
        mock_pool_instance.get_connection.return_value = mock_connection
        mock_pool.return_value = mock_pool_instance
        
        db = DatabaseConnection(mock_settings)
        
        with patch.object(db, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_connection
            
            tables = db.get_table_names()
        
        assert tables == ['users', 'products', 'orders']
        mock_cursor.execute.assert_called_with("SHOW TABLES")


class TestQueryResult:
    """Test cases for QueryResult model."""
    
    def test_query_result_success(self):
        """Test successful QueryResult."""
        result = QueryResult(
            columns=['id', 'name'],
            rows=[[1, 'test']],
            row_count=1,
            execution_time=0.5,
            query="SELECT * FROM test",
            timestamp=datetime.now()
        )
        
        assert result.is_success
        assert result.columns == ['id', 'name']
        assert result.row_count == 1
    
    def test_query_result_failure(self):
        """Test failed QueryResult."""
        result = QueryResult(
            columns=[],
            rows=[],
            row_count=0,
            execution_time=0.1,
            query="SELECT * FROM nonexistent",
            timestamp=datetime.now(),
            error="Table doesn't exist"
        )
        
        assert not result.is_success
        assert result.error == "Table doesn't exist"
    
    def test_query_result_to_dict(self):
        """Test QueryResult to_dict method."""
        timestamp = datetime.now()
        result = QueryResult(
            columns=['id'],
            rows=[[1]],
            row_count=1,
            execution_time=0.5,
            query="SELECT * FROM test",
            timestamp=timestamp
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['columns'] == ['id']
        assert result_dict['rows'] == [[1]]
        assert result_dict['row_count'] == 1
        assert result_dict['execution_time'] == 0.5
        assert result_dict['query'] == "SELECT * FROM test"
        assert result_dict['timestamp'] == timestamp.isoformat()
        assert result_dict['success'] is True
        assert result_dict['error'] is None


class TestTableSchema:
    """Test cases for TableSchema model."""
    
    def test_table_schema_creation(self):
        """Test TableSchema creation."""
        columns = [
            {'name': 'id', 'type': 'INT', 'nullable': False, 'key': 'PRI', 'default': None, 'extra': 'auto_increment'},
            {'name': 'name', 'type': 'VARCHAR(255)', 'nullable': False, 'key': '', 'default': None, 'extra': ''}
        ]
        
        schema = TableSchema(
            name='users',
            columns=columns,
            primary_keys=['id'],
            foreign_keys=[]
        )
        
        assert schema.name == 'users'
        assert len(schema.columns) == 2
        assert schema.primary_keys == ['id']
        assert schema.foreign_keys == []
    
    def test_table_schema_to_dict(self):
        """Test TableSchema to_dict method."""
        columns = [{'name': 'id', 'type': 'INT', 'nullable': False, 'key': 'PRI', 'default': None, 'extra': ''}]
        
        schema = TableSchema(
            name='test_table',
            columns=columns,
            primary_keys=['id'],
            foreign_keys=[]
        )
        
        schema_dict = schema.to_dict()
        
        assert schema_dict['name'] == 'test_table'
        assert schema_dict['columns'] == columns
        assert schema_dict['primary_keys'] == ['id']
        assert schema_dict['foreign_keys'] == []


class TestDatabaseSchema:
    """Test cases for DatabaseSchema model."""
    
    def test_database_schema_creation(self):
        """Test DatabaseSchema creation."""
        table1 = TableSchema(name='users', columns=[], primary_keys=[], foreign_keys=[])
        table2 = TableSchema(name='orders', columns=[], primary_keys=[], foreign_keys=[])
        
        schema = DatabaseSchema(
            tables=[table1, table2],
            database_name='test_db'
        )
        
        assert schema.database_name == 'test_db'
        assert len(schema.tables) == 2
    
    def test_get_table(self):
        """Test getting table by name."""
        table1 = TableSchema(name='users', columns=[], primary_keys=[], foreign_keys=[])
        table2 = TableSchema(name='orders', columns=[], primary_keys=[], foreign_keys=[])
        
        schema = DatabaseSchema(
            tables=[table1, table2],
            database_name='test_db'
        )
        
        found_table = schema.get_table('users')
        assert found_table is not None
        assert found_table.name == 'users'
        
        # Test case insensitive
        found_table = schema.get_table('USERS')
        assert found_table is not None
        assert found_table.name == 'users'
        
        # Test non-existent table
        found_table = schema.get_table('nonexistent')
        assert found_table is None