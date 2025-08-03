"""Pytest configuration and shared fixtures."""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch
from datetime import datetime

from src.config.settings import Settings
from src.database.models import QueryResult, TableSchema, DatabaseSchema


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("""
DB_HOST=test_host
DB_PORT=3306
DB_NAME=test_db
DB_USER=test_user
DB_PASSWORD=test_password
OPENAI_API_KEY=test-api-key
DEBUG=true
""")
        temp_file_path = f.name
    
    yield temp_file_path
    
    # Cleanup
    os.unlink(temp_file_path)


@pytest.fixture
def mock_settings():
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
    settings.openai_api_key = "test-api-key"
    settings.debug = False
    settings.log_level = "INFO"
    settings.default_output_format = "table"
    settings.query_history_size = 100
    settings.secret_key = "test-secret-key"
    settings.max_query_length = 1000
    settings.query_timeout = 30
    settings.database_url = "mysql+mysqlconnector://test_user:test_password@localhost:3306/test_db"
    return settings


@pytest.fixture
def sample_query_result():
    """Create a sample successful query result."""
    return QueryResult(
        columns=['id', 'name', 'email'],
        rows=[
            [1, 'John Doe', 'john@example.com'],
            [2, 'Jane Smith', 'jane@example.com'],
            [3, 'Bob Johnson', 'bob@example.com']
        ],
        row_count=3,
        execution_time=0.123,
        query="SELECT id, name, email FROM users LIMIT 3;",
        timestamp=datetime(2025, 1, 15, 12, 0, 0)
    )


@pytest.fixture
def sample_error_result():
    """Create a sample error query result."""
    return QueryResult(
        columns=[],
        rows=[],
        row_count=0,
        execution_time=0.045,
        query="SELECT * FROM nonexistent_table;",
        timestamp=datetime(2025, 1, 15, 12, 0, 0),
        error="Table 'test_db.nonexistent_table' doesn't exist"
    )


@pytest.fixture
def sample_table_schema():
    """Create a sample table schema."""
    columns = [
        {
            'name': 'id',
            'type': 'INT',
            'nullable': False,
            'key': 'PRI',
            'default': None,
            'extra': 'auto_increment'
        },
        {
            'name': 'name',
            'type': 'VARCHAR(255)',
            'nullable': False,
            'key': '',
            'default': None,
            'extra': ''
        },
        {
            'name': 'email',
            'type': 'VARCHAR(255)',
            'nullable': True,
            'key': '',
            'default': None,
            'extra': ''
        },
        {
            'name': 'created_at',
            'type': 'TIMESTAMP',
            'nullable': False,
            'key': '',
            'default': 'CURRENT_TIMESTAMP',
            'extra': ''
        }
    ]
    
    return TableSchema(
        name='users',
        columns=columns,
        primary_keys=['id'],
        foreign_keys=[]
    )


@pytest.fixture
def sample_database_schema(sample_table_schema):
    """Create a sample database schema with multiple tables."""
    # Users table (from sample_table_schema)
    users_table = sample_table_schema
    
    # Products table
    products_columns = [
        {
            'name': 'id',
            'type': 'INT',
            'nullable': False,
            'key': 'PRI',
            'default': None,
            'extra': 'auto_increment'
        },
        {
            'name': 'name',
            'type': 'VARCHAR(255)',
            'nullable': False,
            'key': '',
            'default': None,
            'extra': ''
        },
        {
            'name': 'price',
            'type': 'DECIMAL(10,2)',
            'nullable': False,
            'key': '',
            'default': None,
            'extra': ''
        },
        {
            'name': 'category_id',
            'type': 'INT',
            'nullable': True,
            'key': 'MUL',
            'default': None,
            'extra': ''
        }
    ]
    
    products_table = TableSchema(
        name='products',
        columns=products_columns,
        primary_keys=['id'],
        foreign_keys=[
            {
                'column': 'category_id',
                'referenced_table': 'categories',
                'referenced_column': 'id'
            }
        ]
    )
    
    # Orders table
    orders_columns = [
        {
            'name': 'id',
            'type': 'INT',
            'nullable': False,
            'key': 'PRI',
            'default': None,
            'extra': 'auto_increment'
        },
        {
            'name': 'user_id',
            'type': 'INT',
            'nullable': False,
            'key': 'MUL',
            'default': None,
            'extra': ''
        },
        {
            'name': 'total_amount',
            'type': 'DECIMAL(10,2)',
            'nullable': False,
            'key': '',
            'default': None,
            'extra': ''
        },
        {
            'name': 'order_date',
            'type': 'TIMESTAMP',
            'nullable': False,
            'key': '',
            'default': 'CURRENT_TIMESTAMP',
            'extra': ''
        }
    ]
    
    orders_table = TableSchema(
        name='orders',
        columns=orders_columns,
        primary_keys=['id'],
        foreign_keys=[
            {
                'column': 'user_id',
                'referenced_table': 'users',
                'referenced_column': 'id'
            }
        ]
    )
    
    return DatabaseSchema(
        tables=[users_table, products_table, orders_table],
        database_name='test_ecommerce_db'
    )


@pytest.fixture
def mock_database_connection():
    """Create a mock database connection."""
    mock_conn = Mock()
    mock_conn.test_connection.return_value = True
    mock_conn.get_table_names.return_value = ['users', 'products', 'orders']
    return mock_conn


@pytest.fixture
def mock_query_processor():
    """Create a mock query processor."""
    mock_processor = Mock()
    mock_processor.process_natural_language_query.return_value = QueryResult(
        columns=['count'],
        rows=[[42]],
        row_count=1,
        execution_time=0.1,
        query="SELECT COUNT(*) FROM users;",
        timestamp=datetime.now()
    )
    mock_processor.explain_query_results.return_value = "This query counts all users in the database."
    mock_processor.get_query_suggestions.return_value = [
        "Show me all users",
        "Count total products",
        "Find recent orders"
    ]
    return mock_processor


@pytest.fixture(autouse=True)
def reset_global_instances():
    """Reset global instances before each test."""
    # Reset settings singleton
    import src.config.settings
    src.config.settings._settings = None
    
    # Reset database connection singleton
    import src.database.connection
    src.database.connection._db_connection = None
    
    yield
    
    # Clean up after test
    src.config.settings._settings = None
    src.database.connection._db_connection = None


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response."""
    mock_response = Mock()
    mock_response.content = "SELECT * FROM users;"
    return mock_response


@pytest.fixture
def mock_mysql_connection():
    """Create a mock MySQL connection and cursor."""
    mock_cursor = Mock()
    mock_cursor.description = [('id',), ('name',), ('email',)]
    mock_cursor.fetchall.return_value = [
        (1, 'John Doe', 'john@example.com'),
        (2, 'Jane Smith', 'jane@example.com')
    ]
    
    mock_connection = Mock()
    mock_connection.cursor.return_value = mock_cursor
    mock_connection.is_connected.return_value = True
    
    return mock_connection, mock_cursor


# Test data constants
TEST_DATABASE_NAME = "test_natural_sql_db"
TEST_API_KEY = "test-openai-api-key-12345"
TEST_SCHEMA_SUMMARY = """Database: test_ecommerce_db

Table: users
  - id: INT PRIMARY KEY (auto_increment)
  - name: VARCHAR(255) NOT NULL
  - email: VARCHAR(255)
  - created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP

Table: products
  - id: INT PRIMARY KEY (auto_increment)
  - name: VARCHAR(255) NOT NULL
  - price: DECIMAL(10,2) NOT NULL
  - category_id: INT
  Foreign Keys:
    - category_id -> categories.id

Table: orders
  - id: INT PRIMARY KEY (auto_increment)
  - user_id: INT NOT NULL
  - total_amount: DECIMAL(10,2) NOT NULL
  - order_date: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  Foreign Keys:
    - user_id -> users.id"""


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires database)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring API key"
    )