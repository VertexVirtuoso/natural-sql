"""Tests for NLP query processing module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.nlp.query_processor import QueryProcessor
from src.nlp.prompts import SQL_GENERATION_PROMPT, SCHEMA_SUMMARY_PROMPT
from src.database.models import QueryResult, DatabaseSchema, TableSchema
from src.config.settings import Settings


class TestQueryProcessor:
    """Test cases for QueryProcessor class."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock(spec=Settings)
        settings.openrouter_api_key = "test-openrouter-api-key"
        settings.openrouter_model = "qwen/qwen3-coder:free"
        settings.site_url = "https://test.com"
        settings.site_name = "Test App"
        settings.openai_api_key = None
        return settings
    
    @pytest.fixture
    def mock_db_connection(self):
        """Create mock database connection."""
        db_conn = Mock()
        db_conn.test_connection.return_value = True
        return db_conn
    
    @pytest.fixture
    def sample_schema(self):
        """Create sample database schema."""
        columns = [
            {'name': 'id', 'type': 'INT', 'nullable': False, 'key': 'PRI', 'default': None, 'extra': 'auto_increment'},
            {'name': 'name', 'type': 'VARCHAR(255)', 'nullable': False, 'key': '', 'default': None, 'extra': ''},
            {'name': 'email', 'type': 'VARCHAR(255)', 'nullable': True, 'key': '', 'default': None, 'extra': ''}
        ]
        
        table = TableSchema(
            name='users',
            columns=columns,
            primary_keys=['id'],
            foreign_keys=[]
        )
        
        return DatabaseSchema(
            tables=[table],
            database_name='test_db'
        )
    
    @patch('src.nlp.query_processor.OpenAI')
    def test_setup_llm_success(self, mock_openai, mock_settings, mock_db_connection):
        """Test successful LLM setup with OpenRouter."""
        processor = QueryProcessor(mock_settings, mock_db_connection)
        
        mock_openai.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1",
            api_key="test-openrouter-api-key"
        )
    
    def test_setup_llm_no_api_key(self, mock_db_connection):
        """Test LLM setup without API key."""
        settings = Mock(spec=Settings)
        settings.openrouter_api_key = None
        settings.openai_api_key = None
        
        with pytest.raises(ValueError, match="Either OPENROUTER_API_KEY or OPENAI_API_KEY is required"):
            QueryProcessor(settings, mock_db_connection)
    
    @patch('src.nlp.query_processor.OpenAI')
    def test_format_schema_for_llm(self, mock_openai, mock_settings, mock_db_connection, sample_schema):
        """Test schema formatting for LLM."""
        processor = QueryProcessor(mock_settings, mock_db_connection)
        
        formatted_schema = processor._format_schema_for_llm(sample_schema)
        
        assert "Database: test_db" in formatted_schema
        assert "Table: users" in formatted_schema
        assert "id: INT" in formatted_schema
        assert "PRIMARY KEY" in formatted_schema
        assert "name: VARCHAR(255)" in formatted_schema
    
    @patch('src.nlp.query_processor.OpenAI')
    def test_validate_query_safe(self, mock_openai, mock_settings, mock_db_connection):
        """Test query validation for safe queries."""
        processor = QueryProcessor(mock_settings, mock_db_connection)
        
        safe_queries = [
            "SELECT * FROM users;",
            "SELECT name, email FROM users WHERE id = 1;",
            "SELECT COUNT(*) FROM users;",
            "select id from users limit 10;"
        ]
        
        for query in safe_queries:
            assert processor._validate_query(query) is True
    
    @patch('src.nlp.query_processor.OpenAI')
    def test_validate_query_unsafe(self, mock_openai, mock_settings, mock_db_connection):
        """Test query validation for unsafe queries."""
        processor = QueryProcessor(mock_settings, mock_db_connection)
        
        unsafe_queries = [
            "DROP TABLE users;",
            "DELETE FROM users;",
            "INSERT INTO users VALUES (1, 'test');",
            "UPDATE users SET name = 'test';",
            "ALTER TABLE users ADD COLUMN test VARCHAR(255);",
            "CREATE TABLE test (id INT);",
            "TRUNCATE TABLE users;",
        ]
        
        for query in unsafe_queries:
            assert processor._validate_query(query) is False
    
    @patch('src.nlp.query_processor.OpenAI')
    def test_validate_query_non_select(self, mock_openai, mock_settings, mock_db_connection):
        """Test query validation for non-SELECT queries."""
        processor = QueryProcessor(mock_settings, mock_db_connection)
        
        non_select_query = "SHOW TABLES;"
        assert processor._validate_query(non_select_query) is False
    
    @patch('src.nlp.query_processor.OpenAI')
    def test_sanitize_query(self, mock_openai, mock_settings, mock_db_connection):
        """Test query sanitization."""
        processor = QueryProcessor(mock_settings, mock_db_connection)
        
        # Test comment removal
        query_with_comments = "SELECT * FROM users; -- This is a comment"
        sanitized = processor._sanitize_query(query_with_comments)
        assert "--" not in sanitized
        
        # Test block comment removal
        query_with_block_comments = "SELECT * FROM users /* block comment */;"
        sanitized = processor._sanitize_query(query_with_block_comments)
        assert "/*" not in sanitized and "*/" not in sanitized
        
        # Test semicolon addition
        query_without_semicolon = "SELECT * FROM users"
        sanitized = processor._sanitize_query(query_without_semicolon)
        assert sanitized.endswith(';')
    
    @patch('src.nlp.query_processor.OpenAI')
    def test_process_natural_language_query_success(self, mock_openai, mock_settings, mock_db_connection, sample_schema):
        """Test successful natural language query processing."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "SELECT * FROM users;"
        mock_llm_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_llm_instance
        
        mock_db_connection.get_database_schema.return_value = sample_schema
        
        successful_result = QueryResult(
            columns=['id', 'name', 'email'],
            rows=[[1, 'John', 'john@example.com']],
            row_count=1,
            execution_time=0.1,
            query="SELECT * FROM users;",
            timestamp=datetime.now()
        )
        mock_db_connection.execute_query.return_value = successful_result
        
        processor = QueryProcessor(mock_settings, mock_db_connection)
        processor._schema_cache = "Database: test_db\nTable: users"
        
        result = processor.process_natural_language_query("Show me all users")
        
        assert result.is_success
        assert result.query == "SELECT * FROM users;"
        assert result.row_count == 1
        mock_db_connection.execute_query.assert_called_once()
    
    @patch('src.nlp.query_processor.OpenAI')
    def test_process_natural_language_query_unsafe(self, mock_openai, mock_settings, mock_db_connection, sample_schema):
        """Test natural language query processing with unsafe SQL generation."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "DROP TABLE users;"  # Unsafe query
        mock_llm_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_llm_instance
        
        mock_db_connection.get_database_schema.return_value = sample_schema
        
        processor = QueryProcessor(mock_settings, mock_db_connection)
        processor._schema_cache = "Database: test_db\nTable: users"
        
        result = processor.process_natural_language_query("Delete all users")
        
        assert not result.is_success
        assert "safety validation" in result.error
        mock_db_connection.execute_query.assert_not_called()
    
    @patch('src.nlp.query_processor.OpenAI')
    def test_explain_query_results(self, mock_openai, mock_settings, mock_db_connection):
        """Test query result explanation generation."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This query retrieves all user records from the users table."
        mock_llm_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_llm_instance
        
        processor = QueryProcessor(mock_settings, mock_db_connection)
        
        result = QueryResult(
            columns=['id', 'name'],
            rows=[[1, 'John']],
            row_count=1,
            execution_time=0.1,
            query="SELECT * FROM users;",
            timestamp=datetime.now()
        )
        
        explanation = processor.explain_query_results("SELECT * FROM users;", result)
        
        assert explanation == "This query retrieves all user records from the users table."
        mock_llm_instance.chat.completions.create.assert_called()
    
    @patch('src.nlp.query_processor.OpenAI')
    def test_get_query_suggestions(self, mock_openai, mock_settings, mock_db_connection, sample_schema):
        """Test query suggestions generation."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "1. Show me all users\n2. Count total users\n3. Find users by name"
        mock_llm_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_llm_instance
        
        mock_db_connection.get_database_schema.return_value = sample_schema
        
        processor = QueryProcessor(mock_settings, mock_db_connection)
        processor._schema_cache = "Database: test_db\nTable: users"
        
        suggestions = processor.get_query_suggestions("show users")
        
        assert len(suggestions) == 3
        assert "Show me all users" in suggestions
        assert "Count total users" in suggestions
        assert "Find users by name" in suggestions
    
    @patch('src.nlp.query_processor.OpenAI')
    def test_clear_cache(self, mock_openai, mock_settings, mock_db_connection):
        """Test cache clearing."""
        processor = QueryProcessor(mock_settings, mock_db_connection)
        processor._schema_cache = "cached schema"
        
        processor.clear_cache()
        
        assert processor._schema_cache is None


class TestPrompts:
    """Test cases for prompt templates."""
    
    def test_sql_generation_prompt(self):
        """Test SQL generation prompt formatting."""
        schema = "Table: users\n  - id: INT PRIMARY KEY\n  - name: VARCHAR(255)"
        question = "Show me all users"
        
        formatted_prompt = SQL_GENERATION_PROMPT.format(schema=schema, question=question)
        
        assert "Database Schema:" in formatted_prompt
        assert schema in formatted_prompt
        assert f"User Question: {question}" in formatted_prompt
        assert "Rules:" in formatted_prompt
    
    def test_schema_summary_prompt(self):
        """Test schema summary prompt formatting."""
        schema = "Table: users\n  - id: INT\n  - name: VARCHAR(255)"
        
        formatted_prompt = SCHEMA_SUMMARY_PROMPT.format(schema=schema)
        
        assert schema in formatted_prompt
        assert "Focus on:" in formatted_prompt
        assert "Table names" in formatted_prompt
    
    def test_natural_language_explanation_prompt(self):
        """Test natural language explanation prompt formatting."""
        query = "SELECT * FROM users;"
        results = "5"
        
        formatted_prompt = NATURAL_LANGUAGE_EXPLANATION_PROMPT.format(
            query=query, 
            results=results
        )
        
        assert f"SQL Query:\n{query}" in formatted_prompt
        assert f"Number of rows returned: {results}" in formatted_prompt
        assert "Provide a clear explanation" in formatted_prompt