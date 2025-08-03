"""Tests for CLI interface."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner
from datetime import datetime

from src.cli.main import app, initialize_components, display_query_result
from src.database.models import QueryResult


class TestCLI:
    """Test cases for CLI interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @patch('src.cli.main.get_settings')
    @patch('src.cli.main.get_db_connection')
    @patch('src.cli.main.QueryProcessor')
    def test_initialize_components_success(self, mock_query_processor, mock_get_db_connection, mock_get_settings):
        """Test successful component initialization."""
        # Setup mocks
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        
        mock_db_conn = Mock()
        mock_db_conn.test_connection.return_value = True
        mock_get_db_connection.return_value = mock_db_conn
        
        mock_processor = Mock()
        mock_query_processor.return_value = mock_processor
        
        result = initialize_components()
        
        assert result is True
        mock_db_conn.test_connection.assert_called_once()
        mock_query_processor.assert_called_once_with(mock_settings, mock_db_conn)
    
    @patch('src.cli.main.get_settings')
    @patch('src.cli.main.get_db_connection')
    def test_initialize_components_db_failure(self, mock_get_db_connection, mock_get_settings):
        """Test component initialization with database failure."""
        # Setup mocks
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        
        mock_db_conn = Mock()
        mock_db_conn.test_connection.return_value = False
        mock_get_db_connection.return_value = mock_db_conn
        
        result = initialize_components()
        
        assert result is False
    
    @patch('src.cli.main.get_settings')
    @patch('src.cli.main.get_db_connection')
    def test_initialize_components_exception(self, mock_get_db_connection, mock_get_settings):
        """Test component initialization with exception."""
        # Setup mocks to raise exception
        mock_get_settings.side_effect = Exception("Configuration error")
        
        result = initialize_components()
        
        assert result is False
    
    def test_display_query_result_success_table(self, capsys):
        """Test displaying successful query results in table format."""
        result = QueryResult(
            columns=['id', 'name', 'email'],
            rows=[[1, 'John', 'john@example.com'], [2, 'Jane', 'jane@example.com']],
            row_count=2,
            execution_time=0.123,
            query="SELECT * FROM users;",
            timestamp=datetime.now()
        )
        
        with patch('src.cli.main.console') as mock_console:
            display_query_result(result, "table")
            
            # Check that table was created and printed
            assert mock_console.print.called
            # Check execution time was displayed
            mock_console.print.assert_any_call("Executed in 0.123 seconds")
    
    def test_display_query_result_success_json(self, capsys):
        """Test displaying successful query results in JSON format."""
        result = QueryResult(
            columns=['id', 'name'],
            rows=[[1, 'John']],
            row_count=1,
            execution_time=0.123,
            query="SELECT * FROM users;",
            timestamp=datetime.now()
        )
        
        with patch('src.cli.main.console') as mock_console:
            display_query_result(result, "json")
            
            # Should print JSON and execution time
            assert mock_console.print.call_count >= 2
    
    def test_display_query_result_success_csv(self, capsys):
        """Test displaying successful query results in CSV format."""
        result = QueryResult(
            columns=['id', 'name'],
            rows=[[1, 'John']],
            row_count=1,
            execution_time=0.123,
            query="SELECT * FROM users;",
            timestamp=datetime.now()
        )
        
        with patch('src.cli.main.console') as mock_console:
            display_query_result(result, "csv")
            
            # Should print CSV and execution time
            assert mock_console.print.call_count >= 2
    
    def test_display_query_result_failure(self, capsys):
        """Test displaying failed query results."""
        result = QueryResult(
            columns=[],
            rows=[],
            row_count=0,
            execution_time=0.123,
            query="SELECT * FROM nonexistent;",
            timestamp=datetime.now(),
            error="Table 'nonexistent' doesn't exist"
        )
        
        with patch('src.cli.main.console') as mock_console:
            display_query_result(result, "table")
            
            # Should print error message
            mock_console.print.assert_called_with("Query failed: Table 'nonexistent' doesn't exist")
    
    def test_display_query_result_no_results(self, capsys):
        """Test displaying query results with no rows."""
        result = QueryResult(
            columns=['id', 'name'],
            rows=[],
            row_count=0,
            execution_time=0.123,
            query="SELECT * FROM users WHERE id = 999;",
            timestamp=datetime.now()
        )
        
        with patch('src.cli.main.console') as mock_console:
            display_query_result(result, "table")
            
            # Should print no results message
            mock_console.print.assert_called_with("No results found.")
    
    @patch('src.cli.main.initialize_components')
    def test_version_command(self, mock_init):
        """Test version command."""
        result = self.runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "Natural SQL CLI" in result.stdout
    
    @patch('src.cli.main.initialize_components')
    def test_test_connection_command_success(self, mock_init):
        """Test successful connection test command."""
        mock_init.return_value = True
        
        with patch('src.cli.main.db_connection') as mock_db_conn:
            mock_db_conn.get_table_names.return_value = ['users', 'products']
            
            result = self.runner.invoke(app, ["test-connection"])
        
        assert result.exit_code == 0
        mock_init.assert_called_once()
    
    @patch('src.cli.main.initialize_components')
    def test_test_connection_command_failure(self, mock_init):
        """Test failed connection test command."""
        mock_init.return_value = False
        
        result = self.runner.invoke(app, ["test-connection"])
        
        assert result.exit_code == 1
        mock_init.assert_called_once()
    
    @patch('src.cli.main.initialize_components')
    @patch('src.cli.main.query_processor')
    def test_query_command_with_question(self, mock_processor, mock_init):
        """Test query command with question argument."""
        mock_init.return_value = True
        
        mock_result = QueryResult(
            columns=['count'],
            rows=[[5]],
            row_count=1,
            execution_time=0.1,
            query="SELECT COUNT(*) FROM users;",
            timestamp=datetime.now()
        )
        mock_processor.process_natural_language_query.return_value = mock_result
        
        with patch('src.cli.main.display_query_result') as mock_display:
            result = self.runner.invoke(app, ["query", "How many users are there?"])
        
        assert result.exit_code == 0
        mock_init.assert_called_once()
        mock_processor.process_natural_language_query.assert_called_once_with("How many users are there?")
        mock_display.assert_called_once()
    
    @patch('src.cli.main.initialize_components')
    def test_query_command_initialization_failure(self, mock_init):
        """Test query command with initialization failure."""
        mock_init.return_value = False
        
        result = self.runner.invoke(app, ["query", "test question"])
        
        assert result.exit_code == 1
        mock_init.assert_called_once()
    
    @patch('src.cli.main.initialize_components')
    @patch('src.cli.main.query_processor')
    def test_query_command_with_explain(self, mock_processor, mock_init):
        """Test query command with explain option."""
        mock_init.return_value = True
        
        mock_result = QueryResult(
            columns=['count'],
            rows=[[5]],
            row_count=1,
            execution_time=0.1,
            query="SELECT COUNT(*) FROM users;",
            timestamp=datetime.now()
        )
        mock_processor.process_natural_language_query.return_value = mock_result
        mock_processor.explain_query_results.return_value = "This query counts all users."
        
        with patch('src.cli.main.display_query_result'):
            result = self.runner.invoke(app, ["query", "--explain", "How many users?"])
        
        assert result.exit_code == 0
        mock_processor.explain_query_results.assert_called_once()
    
    @patch('src.cli.main.initialize_components')
    @patch('src.cli.main.query_processor')
    def test_query_command_different_formats(self, mock_processor, mock_init):
        """Test query command with different output formats."""
        mock_init.return_value = True
        
        mock_result = QueryResult(
            columns=['id', 'name'],
            rows=[[1, 'John']],
            row_count=1,
            execution_time=0.1,
            query="SELECT id, name FROM users;",
            timestamp=datetime.now()
        )
        mock_processor.process_natural_language_query.return_value = mock_result
        
        formats = ['table', 'json', 'csv']
        
        for format_type in formats:
            with patch('src.cli.main.display_query_result') as mock_display:
                result = self.runner.invoke(app, ["query", "--format", format_type, "Show users"])
                
                assert result.exit_code == 0
                mock_display.assert_called_once()
                # Check that the correct format was passed
                call_args = mock_display.call_args
                assert call_args[0][1] == format_type  # Second argument should be format


class TestCLIHelpers:
    """Test cases for CLI helper functions."""
    
    def test_query_result_large_dataset_display(self):
        """Test displaying query results with large datasets."""
        # Create a result with more than 50 rows
        rows = [[i, f'user{i}'] for i in range(100)]
        result = QueryResult(
            columns=['id', 'name'],
            rows=rows,
            row_count=100,
            execution_time=0.5,
            query="SELECT id, name FROM users;",
            timestamp=datetime.now()
        )
        
        with patch('src.cli.main.console') as mock_console:
            display_query_result(result, "table")
            
            # Should create table and show truncation message
            assert mock_console.print.call_count >= 2
            # Check for truncation message by looking at call arguments
            printed_messages = [call.args[0] for call in mock_console.print.call_args_list]
            truncation_messages = [msg for msg in printed_messages if "Showing first 50 of 100 results" in str(msg)]
            assert len(truncation_messages) > 0