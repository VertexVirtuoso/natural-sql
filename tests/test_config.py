"""Tests for configuration module."""

import pytest
import os
from unittest.mock import patch, Mock

from src.config.settings import Settings, get_settings


class TestSettings:
    """Test cases for Settings class."""
    
    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()
        
        assert settings.db_host == "localhost"
        assert settings.db_port == 3306
        assert settings.db_name == "natural_sql_db"
        assert settings.db_user == "root"
        assert settings.db_password == ""
        assert settings.db_pool_size == 5
        assert settings.db_max_overflow == 10
        assert settings.db_pool_timeout == 30
        assert settings.openai_api_key is None
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.default_output_format == "table"
        assert settings.query_history_size == 100
        assert settings.secret_key == "dev-secret-key"
        assert settings.max_query_length == 1000
        assert settings.query_timeout == 30
    
    def test_settings_from_env_vars(self):
        """Test settings loaded from environment variables."""
        env_vars = {
            'DB_HOST': 'prod-db.example.com',
            'DB_PORT': '5432',
            'DB_NAME': 'production_db',
            'DB_USER': 'prod_user',
            'DB_PASSWORD': 'secure_password',
            'DB_POOL_SIZE': '10',
            'DB_MAX_OVERFLOW': '20',
            'DB_POOL_TIMEOUT': '60',
            'OPENAI_API_KEY': 'sk-test-key-12345',
            'DEBUG': 'true',
            'LOG_LEVEL': 'DEBUG',
            'DEFAULT_OUTPUT_FORMAT': 'json',
            'QUERY_HISTORY_SIZE': '200',
            'SECRET_KEY': 'production-secret-key',
            'MAX_QUERY_LENGTH': '2000',
            'QUERY_TIMEOUT': '60'
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
        
        assert settings.db_host == 'prod-db.example.com'
        assert settings.db_port == 5432
        assert settings.db_name == 'production_db'
        assert settings.db_user == 'prod_user'
        assert settings.db_password == 'secure_password'
        assert settings.db_pool_size == 10
        assert settings.db_max_overflow == 20
        assert settings.db_pool_timeout == 60
        assert settings.openai_api_key == 'sk-test-key-12345'
        assert settings.debug is True
        assert settings.log_level == 'DEBUG'
        assert settings.default_output_format == 'json'
        assert settings.query_history_size == 200
        assert settings.secret_key == 'production-secret-key'
        assert settings.max_query_length == 2000
        assert settings.query_timeout == 60
    
    def test_database_url_property(self):
        """Test database URL generation."""
        settings = Settings()
        settings.db_user = 'testuser'
        settings.db_password = 'testpass'
        settings.db_host = 'testhost'
        settings.db_port = 3306
        settings.db_name = 'testdb'
        
        expected_url = 'mysql+mysqlconnector://testuser:testpass@testhost:3306/testdb'
        assert settings.database_url == expected_url
    
    def test_database_url_with_empty_password(self):
        """Test database URL generation with empty password."""
        settings = Settings()
        settings.db_user = 'testuser'
        settings.db_password = ''
        settings.db_host = 'testhost'
        settings.db_port = 3306
        settings.db_name = 'testdb'
        
        expected_url = 'mysql+mysqlconnector://testuser:@testhost:3306/testdb'
        assert settings.database_url == expected_url
    
    def test_settings_with_env_file(self):
        """Test settings loading from .env file."""
        # This test would require creating a temporary .env file
        # For now, we'll test that the Config class is properly set
        settings = Settings()
        
        assert settings.Config.env_file == '.env'
        assert settings.Config.env_file_encoding == 'utf-8'
    
    def test_boolean_env_var_parsing(self):
        """Test boolean environment variable parsing."""
        # Test various boolean representations
        boolean_tests = {
            'true': True,
            'True': True,
            'TRUE': True,
            '1': True,
            'yes': True,
            'false': False,
            'False': False,
            'FALSE': False,
            '0': False,
            'no': False
        }
        
        for env_value, expected in boolean_tests.items():
            with patch.dict(os.environ, {'DEBUG': env_value}):
                settings = Settings()
                assert settings.debug == expected
    
    def test_integer_env_var_parsing(self):
        """Test integer environment variable parsing."""
        with patch.dict(os.environ, {'DB_PORT': '5432', 'QUERY_TIMEOUT': '120'}):
            settings = Settings()
            
            assert isinstance(settings.db_port, int)
            assert settings.db_port == 5432
            assert isinstance(settings.query_timeout, int)
            assert settings.query_timeout == 120
    
    def test_optional_env_vars(self):
        """Test optional environment variables."""
        # Test with no optional env vars set
        settings = Settings()
        
        assert settings.openai_api_key is None
        assert settings.anthropic_api_key is None
        assert settings.huggingface_api_token is None
        
        # Test with optional env vars set
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'ant-key-123',
            'HUGGINGFACE_API_TOKEN': 'hf-token-456'
        }):
            settings = Settings()
            
            assert settings.anthropic_api_key == 'ant-key-123'
            assert settings.huggingface_api_token == 'hf-token-456'


class TestGetSettings:
    """Test cases for get_settings function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset the global settings instance before each test
        import src.config.settings
        src.config.settings._settings = None
    
    def teardown_method(self):
        """Clean up after each test."""
        # Reset the global settings instance after each test
        import src.config.settings
        src.config.settings._settings = None
    
    def test_get_settings_singleton(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2
        assert isinstance(settings1, Settings)
    
    def test_get_settings_creates_new_instance_if_none(self):
        """Test that get_settings creates new instance if none exists."""
        # Ensure no global instance exists
        import src.config.settings
        src.config.settings._settings = None
        
        settings = get_settings()
        
        assert settings is not None
        assert isinstance(settings, Settings)
        
        # Check that the global instance was set
        assert src.config.settings._settings is settings
    
    def test_get_settings_with_env_changes(self):
        """Test get_settings behavior with environment changes."""
        # First call with default env
        settings1 = get_settings()
        original_host = settings1.db_host
        
        # The singleton should return the same instance even if env changes
        with patch.dict(os.environ, {'DB_HOST': 'changed-host'}):
            settings2 = get_settings()
            
            # Should be the same instance
            assert settings1 is settings2
            # Should still have the original value (singleton behavior)
            assert settings2.db_host == original_host
    
    @patch('src.config.settings.Settings')
    def test_get_settings_exception_handling(self, mock_settings_class):
        """Test get_settings behavior when Settings creation fails."""
        # Make Settings constructor raise an exception
        mock_settings_class.side_effect = Exception("Configuration error")
        
        with pytest.raises(Exception, match="Configuration error"):
            get_settings()
    
    def test_settings_field_validation(self):
        """Test settings field validation."""
        # Test that invalid port raises error
        with patch.dict(os.environ, {'DB_PORT': 'invalid_port'}):
            with pytest.raises(Exception):  # Should raise validation error
                Settings()
        
        # Test that negative numbers work where expected
        with patch.dict(os.environ, {'QUERY_TIMEOUT': '-1'}):
            settings = Settings()
            assert settings.query_timeout == -1
    
    def test_settings_immutability(self):
        """Test that settings can be modified after creation."""
        settings = Settings()
        original_host = settings.db_host
        
        # Settings should be mutable (not frozen)
        settings.db_host = "new-host"
        assert settings.db_host == "new-host"
        assert settings.db_host != original_host