"""Configuration settings for the Natural SQL application."""

import os
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database Configuration
    db_host: str = Field(default="localhost", env="DB_HOST")
    db_port: int = Field(default=3306, env="DB_PORT")
    db_name: str = Field(default="natural_sql_db", env="DB_NAME")
    db_user: str = Field(default="root", env="DB_USER")
    db_password: str = Field(default="", env="DB_PASSWORD")
    
    # Connection Pool Settings
    db_pool_size: int = Field(default=5, env="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, env="DB_MAX_OVERFLOW")
    db_pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    
    # OpenAI Configuration (deprecated - use OpenRouter instead)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # OpenRouter Configuration
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="qwen/qwen3-coder:free", env="OPENROUTER_MODEL")
    site_url: Optional[str] = Field(default=None, env="SITE_URL")
    site_name: Optional[str] = Field(default="Natural SQL CLI", env="SITE_NAME")
    
    # Alternative LLM Configuration
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    huggingface_api_token: Optional[str] = Field(default=None, env="HUGGINGFACE_API_TOKEN")
    
    # Application Configuration
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # CLI Configuration
    default_output_format: str = Field(default="table", env="DEFAULT_OUTPUT_FORMAT")
    query_history_size: int = Field(default=100, env="QUERY_HISTORY_SIZE")
    
    # Security
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    
    # Query Processing Settings
    max_query_length: int = Field(default=1000, env="MAX_QUERY_LENGTH")
    query_timeout: int = Field(default=30, env="QUERY_TIMEOUT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @property
    def database_url(self) -> str:
        """Generate database URL for SQLAlchemy."""
        return f"mysql+mysqlconnector://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings