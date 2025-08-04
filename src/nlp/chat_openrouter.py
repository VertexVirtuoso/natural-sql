"""ChatOpenRouter wrapper for LangChain integration with OpenRouter API."""

import os
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.utils.utils import secret_from_env
from pydantic import Field, SecretStr


class ChatOpenRouter(ChatOpenAI):
    """LangChain-compatible wrapper for OpenRouter API using ChatOpenAI."""
    
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=lambda: secret_from_env("OPENROUTER_API_KEY", default=None),
    )
    
    @property
    def lc_secrets(self) -> Dict[str, str]:
        """Return secrets for LangChain serialization."""
        return {"openai_api_key": "OPENROUTER_API_KEY"}
    
    def __init__(
        self,
        model: str = "qwen/qwen3-coder:free",
        openai_api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        **kwargs
    ):
        """Initialize ChatOpenRouter with OpenRouter-specific configuration.
        
        Args:
            model: The OpenRouter model to use (default: qwen/qwen3-coder:free)
            openai_api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            site_url: Site URL for OpenRouter rankings (optional)
            site_name: Site name for OpenRouter rankings (optional)
            **kwargs: Additional arguments passed to ChatOpenAI
        """
        # Get API key from parameter or environment
        api_key = openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required. Please set it in your environment "
                "or pass it as openai_api_key parameter."
            )
        
        # Set up default headers for OpenRouter
        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if site_name:
            default_headers["X-Title"] = site_name
        
        # Merge with any existing headers
        if "default_headers" in kwargs:
            default_headers.update(kwargs["default_headers"])
        kwargs["default_headers"] = default_headers
        
        # Initialize ChatOpenAI with OpenRouter base URL
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model,
            temperature=kwargs.get("temperature", 0),
            **kwargs
        )
    
    @classmethod
    def from_settings(cls, settings, **kwargs):
        """Create ChatOpenRouter instance from application settings.
        
        Args:
            settings: Application settings object with OpenRouter configuration
            **kwargs: Additional arguments to override settings
        """
        return cls(
            model=kwargs.get("model", settings.openrouter_model),
            openai_api_key=kwargs.get("openai_api_key", settings.openrouter_api_key),
            site_url=kwargs.get("site_url", settings.site_url),
            site_name=kwargs.get("site_name", settings.site_name),
            **{k: v for k, v in kwargs.items() if k not in ["model", "openai_api_key", "site_url", "site_name"]}
        )