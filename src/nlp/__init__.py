"""Natural Language Processing module for SQL generation."""

from .query_processor import QueryProcessor
from .prompts import SQL_GENERATION_PROMPT

__all__ = ["QueryProcessor", "SQL_GENERATION_PROMPT"]