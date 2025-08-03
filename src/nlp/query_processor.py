"""Natural language query processing using LangChain."""

import logging
import re
from typing import Optional, Dict, Any, List
from datetime import datetime

from openai import OpenAI
from langchain.schema import HumanMessage, SystemMessage

from ..config.settings import get_settings
from ..database.connection import get_db_connection
from ..database.models import DatabaseSchema, QueryResult
from .prompts import (
    SQL_GENERATION_PROMPT,
    SCHEMA_SUMMARY_PROMPT,
    QUERY_VALIDATION_PROMPT,
    NATURAL_LANGUAGE_EXPLANATION_PROMPT
)

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Processes natural language queries and converts them to SQL."""
    
    def __init__(self, settings=None, db_connection=None):
        """Initialize the query processor."""
        self.settings = settings or get_settings()
        self.db_connection = db_connection or get_db_connection()
        self._llm = None
        self._schema_cache: Optional[str] = None
        self._setup_llm()
    
    def _setup_llm(self) -> None:
        """Set up the language model using OpenRouter."""
        # Check for OpenRouter API key first, then fall back to OpenAI
        if self.settings.openrouter_api_key:
            try:
                self._llm = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.settings.openrouter_api_key
                )
                self._model_name = self.settings.openrouter_model
                self._extra_headers = {}
                
                # Add optional headers for OpenRouter rankings
                if self.settings.site_url:
                    self._extra_headers["HTTP-Referer"] = self.settings.site_url
                if self.settings.site_name:
                    self._extra_headers["X-Title"] = self.settings.site_name
                
                logger.info(f"OpenRouter LLM initialized successfully with model: {self._model_name}")
                return
            except Exception as e:
                logger.error(f"Failed to initialize OpenRouter LLM: {e}")
                raise
        
        # Fall back to direct OpenAI if no OpenRouter key
        elif self.settings.openai_api_key:
            try:
                self._llm = OpenAI(api_key=self.settings.openai_api_key)
                self._model_name = "gpt-3.5-turbo"
                self._extra_headers = {}
                logger.info("Direct OpenAI LLM initialized successfully")
                return
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI LLM: {e}")
                raise
        else:
            raise ValueError(
                "Either OPENROUTER_API_KEY or OPENAI_API_KEY is required. "
                "Please set one of these environment variables."
            )
    
    def _get_schema_summary(self) -> str:
        """Get a formatted schema summary for the LLM."""
        if self._schema_cache is not None:
            return self._schema_cache
        
        try:
            schema = self.db_connection.get_database_schema()
            schema_text = self._format_schema_for_llm(schema)
            
            # Generate a summary using LLM
            messages = [
                {"role": "system", "content": "You are a database schema analyst."},
                {"role": "user", "content": SCHEMA_SUMMARY_PROMPT.format(schema=schema_text)}
            ]
            
            response = self._llm.chat.completions.create(
                model=self._model_name,
                messages=messages,
                extra_headers=self._extra_headers,
                temperature=0,
                max_tokens=1000
            )
            self._schema_cache = response.choices[0].message.content
            
            logger.info("Schema summary generated and cached")
            return self._schema_cache
            
        except Exception as e:
            logger.error(f"Failed to generate schema summary: {e}")
            return "Schema information unavailable."
    
    def _format_schema_for_llm(self, schema: DatabaseSchema) -> str:
        """Format the database schema for LLM consumption."""
        schema_parts = [f"Database: {schema.database_name}\n"]
        
        for table in schema.tables:
            schema_parts.append(f"Table: {table.name}")
            
            # Add columns
            for column in table.columns:
                col_info = f"  - {column['name']}: {column['type']}"
                if not column['nullable']:
                    col_info += " NOT NULL"
                if column['key'] == 'PRI':
                    col_info += " PRIMARY KEY"
                if column['default']:
                    col_info += f" DEFAULT {column['default']}"
                schema_parts.append(col_info)
            
            # Add foreign keys
            if table.foreign_keys:
                schema_parts.append("  Foreign Keys:")
                for fk in table.foreign_keys:
                    schema_parts.append(f"    - {fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}")
            
            schema_parts.append("")  # Empty line between tables
        
        return "\n".join(schema_parts)
    
    def _validate_query(self, query: str, admin_mode: bool = False) -> bool:
        """Validate the generated SQL query for safety and correctness."""
        if admin_mode:
            # In admin mode, allow modification queries but block destructive schema operations
            destructive_keywords = ['DROP', 'ALTER', 'CREATE', 'TRUNCATE', 'GRANT', 'REVOKE']
            query_upper = query.upper()
            for keyword in destructive_keywords:
                if keyword in query_upper:
                    logger.warning(f"Destructive keyword '{keyword}' found in query")
                    return False
            return True
        
        # Standard mode: Basic safety checks
        dangerous_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE',
            'TRUNCATE', 'REPLACE', 'GRANT', 'REVOKE'
        ]
        
        query_upper = query.upper()
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                logger.warning(f"Dangerous keyword '{keyword}' found in query")
                return False
        
        # Must start with SELECT
        if not query_upper.strip().startswith('SELECT'):
            logger.warning("Query does not start with SELECT")
            return False
        
        return True
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize the SQL query."""
        # Remove any potential SQL injection patterns
        query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)  # Remove comments
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)  # Remove block comments
        query = query.strip()
        
        # Ensure query ends with semicolon
        if not query.endswith(';'):
            query += ';'
        
        return query
    
    def process_natural_language_query(self, natural_query: str, admin_mode: bool = False) -> QueryResult:
        """Process a natural language query and return SQL results."""
        try:
            # Get schema summary
            schema_summary = self._get_schema_summary()
            
            # Generate SQL query
            if admin_mode:
                query_type_rules = "6. Generate INSERT, UPDATE, or DELETE statements when the user asks for data modifications"
            else:
                query_type_rules = "6. Ensure the query is safe and does not modify data (SELECT only)"
            
            messages = [
                {"role": "system", "content": "You are a SQL expert assistant."},
                {"role": "user", "content": SQL_GENERATION_PROMPT.format(
                    schema=schema_summary,
                    question=natural_query,
                    query_type_rules=query_type_rules
                )}
            ]
            
            response = self._llm.chat.completions.create(
                model=self._model_name,
                messages=messages,
                extra_headers=self._extra_headers,
                temperature=0,
                max_tokens=1000
            )
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the response (remove code blocks, extra formatting)
            sql_query = re.sub(r'```sql\n?', '', sql_query)
            sql_query = re.sub(r'```\n?', '', sql_query)
            sql_query = sql_query.strip()
            
            logger.info(f"Generated SQL query: {sql_query}")
            
            # Validate query
            if not self._validate_query(sql_query, admin_mode):
                error_msg = "Generated query failed safety validation"
                if not admin_mode:
                    error_msg += " (use admin mode for data modifications)"
                return QueryResult(
                    columns=[],
                    rows=[],
                    row_count=0,
                    execution_time=0.0,
                    query=sql_query,
                    timestamp=datetime.now(),
                    error=error_msg
                )
            
            # Sanitize query
            sql_query = self._sanitize_query(sql_query)
            
            # Execute query
            result = self.db_connection.execute_query(sql_query)
            
            if result.is_success:
                logger.info(f"Natural language query processed successfully: {result.row_count} rows returned")
            else:
                logger.error(f"Query execution failed: {result.error}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process natural language query: {e}")
            return QueryResult(
                columns=[],
                rows=[],
                row_count=0,
                execution_time=0.0,
                query="",
                timestamp=datetime.now(),
                error=f"Query processing failed: {str(e)}"
            )
    
    def explain_query_results(self, query: str, result: QueryResult) -> str:
        """Generate a natural language explanation of query results."""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that explains database queries and results."},
                {"role": "user", "content": NATURAL_LANGUAGE_EXPLANATION_PROMPT.format(
                    query=query,
                    results=result.row_count
                )}
            ]
            
            response = self._llm.chat.completions.create(
                model=self._model_name,
                messages=messages,
                extra_headers=self._extra_headers,
                temperature=0,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return "Unable to generate explanation for the query results."
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """Get query suggestions based on partial input."""
        try:
            schema_summary = self._get_schema_summary()
            
            prompt = f"""Based on the database schema and partial query below, suggest 3 complete natural language questions that could be asked:

Database Schema:
{schema_summary}

Partial Query: "{partial_query}"

Provide 3 specific, actionable questions that would work with this database:"""
            
            messages = [
                {"role": "system", "content": "You are a helpful database query assistant."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._llm.chat.completions.create(
                model=self._model_name,
                messages=messages,
                extra_headers=self._extra_headers,
                temperature=0.3,
                max_tokens=300
            )
            suggestions = response.choices[0].message.content.strip().split('\n')
            
            # Clean up suggestions
            suggestions = [s.strip('123456789. -') for s in suggestions if s.strip()]
            return suggestions[:3]
            
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return []
    
    def generate_preview_query(self, modification_query: str) -> Optional[str]:
        """Generate a SELECT query to preview what would be affected by a modification query."""
        try:
            query_upper = modification_query.upper().strip()
            
            if query_upper.startswith('DELETE'):
                # Convert DELETE to SELECT to show what would be deleted
                # Example: DELETE FROM table WHERE condition -> SELECT * FROM table WHERE condition
                delete_pattern = r'DELETE\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+?))?(?:\s*;)?$'
                match = re.match(delete_pattern, modification_query, re.IGNORECASE | re.DOTALL)
                if match:
                    table_name = match.group(1)
                    where_clause = match.group(2) if match.group(2) else ""
                    if where_clause:
                        return f"SELECT * FROM {table_name} WHERE {where_clause};"
                    else:
                        return f"SELECT COUNT(*) as total_rows_to_delete FROM {table_name};"
            
            elif query_upper.startswith('UPDATE'):
                # Convert UPDATE to SELECT to show what would be updated
                # Example: UPDATE table SET col=val WHERE condition -> SELECT * FROM table WHERE condition
                update_pattern = r'UPDATE\s+(\w+)\s+SET\s+.+?(?:\s+WHERE\s+(.+?))?(?:\s*;)?$'
                match = re.match(update_pattern, modification_query, re.IGNORECASE | re.DOTALL)
                if match:
                    table_name = match.group(1)
                    where_clause = match.group(2) if match.group(2) else ""
                    if where_clause:
                        return f"SELECT * FROM {table_name} WHERE {where_clause};"
                    else:
                        return f"SELECT COUNT(*) as total_rows_to_update FROM {table_name};"
            
            # For INSERT queries, we can't preview existing data, so return None
            return None
            
        except Exception as e:
            logger.error(f"Failed to generate preview query: {e}")
            return None
    
    def clear_cache(self) -> None:
        """Clear the schema cache."""
        self._schema_cache = None
        logger.info("Schema cache cleared")