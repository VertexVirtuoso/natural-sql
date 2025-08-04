"""Natural language query processing using LangChain."""

import logging
import re
from typing import Optional, Dict, Any, List
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import ChatPromptTemplate

from ..config.settings import get_settings
from ..database.connection import get_db_connection
from ..database.models import DatabaseSchema, QueryResult
from .chat_openrouter import ChatOpenRouter
from .vector_knowledge import VectorKnowledgeManager
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
        self._memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="sql_query"
        )
        
        # Initialize vector knowledge manager
        try:
            self._vector_knowledge = VectorKnowledgeManager()
            # Seed initial knowledge on first run
            self._vector_knowledge.seed_initial_knowledge()
            logger.info("Vector knowledge manager initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize vector knowledge manager: {e}")
            self._vector_knowledge = None
        
        self._setup_llm()
    
    def _setup_llm(self) -> None:
        """Set up the language model using LangChain with OpenRouter."""
        try:
            if self.settings.openrouter_api_key:
                # Use ChatOpenRouter for OpenRouter API
                self._llm = ChatOpenRouter.from_settings(self.settings)
                logger.info(f"LangChain ChatOpenRouter initialized with model: {self.settings.openrouter_model}")
            elif self.settings.openai_api_key:
                # Fallback to standard ChatOpenAI for direct OpenAI
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    api_key=self.settings.openai_api_key,
                    model="gpt-3.5-turbo",
                    temperature=0
                )
                logger.info("LangChain ChatOpenAI initialized successfully")
            else:
                raise ValueError(
                    "Either OPENROUTER_API_KEY or OPENAI_API_KEY is required. "
                    "Please set one of these environment variables."
                )
        except Exception as e:
            logger.error(f"Failed to initialize LangChain LLM: {e}")
            raise
    
    def _get_schema_summary(self) -> str:
        """Get a formatted schema summary for the LLM."""
        if self._schema_cache is not None:
            return self._schema_cache
        
        try:
            schema = self.db_connection.get_database_schema()
            schema_text = self._format_schema_for_llm(schema)
            
            # Create prompt template for schema summary
            schema_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a database schema analyst."),
                ("human", SCHEMA_SUMMARY_PROMPT.template)
            ])
            
            # Create LLMChain for schema summarization
            schema_chain = LLMChain(
                llm=self._llm,
                prompt=schema_prompt,
                output_key="schema_summary"
            )
            
            # Generate summary using LangChain
            result = schema_chain.invoke({"schema": schema_text})
            self._schema_cache = result["schema_summary"]
            
            logger.info("Schema summary generated and cached using LangChain")
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
        """Process a natural language query and return SQL results using SequentialChain."""
        try:
            # Get schema summary
            schema_summary = self._get_schema_summary()
            
            # Define query type rules based on admin mode
            if admin_mode:
                query_type_rules = "6. Generate INSERT, UPDATE, or DELETE statements when the user asks for data modifications"
            else:
                query_type_rules = "6. Ensure the query is safe and does not modify data (SELECT only)"
            
            # Create SequentialChain for query generation
            sql_query = self._generate_sql_with_chain(
                natural_query, schema_summary, query_type_rules
            )
            
            # Clean up the response (remove code blocks, extra formatting)
            sql_query = re.sub(r'```sql\n?', '', sql_query)
            sql_query = re.sub(r'```\n?', '', sql_query)
            sql_query = sql_query.strip()
            
            logger.info(f"Generated SQL query using LangChain: {sql_query}")
            
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
            
            # Learn from this interaction
            if self._vector_knowledge:
                try:
                    self._vector_knowledge.learn_from_interaction(
                        user_query=natural_query,
                        generated_sql=sql_query,
                        success=result.is_success,
                        error_msg=result.error if not result.is_success else None,
                        context=f"Admin mode: {admin_mode}, Row count: {result.row_count if result.is_success else 0}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to save learning from interaction: {e}")
            
            # Add query to conversation memory
            self._memory.save_context(
                {"question": natural_query},
                {"sql_query": sql_query}
            )
            
            if result.is_success:
                logger.info(f"Natural language query processed successfully using LangChain: {result.row_count} rows returned")
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
    
    def _generate_sql_with_chain(self, question: str, schema: str, query_type_rules: str) -> str:
        """Generate SQL query using LangChain with conversation memory and vector knowledge."""
        # Enhanced prompt with vector knowledge context
        enhanced_schema = schema
        context_info = ""
        similar_examples = ""
        
        if self._vector_knowledge:
            try:
                # Find similar queries from vector store
                similar_queries = self._vector_knowledge.find_similar_queries(question, k=3)
                if similar_queries:
                    similar_examples = "\n\nSimilar Query Examples:\n"
                    for i, sim_query in enumerate(similar_queries, 1):
                        similar_examples += f"{i}. User asked: \"{sim_query['metadata'].get('user_query', '')}\"\n"
                        similar_examples += f"   Successful SQL: {sim_query.get('sql', '')}\n"
                        if sim_query.get('context'):
                            similar_examples += f"   Context: {sim_query['context']}\n"
                        similar_examples += "\n"
                
                # Find domain context and business rules
                domain_context = self._vector_knowledge.find_domain_context(question, k=3)
                if domain_context:
                    context_info = "\n\nDomain Knowledge & Business Rules:\n"
                    for context in domain_context:
                        context_info += f"- {context['content']}\n"
                        if context.get('sql_context'):
                            context_info += f"  SQL Context: {context['sql_context']}\n"
                        context_info += "\n"
                
                logger.info(f"Enhanced query with {len(similar_queries)} similar examples and {len(domain_context)} context items")
                
            except Exception as e:
                logger.warning(f"Failed to retrieve vector knowledge: {e}")
        
        # Create enhanced prompt template
        enhanced_prompt_text = SQL_GENERATION_PROMPT.template
        if similar_examples:
            enhanced_prompt_text += similar_examples
        if context_info:
            enhanced_prompt_text += context_info
        
        enhanced_prompt_text += """

Use the similar examples and domain knowledge above to generate better SQL. Pay attention to:
1. Similar successful patterns from the examples
2. Domain-specific terminology and business rules
3. Common mistakes and their corrections
4. Proper table and column names from the schema"""
        
        sql_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a SQL expert assistant with access to domain knowledge and successful query patterns."),
            ("human", enhanced_prompt_text)
        ])
        
        # Create LLMChain for SQL generation with memory
        sql_chain = LLMChain(
            llm=self._llm,
            prompt=sql_prompt,
            memory=self._memory,
            output_key="sql_query"
        )
        
        # Generate SQL using the enhanced chain
        result = sql_chain.invoke({
            "schema": enhanced_schema,
            "question": question,
            "query_type_rules": query_type_rules
        })
        
        return result["sql_query"]
    
    def explain_query_results(self, query: str, result: QueryResult) -> str:
        """Generate a natural language explanation of query results using LangChain."""
        try:
            # Create prompt template for explanation
            explanation_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that explains database queries and results."),
                ("human", NATURAL_LANGUAGE_EXPLANATION_PROMPT.template)
            ])
            
            # Create LLMChain for explanation
            explanation_chain = LLMChain(
                llm=self._llm,
                prompt=explanation_prompt,
                output_key="explanation"
            )
            
            # Generate explanation using LangChain
            explanation_result = explanation_chain.invoke({
                "query": query,
                "results": result.row_count
            })
            
            return explanation_result["explanation"].strip()
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return "Unable to generate explanation for the query results."
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """Get query suggestions based on partial input using LangChain."""
        try:
            schema_summary = self._get_schema_summary()
            
            # Create prompt template for suggestions
            suggestion_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful database query assistant."),
                ("human", """Based on the database schema and partial query below, suggest 3 complete natural language questions that could be asked:

Database Schema:
{schema}

Partial Query: "{partial_query}"

Provide 3 specific, actionable questions that would work with this database:""")
            ])
            
            # Create LLMChain for suggestions with higher temperature for creativity
            suggestion_chain = LLMChain(
                llm=self._llm,
                prompt=suggestion_prompt,
                output_key="suggestions"
            )
            
            # Set higher temperature for more creative suggestions
            suggestion_chain.llm.temperature = 0.3
            
            # Generate suggestions using LangChain
            result = suggestion_chain.invoke({
                "schema": schema_summary,
                "partial_query": partial_query
            })
            
            suggestions = result["suggestions"].strip().split('\n')
            
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
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history from memory."""
        try:
            messages = self._memory.chat_memory.messages
            history = []
            
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    human_msg = messages[i]
                    ai_msg = messages[i + 1]
                    history.append({
                        "question": human_msg.content,
                        "sql_query": ai_msg.content
                    })
            
            return history
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    def clear_conversation_memory(self) -> None:
        """Clear the conversation memory."""
        self._memory.clear()
        logger.info("Conversation memory cleared")
    
    def add_context_to_memory(self, context: str) -> None:
        """Add additional context to the conversation memory."""
        try:
            self._memory.save_context(
                {"question": f"[CONTEXT] {context}"},
                {"sql_query": "[ACKNOWLEDGED]"}
            )
            logger.info("Context added to conversation memory")
        except Exception as e:
            logger.error(f"Failed to add context to memory: {e}")
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector knowledge base."""
        if not self._vector_knowledge:
            return {"error": "Vector knowledge not available"}
        
        return self._vector_knowledge.get_knowledge_stats()
    
    def add_manual_correction(self, original_query: str, failed_sql: str, 
                             corrected_sql: str, lesson: str) -> None:
        """Add a manual correction to the knowledge base."""
        if not self._vector_knowledge:
            logger.warning("Vector knowledge not available for adding correction")
            return
        
        try:
            self._vector_knowledge.add_correction(original_query, failed_sql, corrected_sql, lesson)
            logger.info(f"Added manual correction for query: {original_query[:50]}...")
        except Exception as e:
            logger.error(f"Failed to add manual correction: {e}")
    
    def clear_vector_knowledge(self) -> None:
        """Clear all vector knowledge (for reset/debugging)."""
        if not self._vector_knowledge:
            logger.warning("Vector knowledge not available")
            return
        
        try:
            self._vector_knowledge.clear_knowledge_base()
            logger.info("Vector knowledge base cleared")
        except Exception as e:
            logger.error(f"Failed to clear vector knowledge: {e}")
    
    def get_similar_queries(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get similar queries from the knowledge base for debugging/inspection."""
        if not self._vector_knowledge:
            return []
        
        try:
            return self._vector_knowledge.find_similar_queries(query, k)
        except Exception as e:
            logger.error(f"Failed to get similar queries: {e}")
            return []
    
    def reinitialize_vector_knowledge(self) -> bool:
        """Reinitialize vector knowledge manager (useful after setup-knowledge)."""
        try:
            self._vector_knowledge = VectorKnowledgeManager()
            self._vector_knowledge.seed_initial_knowledge()
            logger.info("Vector knowledge manager reinitialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reinitialize vector knowledge: {e}")
            self._vector_knowledge = None
            return False