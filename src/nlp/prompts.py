"""Prompt templates for SQL generation."""

from langchain.prompts import PromptTemplate

SQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""You are a SQL expert. Given the database schema below, write a SQL query to answer the user's question.

Database Schema:
{schema}

User Question: {question}

Rules:
1. Only use tables and columns that exist in the provided schema
2. Write valid MySQL syntax
3. Use proper JOIN clauses when accessing multiple tables
4. Include appropriate WHERE clauses for filtering
5. Use aggregate functions (COUNT, SUM, AVG, etc.) when needed
6. Ensure the query is safe and does not modify data (SELECT only)
7. Add LIMIT clause if the result set might be very large
8. Use proper column aliases for better readability

Return only the SQL query without any explanation or formatting:"""
)

SCHEMA_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["schema"],
    template="""Summarize the following database schema in a clear, concise way that helps with SQL generation:

{schema}

Focus on:
1. Table names and their purposes
2. Key columns and their data types
3. Relationships between tables (foreign keys)
4. Primary keys

Provide a structured summary that would help in generating accurate SQL queries:"""
)

QUERY_VALIDATION_PROMPT = PromptTemplate(
    input_variables=["query", "schema"],
    template="""Review the following SQL query for correctness and safety:

SQL Query:
{query}

Database Schema:
{schema}

Check for:
1. Syntax errors
2. Non-existent tables or columns
3. Unsafe operations (INSERT, UPDATE, DELETE, DROP, etc.)
4. Missing JOIN conditions
5. Potential performance issues

If the query is safe and correct, respond with "VALID".
If there are issues, respond with "INVALID: [explanation of issues]":"""
)

NATURAL_LANGUAGE_EXPLANATION_PROMPT = PromptTemplate(
    input_variables=["query", "results"],
    template="""Explain the following SQL query and its results in simple, natural language:

SQL Query:
{query}

Query Results Summary:
- Number of rows returned: {results}

Provide a clear explanation of:
1. What the query is asking for
2. Which tables are being used
3. Any filters or conditions applied
4. What the results represent

Keep the explanation simple and accessible to non-technical users:"""
)