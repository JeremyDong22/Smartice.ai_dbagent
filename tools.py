# tools.py
# SQL and vector search tools for the Supabase Multiagent System
# Created to provide SQL execution and embedding-based vector search capabilities

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
import numpy as np
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from utils import execute_sql_query

# Initialize OpenAI client (used if embedding is not pre-computed)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

class SQLQueryInput(BaseModel):
    query: str = Field(description="The SQL query to execute")

class VectorSearchInput(BaseModel):
    text: str = Field(description="The text to search for (used for context or if embedding not provided)")
    num_results: int = Field(description="Number of results to return", default=5)
    embedding_str: Optional[str] = Field(default=None, description="Pre-computed embedding vector string (e.g., '[0.1, 0.2, ...]')")
    sql_filter: Optional[str] = Field(default=None, description="Optional SQL WHERE clause condition (e.g., \"b.name = '三出山'\")")

@tool
def sql_query_tool(query_input: SQLQueryInput) -> str:
    """Execute a SQL query against the Supabase database and return the results."""
    try:
        results = execute_sql_query(query_input.query)
        # Use the formatter here too for consistency
        from utils import format_sql_results 
        return f"SQL results: {format_sql_results(results)}"
    except Exception as e:
        return f"Error executing SQL query '{query_input.query}': {str(e)}"

@tool
def validate_column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a specified table."""
    query = f"""
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        AND column_name = '{column_name}'
    ) as exists;
    """
    result = execute_sql_query(query)
    return result[0]['exists'] if result else False

@tool
def validate_value_exists(table_name: str, column_name: str, value: str) -> bool:
    """Check if a value exists in a specified column of a table."""
    # Using LIKE for more flexible matching, especially for Chinese characters
    # Ensure value is properly escaped for SQL LIKE
    escaped_value = value.replace("'", "''")
    query = f"""
    SELECT EXISTS (
        SELECT 1
        FROM {table_name}
        WHERE "{column_name}" LIKE '%{escaped_value}%'
    ) as exists;
    """
    print(f"Debug: Executing validation query: {query}") # Add debug print
    result = execute_sql_query(query)
    return result[0]['exists'] if result else False

@tool
def get_similar_values(table_name: str, column_name: str, value: str, limit: int = 5) -> List[str]:
    """Get similar values to the given value in a specified column."""
    escaped_value = value.replace("'", "''")
    query = f"""
    SELECT DISTINCT "{column_name}"
    FROM {table_name}
    WHERE "{column_name}" LIKE '%{escaped_value}%'
    LIMIT {limit};
    """
    results = execute_sql_query(query)
    return [row[column_name] for row in results] if results else []

@tool
def vector_search_tool(search_input: VectorSearchInput) -> str:
    """Search for similar content in the posts table using vector embeddings, optionally applying an SQL filter."""
    final_embedding_str = None

    # Use pre-computed embedding if provided
    if search_input.embedding_str:
        print("Debug: Using pre-computed embedding string.")
        if search_input.embedding_str.startswith('[') and search_input.embedding_str.endswith(']'):
             final_embedding_str = search_input.embedding_str
        else:
             print(f"Warning: Provided embedding_str has unexpected format: {search_input.embedding_str[:50]}...")
             # Fallback handled below
             pass

    # If no valid pre-computed embedding, generate it from text
    if not final_embedding_str:
        print(f"Debug: Generating embedding from text: '{search_input.text}'")
        if not search_input.text:
             return "Error: Cannot perform vector search without text or a valid pre-computed embedding."
        try:
            response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL_NAME,
                input=search_input.text
            )
            embedding = response.data[0].embedding
            final_embedding_str = str(embedding) # Convert list to string
        except Exception as e:
            return f"Error creating embedding for text '{search_input.text}': {str(e)}"

    # Ensure we have a valid embedding string now
    if not final_embedding_str or not (final_embedding_str.startswith('[') and final_embedding_str.endswith(']')):
        return f"Error: Failed to obtain a valid embedding vector string."

    pgvector_embedding_str = final_embedding_str # Use '[...]' format directly
    pgvector_embedding_str_escaped = pgvector_embedding_str.replace("'", "''")

    # --- Construct the SQL Query with Optional Filter --- 
    sql_filter_clause = ""
    # If a filter is provided, construct the necessary JOINs and WHERE clause part
    if search_input.sql_filter:
        # Basic assumption: filter involves brand name like "b.name = 'Some Brand'"
        # More robust parsing might be needed for complex filters
        print(f"Debug: Applying SQL filter: {search_input.sql_filter}")
        # We need the JOINs to apply the filter
        sql_filter_clause = f"""JOIN post_brand pb ON p.post_id = pb.post_id
JOIN brand b ON pb.brand_id = b.brand_id
WHERE {search_input.sql_filter}"""
    else:
        # If no filter, we don't need the JOINs for the basic vector search
        print("Debug: No SQL filter applied.")
        pass # No WHERE clause needed if no filter

    # Query for similar posts using vector similarity
    # Note: We select columns from 'p' (posts table)
    query = f"""
    SELECT p.post_id, p.title, p.content, p.likes, p.author, p.images, p.likes, p.collections, p.comments
    FROM posts p 
    {sql_filter_clause} -- Add JOINs and WHERE clause if filter exists
    ORDER BY p.vec_content <=> '{pgvector_embedding_str_escaped}'::vector
    LIMIT {search_input.num_results};
    """

    print(f"Debug: Executing vector search query (embedding omitted for brevity): {query}")
    try:
        results = execute_sql_query(query)
        from utils import format_sql_results
        formatted_results = format_sql_results(results)
        return f"Vector search results: {formatted_results}"
    except Exception as e:
        print(f"Error executing vector search query: {e}")
        # Provide more context in error message
        return f"Error executing vector search query with filter '{search_input.sql_filter}': {str(e)}" 