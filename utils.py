# utils.py
# Helper functions for the Supabase Multiagent System
# Created with database connection functions and environment variable handling
# Updated format_sql_results to handle Decimal and date types
# Updated get_direct_db_connection to prioritize DATABASE_URL for Render compatibility

import os
from typing import Dict, Any, List
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from supabase import create_client
import json
from decimal import Decimal
import datetime # Import datetime

# Load environment variables
load_dotenv()

# --- Added DATABASE_URL --- 
DATABASE_URL = os.getenv("DATABASE_URL") 

# Environment variables for Supabase and OpenAI
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Database connection details
DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_NAME = os.getenv("SUPABASE_DB_NAME")
DB_USER = os.getenv("SUPABASE_DB_USER")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_PORT = os.getenv("SUPABASE_DB_PORT")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def get_direct_db_connection():
    """Create a direct connection to the Postgres database.
    Prioritizes DATABASE_URL environment variable (used by Render).
    Falls back to individual DB_* variables if DATABASE_URL is not set.
    """
    try:
        if DATABASE_URL:
            # Use DATABASE_URL if available (preferred for Render)
            print("Connecting using DATABASE_URL...") # Add log
            conn = psycopg2.connect(DATABASE_URL)
        else:
            # Fallback to individual variables (for local dev without DATABASE_URL)
            print("Connecting using individual DB variables...") # Add log
            if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
                raise ValueError("Missing one or more required DB environment variables (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT) when DATABASE_URL is not set.")
            conn = psycopg2.connect(
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                port=DB_PORT
            )
        return conn
    except ValueError as ve:
        print(f"Configuration Error: {ve}") # Log missing variable error
        return None
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def execute_sql_query(query: str) -> List[Dict[str, Any]]:
    """Execute a SQL query and return results as a list of dictionaries."""
    conn = get_direct_db_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
            # Convert from RealDictRow to regular dict
            return [dict(row) for row in result]
    except Exception as e:
        print(f"Error executing query: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_database_metadata() -> str:
    """Read the database metadata from the markdown file."""
    try:
        with open("database_metadata.md", "r") as f:
            return f.read()
    except FileNotFoundError:
        print("Error: database_metadata.md not found.")
        return ""
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        return ""

# Custom JSON encoder to handle Decimal and date/datetime
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            # Convert Decimal to string to preserve precision
            return str(obj) 
        elif isinstance(obj, (datetime.date, datetime.datetime)):
             # Convert date/datetime to ISO 8601 string format
            return obj.isoformat()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

def format_sql_results(results: List[Dict[str, Any]]) -> str:
    """Format SQL query results into a readable string, handling Decimal and date types."""
    if not results:
        return "No results found."
    
    try:
        # Use the custom encoder
        formatted = json.dumps(results, ensure_ascii=False, indent=2, cls=CustomEncoder)
        return formatted 
    except Exception as e:
        print(f"Error formatting SQL results: {e}")
        # Fallback or simplified representation if encoding fails
        # Ensure fallback can handle the problematic types too
        try:
            return str([{k: str(v) for k, v in row.items()} for row in results])
        except Exception:
             return "Error: Could not format results even with fallback." 