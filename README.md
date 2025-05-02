# Rewritten by AI Assistant

# Supabase Multiagent System

A multi-agent system built using LangGraph to query restaurant and related post data from a Supabase database.

## Description

This project is a multi-agent system that uses LangGraph to query restaurant and related post data from a Supabase database. The system consists of five agents: Initial Planner, Validation Planner, SQL Generator, Vector Generator, and Synthesizer.

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- OpenAI API key
- Google Generative AI (Gemini) API key

### Installation

1. Clone the repository
2. Install dependencies
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   Create a `.env` file in the project root directory with the following content:
   ```
   # LLM API KEYS
   GOOGLE_API_KEY=your_Google_API_key
   OPENAI_API_KEY=your_OpenAI_API_key

   # SUPABASE KEYS
   SUPABASE_URL=https://wdpeoyugsxqnpwwtkqsl.supabase.co
   SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndkcGVveXVnc3hxbnB3d3RrcXNsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQxNDgwNzgsImV4cCI6MjA1OTcyNDA3OH0.9bUpuZCOZxDSH3KsIu6FwWZyAvnV5xPJGNpO3luxWOE

   # Vector search requirements
   SUPABASE_DB_PASSWORD=Supabase864259
   SUPABASE_DB_HOST=aws-0-us-east-1.pooler.supabase.com 
   SUPABASE_DB_USER=postgres.wdpeoyugsxqnpwwtkqsl 
   SUPABASE_DB_NAME=postgres 
   SUPABASE_DB_PORT=5432

   # (Optional) Specify the model name to use
   GEMINI_MODEL_NAME="gemini-2.0-flash"
   EMBEDDING_MODEL_NAME="text-embedding-3-small"
   ```

## Usage

Run the main program:

```bash
python3 main.py
```

The system will start an interactive command-line interface. Enter your query, and the system will process and return the results.

Example queries:
- "What are the top 3 hotpot restaurants in Shanghai? What are their scores and average prices?"
- "Which restaurant serves delicious Beijing Duck on Xiaohongshu?"
- "Find related posts about popular restaurants in Beijing, with a focus on elegant environments"

Enter `exit` to quit the system.

## Project Structure

- `main.py`: Main entry point, sets up and runs the multi-agent system
- `agents.py`: Defines all agent nodes
- `tools.py`: Contains tools for SQL queries and vector searches
- `utils.py`: Helper functions and utilities
- `database_metadata.md`: Database metadata

```
/project-root
├── /src
│   ├── main.py
│   ├── agents.py
│   ├── tools.py
│   └── utils.py
├── /data
│   └── database_metadata.md
├── requirements.txt
└── README.md
```

## Contributing

(Optional: Guidelines for contributing to the project.)

## License

(Optional: Information about the project's license.)

## Acknowledgements

(Optional: Credits or acknowledgements.)