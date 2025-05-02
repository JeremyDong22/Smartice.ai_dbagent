# main.py
# Main entry point for the Supabase Multiagent System
# Created to set up and run the multiagent graph
# Updated with logging functionality to visualize agent thinking process
# Updated to handle low-intent queries efficiently
# Removed dependency on deleted prompt_config.py

import os
import argparse
from typing import Dict, Any, List, Tuple, Annotated, TypedDict, Literal, Optional
from dotenv import load_dotenv
from langchain.schema import BaseMessage
from langgraph.graph import StateGraph, END
from tools import sql_query_tool, vector_search_tool, SQLQueryInput
from agents import (
    initial_planner,
    validation_planner,
    sql_generator,
    vector_generator,
    synthesizer
)
from utils import execute_sql_query, format_sql_results
from logger import (
    log_agent_start, 
    log_agent_end, 
    log_router_decision, 
    print_divider,
    print_query_start,
    print_response
)

# Load environment variables
load_dotenv()

# Define state schema
class AgentState(TypedDict):
    query: str
    # API Keys (passed from invocation)
    openai_api_key: Optional[str] # Added
    google_api_key: Optional[str] # Added
    # Agent flow fields
    has_query_intent: bool # 新增：标记是否有明确查询意图
    initial_plan: str
    identified_brand: Optional[str] # Added: Brand identified by initial planner
    identified_city: Optional[str] # Added: City identified by initial planner
    identified_list: Optional[str] # Added: List/Category identified by initial planner
    vector_search_query_suggestion: Optional[str] # Added from agents.py
    validation_result: str
    query_type: str
    sql_query: str
    sql_results: str
    sql_filter: Optional[str] # 新增: 用于向量搜索的SQL过滤条件
    vector_search_text: str # Text used for embedding/context
    embedding_vector_str: Optional[str] # Added from agents.py
    vector_results: str
    final_response: str

# Wrap agents with logging
def wrap_with_logging(name, agent_func):
    """Wraps an agent function with logging capabilities."""
    def wrapped(state):
        log_agent_start(name, state)
        result = agent_func(state)
        log_agent_end(name, result)
        return result
    return wrapped

# Define the agent graph
def create_agent_graph() -> StateGraph:
    """Create the agent graph with defined nodes and edges."""
    
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each agent with logging wrappers
    workflow.add_node("initial_planner", wrap_with_logging("initial_planner", initial_planner))
    workflow.add_node("validation_planner", wrap_with_logging("validation_planner", validation_planner))
    workflow.add_node("sql_generator", wrap_with_logging("sql_generator", sql_generator))
    workflow.add_node("vector_generator", wrap_with_logging("vector_generator", vector_generator))
    workflow.add_node("execute_sql", wrap_with_logging("execute_sql", execute_sql))
    workflow.add_node("execute_vector_search", wrap_with_logging("execute_vector_search", execute_vector_search))
    workflow.add_node("synthesizer", wrap_with_logging("synthesizer", synthesizer))
    
    # Define conditional edges
    # 从 initial_planner 开始，根据是否有意图决定下一步
    workflow.add_conditional_edges(
        "initial_planner",
        intent_router, # 新的路由函数
        {
            "continue": "validation_planner", # 有意图，继续验证
            "end": "synthesizer"            # 无意图，直接结束
        }
    )
    
    # Add conditional routing based on query_type after validation
    workflow.add_conditional_edges(
        "validation_planner",
        validation_router,
        {
            "sql": "sql_generator",
            "vector": "vector_generator", # Go to vector gen if only vector needed
            "sql+vector": "sql_generator" # Go to sql gen first if both needed
        }
    )
    
    # Add edges from generators to executors
    workflow.add_edge("sql_generator", "execute_sql")
    
    # Conditional routing after SQL execution
    workflow.add_conditional_edges(
        "execute_sql",
        sql_router,
        {
            "need_vector": "vector_generator", # If vector is needed after SQL, go to vector gen
            "synthesize": "synthesizer" # If only SQL was needed, go to synthesize
        }
    )
    
    # After vector generator (which now creates embedding), go execute the search
    workflow.add_edge("vector_generator", "execute_vector_search")
    workflow.add_edge("execute_vector_search", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    # Set entry point
    workflow.set_entry_point("initial_planner")
    
    return workflow

# 新增: 路由函数，判断是否有查询意图
def intent_router(state: AgentState) -> Literal["continue", "end"]:
    """Routes based on whether a query intent was detected."""
    if state.get("has_query_intent", False):
        decision = "continue"
    else:
        decision = "end"
    log_router_decision("intent_router", decision)
    return decision

# Define router for determining next step after validation with logging
def validation_router(state: AgentState) -> Literal["sql", "vector", "sql+vector"]:
    """Route to the appropriate generator based on query type with logging."""
    decision = state["query_type"]
    log_router_decision("validation_router", decision)
    return decision

# Define router for determining next step after SQL execution with logging
def sql_router(state: AgentState) -> Literal["need_vector", "synthesize"]:
    """Route to vector search or synthesizer after SQL execution with logging."""
    # Check if the original plan involved vector search
    if state.get("query_type") == "sql+vector":
        decision = "need_vector"
    else:
        decision = "synthesize"
    
    log_router_decision("sql_router", decision)
    return decision

# Execute SQL query node
def execute_sql(state: AgentState) -> AgentState:
    """Execute the SQL query and update state with results."""
    sql_query = state.get("sql_query", "")
    
    if not sql_query:
        print("Debug: No SQL query to execute.")
        return {**state, "sql_results": "No SQL query provided."}
        
    try:
        # Execute the query
        results = execute_sql_query(sql_query)
        formatted_results = format_sql_results(results)
    except Exception as e:
        formatted_results = f"Error executing SQL query: {str(e)}"
        print(formatted_results)
    
    return {
        **state,
        "sql_results": formatted_results
    }

# Execute vector search node
def execute_vector_search(state: AgentState) -> AgentState:
    """Execute the vector search using pre-computed embedding and optional SQL filter.
       Passes the resolved OpenAI API key to the tool for potential fallback usage.
    """
    vector_search_text = state.get("vector_search_text", "")
    embedding_vector_str = state.get("embedding_vector_str")
    sql_filter = state.get("sql_filter")
    # --- Get the resolved OpenAI API key from state --- 
    openai_api_key = state.get("openai_api_key")
    num_results = 5

    # Prepare the input for the vector search tool, including the API key
    tool_input_dict = {
        "text": vector_search_text,
        "num_results": num_results,
        "embedding_str": embedding_vector_str,
        "sql_filter": sql_filter,
        "openai_api_key": openai_api_key # Pass the key here
    }

    print(f"Debug: Invoking vector_search_tool with input (API key omitted for brevity): { {k:v for k,v in tool_input_dict.items() if k != 'openai_api_key'} }")

    try:
        # Invoke the tool using the structured input
        results = vector_search_tool.invoke(tool_input_dict)
    except Exception as e:
        results = f"Error executing vector search: {str(e)}"
        print(f"Error details: {e}")

    return {
        **state,
        "vector_results": results
    }

# Create app
def initialize_app():
    """Initialize the app."""
    # 创建并编译agent图
    return create_agent_graph().compile()

# Process query
def process_query(app, query: str) -> AgentState:
    """Process a user query through the agent graph and return the full final state."""
    # Initialize state with query
    initial_state = {"query": query}
    
    # Log query start
    print_query_start(query)
    
    # Run the graph
    # Add config to potentially handle recursion limits if cycles occur
    # The result is the final state dictionary (AgentState)
    result = app.invoke(initial_state, config={"recursion_limit": 10})
    
    # Return the entire final state dictionary
    return result

def main():
    """Main function to run the app interactively."""
    # # Removed argparse related to prompt config
    # parser = argparse.ArgumentParser(description="Supabase多智能体查询系统")
    # parser.add_argument("--prompts", type=str, help="自定义prompt配置文件路径")
    # args = parser.parse_args()
    
    # 初始化应用
    # Removed prompt config loading: app = initialize_app(args.prompts)
    app = initialize_app()
    
    print("欢迎使用Supabase多智能体查询系统!")
    print("输入 'exit' 退出系统。\n")
    
    while True:
        query = input("\n请输入您的查询: ")
        
        if query.lower() == 'exit':
            print("感谢使用Supabase多智能体查询系统，再见!")
            break
        
        print_divider()
        print("正在处理您的查询...\n")
        # Modify how response is handled in main interactive loop if needed
        # Since process_query now returns a dict, extract final_response for printing
        full_state = process_query(app, query)
        response_text = full_state.get("final_response", "No response generated.")
        print_response(response_text)
        # Optionally print the full state for debugging in interactive mode
        # print("\n--- Full State ---")
        # import json
        # print(json.dumps(full_state, indent=2, ensure_ascii=False))
        # print("--- End Full State ---")

if __name__ == "__main__":
    main() 