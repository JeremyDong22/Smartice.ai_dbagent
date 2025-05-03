# api_server.py
# FastAPI server to expose the LangGraph agent as an API.
# Handles incoming requests, manages state, and streams responses.
# API keys (OpenAI, Google) are primarily sourced from environment variables 
# (OPENAI_API_KEY, GOOGLE_API_KEY) unless explicitly provided in the request body.

from itertools import tee
import uvicorn
import asyncio
import json # Added for SSE
from fastapi import FastAPI, HTTPException, Request # Added Request
from fastapi.responses import StreamingResponse # Added for streaming
from fastapi.middleware.cors import CORSMiddleware # Added for Frontend
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncGenerator, List # Added AsyncGenerator and List
import os

# Import the necessary functions from main.py
# Assuming main.py is in the same directory or accessible in PYTHONPATH
try:
    # AgentState type hint might be useful if defined globally in main
    from main import initialize_app, process_query, AgentState 
except ImportError as e:
    print(f"Error importing from main.py: {e}")
    print("Please ensure main.py is in the same directory or accessible via PYTHONPATH.")
    # Define dummy functions if import fails, so server can start but will error on invoke
    AgentState = Dict[str, Any] # Fallback type
    def initialize_app():
        print("ERROR: Could not initialize app from main.py")
        return None
    def process_query(app, query) -> AgentState:
        raise RuntimeError("Agent app could not be initialized.")

# --- Pydantic Models for Request and Response ---

class QueryRequest(BaseModel):
    """Request model expects a 'query' field."""
    query: str
    google_api_key: Optional[str] = None # Added
    openai_api_key: Optional[str] = None # Added

# Updated Response Model
class QueryResponseWithSteps(BaseModel):
    """Response model includes final response and intermediate steps."""
    final_response: Optional[str] # Make optional in case of failure
    intermediate_steps: Dict[str, Any] = {} # Dictionary to hold step outputs

# --- FastAPI Application Setup ---

app = FastAPI(
    title="LangGraph DB Agent API",
    description="API endpoint to interact with the Supabase multiagent system. Provides standard and streaming responses with real-time steps.",
    version="0.4.0", # Bumped version
)

# --- CORS Middleware --- Add this for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity (adjust for production)
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"], # Allow all headers
)

# Initialize the LangGraph application when the API server starts
# The graph structure itself is initialized once.
# LLM configurations (like API keys) are handled per-request via config.
langgraph_app = initialize_app()

# Helper function to build the config dictionary FOR LLMs
def build_llm_config(google_api_key_req: Optional[str], openai_api_key_req: Optional[str]) -> Dict:
    """Builds the LLM configuration dictionary, prioritizing request keys over environment variables."""
    # Use request key if provided, otherwise fall back to environment variable
    resolved_google_key = google_api_key_req or os.getenv("GOOGLE_API_KEY")
    resolved_openai_key = openai_api_key_req or os.getenv("OPENAI_API_KEY")
    
    # Return configuration suitable for LangGraph state
    return {
        "configurable": {
            "google_api_key": resolved_google_key,
            "openai_api_key": resolved_openai_key,
        }
    }

# --- API Endpoint (Updated) ---

@app.post("/invoke", response_model=QueryResponseWithSteps) # Use updated response model
async def invoke_agent(request: QueryRequest):
    """
    Receives a user query, processes it through the LangGraph agent,
    and returns the final response along with intermediate step outputs.
    Accepts optional API keys in the request body.
    """
    if langgraph_app is None:
         raise HTTPException(status_code=500, detail="Agent application failed to initialize.")

    query = request.query
    google_api_key_req = request.google_api_key
    openai_api_key_req = request.openai_api_key
    print(f"Received query for API: {query}")
    print(f"Invoke: Received Google Key from request: {'******' if google_api_key_req else 'None'}")
    print(f"Invoke: Received OpenAI Key from request: {'******' if openai_api_key_req else 'None'}")

    # --- Resolve keys (request OR environment) --- 
    # These will be put into the initial state for non-LLM components like embedding client
    resolved_openai_key = openai_api_key_req or os.getenv("OPENAI_API_KEY")
    resolved_google_key = google_api_key_req or os.getenv("GOOGLE_API_KEY")

    try:
        # --- Build LLM Config --- (Uses keys specifically from request for override)
        request_config = build_llm_config(google_api_key_req, openai_api_key_req)
        print(f"Invoke LLM Config: {request_config}")

        # --- Create Initial State with Resolved Keys --- 
        initial_state = {
            "query": query,
            "openai_api_key": resolved_openai_key, # Pass resolved key via state
            "google_api_key": resolved_google_key # Pass resolved key via state
        }
        print(f"Invoke Initial State Keys: OpenAI set: {bool(resolved_openai_key)}, Google set: {bool(resolved_google_key)}")

        # --- Pass initial_state AND config to invoke --- 
        full_state: AgentState = langgraph_app.invoke(initial_state, config=request_config)
        
        # Extract the final response
        final_response = full_state.get("final_response")
        
        # Prepare intermediate steps dictionary
        # Exclude the original query and the final response itself from steps
        # Also filter out None values for cleaner output
        excluded_keys = {'query', 'final_response'}
        intermediate_steps = {
            k: v for k, v in full_state.items() 
            if k not in excluded_keys and v is not None
        }
        
        print(f"Generated response: {final_response}") 
        print(f"Intermediate steps: {intermediate_steps}")

        # Return the structured response
        return QueryResponseWithSteps(
            final_response=final_response,
            intermediate_steps=intermediate_steps
        )
        
    except Exception as e:
        print(f"Error during agent invocation: {e}")
        # Log the exception traceback for detailed debugging if needed
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent processing error: {str(e)}")

# --- New Streaming Generator using app.stream() ---
async def langgraph_stream_generator(query: str, google_api_key_req: Optional[str] = None, openai_api_key_req: Optional[str] = None) -> AsyncGenerator[str, None]:
    """Generates SSE events by streaming LangGraph execution.
    Yields intermediate step results and final response chunks.
    Accepts optional API keys.
    """
    if langgraph_app is None:
        # Handle case where app failed to initialize
        error_data = json.dumps({"error": "Agent application failed to initialize."})
        yield f"event: error\ndata: {error_data}\n\n"
        return
        
    final_response_text = None # Initialize here
    last_state = None # To store the very last state
    try:
        # --- Resolve keys (request OR environment) --- 
        resolved_openai_key = openai_api_key_req or os.getenv("OPENAI_API_KEY")
        resolved_google_key = google_api_key_req or os.getenv("GOOGLE_API_KEY")

        # --- Build LLM Config --- (Uses keys specifically from request for override)
        request_config = build_llm_config(google_api_key_req, openai_api_key_req)
        print(f"Stream LLM Config: {request_config}")
        
        # --- Create Initial State with Resolved Keys --- 
        initial_state = {
            "query": query,
            "openai_api_key": resolved_openai_key, # Pass resolved key via state
            "google_api_key": resolved_google_key # Pass resolved key via state
        }
        print(f"Stream Initial State Keys: OpenAI set: {bool(resolved_openai_key)}, Google set: {bool(resolved_google_key)}")

        # --- Pass initial_state AND config to stream --- 
        for event in langgraph_app.stream(initial_state, config=request_config):
            last_state = event # Keep track of the last event data
            for node_name, output in event.items():
                if isinstance(output, dict):
                     # Let's extract relevant info from the node's output dictionary
                     # Customize this based on what's useful to see from each node
                     step_output = {}
                     # Example: Extract specific keys if they exist
                     if 'initial_plan' in output: step_output['initial_plan'] = output['initial_plan']
                     if 'validation_result' in output: step_output['validation_result'] = output['validation_result']
                     if 'sql_query' in output: step_output['sql_query'] = output['sql_query']
                     if 'sql_results' in output: step_output['sql_results'] = output['sql_results']
                     if 'vector_search_text' in output: step_output['vector_search_text'] = output['vector_search_text']
                     if 'vector_results' in output: step_output['vector_results'] = output['vector_results']
                     # Add more keys as needed...
                     
                     # Check if this is the final state containing the response
                     if node_name == "__end__" and "final_response" in output:
                         final_response_text = output.get("final_response")
                         print("Stream: Identified final response.")

                     # Check if this is the synthesizer output containing the response
                     if node_name == 'synthesizer' and 'final_response' in output:
                         final_response_text = output.get("final_response")
                         print(f"Stream: Captured final response from synthesizer: {final_response_text[:50]}...")
                         step_output['final_response_preview'] = output['final_response'][:50] + '...' # Maybe add preview to steps?
                     
                     if step_output:
                        step_data = json.dumps({"node": node_name, "output": step_output})
                        print(f"Stream: Sending step - Node: {node_name}, Output Keys: {list(step_output.keys())}")
                        yield f"event: step\ndata: {step_data}\n\n"
                        await asyncio.sleep(0.05) # Small delay between steps

    except Exception as e:
        print(f"Error during LangGraph stream: {e}")
        import traceback
        traceback.print_exc()
        error_data = json.dumps({"error": f"Agent streaming error: {str(e)}"})
        yield f"event: error\ndata: {error_data}\n\n"
        return # Stop generation on error

    # --- Stream the final response AFTER the loop --- 
    # Double-check if we captured it from synthesizer, or get from the absolute last state
    if final_response_text is None and last_state:
         # Find the state dict within the last event (it's often keyed by node name)
         final_state_dict = list(last_state.values())[0] if last_state else {}
         if isinstance(final_state_dict, dict):
              final_response_text = final_state_dict.get("final_response")
              print("Stream: Captured final response from last state.")

    if final_response_text:
        print("Stream: Starting final response text stream")
        for char in final_response_text:
            yield f"event: text\ndata: {json.dumps(char)}\n\n"
            await asyncio.sleep(0.01)
        print("\nStream: Finished final response text stream")
    else:
        print("Stream: No final response text captured to stream.")
        yield f"event: text\ndata: {json.dumps('(No final answer generated)')}\n\n" # Send placeholder

    # Send end-of-stream event
    yield f"event: end\ndata: Stream finished\n\n"
    print("Stream: Sent end event")

# --- Updated Streaming API Endpoint --- 

@app.post("/stream")
async def stream_agent(request: QueryRequest):
    """
    Receives a query, processes it using langgraph.stream(), 
    and streams intermediate steps and the final response via SSE.
    Accepts optional API keys in the request body.
    """
    query = request.query
    google_api_key_req = request.google_api_key 
    openai_api_key_req = request.openai_api_key 
    print(f"Received query for /stream: {query}")
    print(f"Stream: Received Google Key from request: {'******' if google_api_key_req else 'None'}")
    print(f"Stream: Received OpenAI Key from request: {'******' if openai_api_key_req else 'None'}")
    # Pass keys from request to the generator
    return StreamingResponse(langgraph_stream_generator(query, google_api_key_req, openai_api_key_req), media_type="text/event-stream")

# --- Run the Server ---

if __name__ == "__main__":
    # Run the FastAPI server using uvicorn
    # Use reload=True for development to automatically restart on code changes
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)

# Example usage with curl:
# curl -X POST http://localhost:8000/invoke -H "Content-Type: application/json" -d '{"query": "上海有什么好吃的火锅？"}' 