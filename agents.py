# agents.py
# Defines the core agents (nodes) for the LangGraph application.
# Includes initial planner, validation planner, SQL generator, vector generator, and synthesizer.
# API keys (OpenAI, Google) are primarily sourced from environment variables 
# (OPENAI_API_KEY, GOOGLE_API_KEY) configured via LangGraph's ConfigurableField mechanism 
# or passed through state, falling back to direct os.getenv within specific tool calls if necessary.

import os
import json
import re
from typing import Dict, List, Any, Tuple, Optional
from langchain.prompts import ChatPromptTemplate
# Use OpenAI for potentially better tool calling
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage, SystemMessage
from langchain_core.runnables import ConfigurableField
from utils import get_database_metadata
# Import validation tools
from tools import validate_value_exists, get_similar_values, validate_column_exists

# Need openai client here for embeddings
from openai import OpenAI

# Define model names from environment or use defaults
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.0-pro")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo-preview") # Use a model known for tool use

# Initialize models with configurable API keys
# Use ConfigurableField to allow overriding the api_key via config
openai_model = ChatOpenAI(
    model=OPENAI_MODEL_NAME,
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY") # Default from env
).configurable_fields(
    openai_api_key=ConfigurableField(
        id="openai_api_key", # ID used in config (matches the target field name and api_server) 
        name="OpenAI API Key",
        description="API Key for OpenAI Model"
    )
)

gemini_model = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_NAME,
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY") # Default from env
).configurable_fields(
    google_api_key=ConfigurableField(
        id="google_api_key", # ID to use in config
        name="Google API Key",
        description="API Key for Google Generative AI Model"
    )
)

# Model for extraction step in validation (Using the configurable OpenAI model)
validation_extraction_model = openai_model

# REMOVED global OpenAI client for embeddings
# openai_embedding_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

# Get database metadata
DATABASE_METADATA = get_database_metadata()

# Initial Planner Agent
initial_planner_prompt = ChatPromptTemplate.from_template("""
你是一个专业的数据库意图解析器和向量查询规划师。
你的首要任务是判断用户输入是否包含明确的数据库查询意图。
只要用户提及metadata（比如品牌，城市，榜单，餐厅，食物）相关的问题，就是有明确查询意图。
但请合理猜测一些你不知道的名词其实为餐厅的品牌名。
请推测是否需要图片去辅助用户的查询意图。

数据库元数据:
{metadata}

用户输入: {query}

输出格式如下：

```json
{{
  "has_query_intent": boolean, // true表示有明确查询意图, false表示没有
  "analysis": "...", // 如果has_query_intent为true，则提供详细分析；如果为false，则说明原因
  "identified_brand": "<提取到的品牌名称 或 null>", // 新增：明确提取的品牌
  "identified_city": "<提取到的城市名称 或 null>", // 新增：明确提取的城市
  "identified_list": "<提取到的榜单/品类名称 或 null>", // 新增：明确提取的榜单/品类
  "vector_search_query_suggestion": "<建议的向量搜索文本 或 null>" // 如果分析表明需要搜索帖子内容（如评价、氛围、特定描述），则提供一个简洁、关键词丰富的搜索语句；否则为 null
}}
```

如果`has_query_intent`为true，请在`analysis`字段中包含以下内容:
1. 用户意图解析: 用户想要查询什么信息？
2. 需要查询的表和列: 哪些表和列包含相关信息？指出哪些信息可能需要从`posts.content`中通过向量搜索获取。
3. 查询条件: 需要应用哪些过滤条件（包括品牌、城市、榜单等）？是否需要最新数据？
4. **明确指出查询中提到的城市名称 (如果存在)，并说明它对应 `dzdpdata.城市` 列。**
5. 把食物品类理解为榜单名，比如海鲜，火锅，烤串，烧烤，甜品，咖啡，小吃，，烤肉，川菜等，都可以理解为榜单名。如果没有提及榜单，就默认使用主榜单查询。

比如：
用户："上海的火锅榜前3名是谁？评分和人均价格呢？"
输出："用户想要查询上海火锅榜单前3名的评分和人均价格，这些信息都可以在dzdpdata table里找到。上海应该在城市列 (对应 `dzdpdata.城市`)，火锅应该在榜单列，评分可能在评分列..." 
                                                          
用户："北京烤鸭在小红书上哪家牛逼？"
输出："用户想要查询北京烤鸭在小红书上牛逼的店，小红书上牛逼可以理解为在小红书上表现出色，应该可以用点赞多来衡量。这些信息可以在dzdpdata table和posts table里找到，北京可能在城市列，烤鸭可能在榜单列，点赞可能在likes列。"

用户："查找长沙主榜单相关的餐厅，要求环境优雅" 
输出："用户想要查找长沙主榜单的餐厅，并且要求环境优雅。这些应该可以在dzdp和posts找到，长沙应该在城市列，主榜单应该在榜单列，环境优雅应该在content里面找到。因为在content，所以应该用向量搜索。"

**向量查询建议 (`vector_search_query_suggestion`) 生成指南**:
- 当用户查询涉及主观评价（口碑、评价如何）、氛围、环境、特定菜品描述等可能存在于`posts.content`中的信息时，生成此建议。
- 建议应包含核心实体（如品牌名）和关键描述词（如 评价、环境优雅、服务好）。
- 例如：用户问"上海三出山在小红书上的评价如何"，建议可以是 "上海 三出山 评价"。
- 例如：用户问"查找环境优雅的长沙主榜单餐厅帖子"，建议可以是 "长沙 餐厅 环境优雅"。（榜单不太可能被小红书帖子提及）
- 如果查询只涉及结构化数据（排名、价格、地址等），则此字段为 null。     


确保只输出JSON格式的响应。
""")

def initial_planner(state: Dict[str, Any]) -> Dict[str, Any]:
    """Initial planner: Analyzes query, detects intent, suggests vector query if needed."""
    query = state.get("query", "")
    print("--- Inside initial_planner ---") # Added debug print
    print(f"Received query: {query}") # Added debug print

    messages = initial_planner_prompt.format_messages(
        query=query,
        metadata=DATABASE_METADATA
    )
    
    print("Formatted prompt messages. Attempting to invoke Gemini model...") # Added debug print

    response = None # Initialize response
    llm_error = None # Variable to store potential LLM error
    try:
        # This invoke call will now use the configured API key if provided
        response = gemini_model.invoke(messages)
        print("Gemini model invoked successfully.") # Added debug print
    except Exception as e:
        print(f"ERROR invoking Gemini model: {e}") # Added explicit error logging
        llm_error = f"Error calling LLM: {str(e)}"

    # Default values
    has_intent = False
    analysis = "无法解析模型响应或模型调用失败。"
    vector_suggestion = None
    identified_brand = None
    identified_city = None
    identified_list = None
    
    # Add LLM error to analysis if it occurred
    if llm_error:
        analysis = llm_error

    # Process response only if the call was successful
    if response:
        print(f"DEBUG: Raw LLM response content:\n---\n{response.content}\n---") # <-- 添加这行日志
        try:
            content_str = response.content.strip()
            if content_str.startswith("```json"):
                content_str = content_str[7:]
            if content_str.endswith("```"):
                content_str = content_str[:-3]
            content_str = content_str.strip()

            response_json = json.loads(content_str)
            has_intent = response_json.get("has_query_intent", False)
            analysis = response_json.get("analysis", "未能解析意图")
            vector_suggestion = response_json.get("vector_search_query_suggestion")
            identified_brand = response_json.get("identified_brand")
            identified_city = response_json.get("identified_city")
            identified_list = response_json.get("identified_list")
            print("Successfully parsed JSON response.") # Added debug print

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from initial_planner: {e}")
            # Use the raw content in analysis if JSON parsing fails but LLM call succeeded
            analysis = f"LLM响应成功但JSON解析失败: {response.content}"
            # Keep other defaults
    else:
        print("Skipping response processing due to LLM invocation failure.") # Added debug print

    print(f"Returning state from initial_planner: has_intent={has_intent}") # Added debug print
    return {
        "query": query,
        "has_query_intent": has_intent,
        "initial_plan": analysis,
        "identified_brand": identified_brand,
        "identified_city": identified_city,
        "identified_list": identified_list,
        "vector_search_query_suggestion": vector_suggestion
    }

# Validation Planner Agent
def validation_planner(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validation planner: Reads values from state, programmatically calls tools, determines query type and SQL filter."""
    initial_plan = state.get("initial_plan", "") # Still useful for context potentially
    vector_suggestion_from_plan = state.get("vector_search_query_suggestion")

    # Step 1: Get values directly from state (NO LLM extraction needed)
    # REMOVED LLM Extraction Call
    # extraction_messages = validation_planner_extraction_prompt.format_messages(...)
    # extraction_response = validation_extraction_model.invoke(extraction_messages)
    # ... parsing logic removed ...

    # For now, let's assume brand extraction logic might need separate handling if required.
    # If initial_planner *also* provides identified_brand, we can use state.get("identified_brand") here too.
    brand_to_validate = state.get("identified_brand") # Get brand from state
    city_to_validate = state.get("identified_city")
    list_to_validate = state.get("identified_list")
    vector_search_needed = bool(vector_suggestion_from_plan)
    
    print(f"Debug: Values read from state for validation: Brand='{brand_to_validate}', City='{city_to_validate}', List='{list_to_validate}'")

    # Step 2: Validate values and build SQL filter
    validation_results_text = []
    sql_filter_parts = []
    brand_exists = False

    if brand_to_validate:
        try:
            exists = validate_value_exists.invoke({"table_name": 'brand', "column_name": 'name', "value": brand_to_validate})
            validation_results_text.append(f"- 品牌 '{brand_to_validate}' 存在状态: {exists}")
            if exists:
                brand_exists = True
                # --- Refactor SQL escaping and f-string --- 
                # 1. Escape the brand name for SQL
                escaped_brand_name = brand_to_validate.replace("'", "''")
                # 2. Append the filter using the escaped value in a simpler f-string
                sql_filter_parts.append(f"b.name = '{escaped_brand_name}'")
        except Exception as e:
            print(f"Error invoking validate_value_exists for brand: {e}")
            validation_results_text.append(f"- 品牌 '{brand_to_validate}' 验证出错: {e}")

    if city_to_validate:
        try:
            exists = validate_value_exists.invoke({"table_name": 'dzdpdata', "column_name": '城市', "value": city_to_validate})
            validation_results_text.append(f"- 城市 '{city_to_validate}' 存在状态: {exists}")
            # if exists:
            #      sql_filter_parts.append(f"p.城市 = '{city_to_validate.replace("'", "''")}'") # REMOVED: Do not add city to vector filter
        except Exception as e:
            print(f"Error invoking validate_value_exists for city: {e}")
            validation_results_text.append(f"- 城市 '{city_to_validate}' 验证出错: {e}")
            
    if list_to_validate:
        try:
            exists = validate_value_exists.invoke({"table_name": 'dzdpdata', "column_name": '榜单', "value": list_to_validate})
            validation_results_text.append(f"- 榜单 '{list_to_validate}' 存在状态: {exists}")
            # if exists:
            #      sql_filter_parts.append(f"p.榜单 = '{list_to_validate.replace("'", "''")}'") # REMOVED: Do not add list to vector filter
        except Exception as e:
            print(f"Error invoking validate_value_exists for list: {e}")
            validation_results_text.append(f"- 榜单 '{list_to_validate}' 验证出错: {e}")

    validation_summary = "验证摘要:\n" + "\n".join(validation_results_text) if validation_results_text else "无需验证特定值。"
    sql_filter_string = " AND ".join(sql_filter_parts) if sql_filter_parts else None

    # Step 3: Determine Query Type
    if vector_search_needed:
        if brand_exists: # If the primary filter (brand) exists, do sql+vector
            query_type = "sql+vector"
        else: # Otherwise, just vector search
            query_type = "vector"
    else: # No vector needed
        query_type = "sql"

    print(f"Debug: Determined query type after validation: {query_type}")
    print(f"Debug: Generated SQL Filter: {sql_filter_string}")
    print(f"Debug: Final Validation Summary: {validation_summary}")

    return {
        **state,
        "validation_result": validation_summary,
        "query_type": query_type,
        "sql_filter": sql_filter_string # Add the filter string to state
    }

# SQL Generator Agent
sql_generator_prompt = ChatPromptTemplate.from_template("""
你是一个PostgreSQL专家，精通中文数据库的查询。你的任务是基于验证后的计划生成SQL查询。

数据库元数据:
{metadata}

初始计划:
{initial_plan}

验证结果:
{validation_result}

请生成精确的SQL查询来获取所需数据：
1. 根据initial_plan，确定需要查询的表和列以及查询条件。
2. 只查询validation_result中确认存在的变量。
3. 不要进行向量搜索。
4. 如果有榜单和品牌，用模糊查询。
                                                        
请只返回完整的SQL查询语句，不要包含其他解释。
""")

def sql_generator(state: Dict[str, Any]) -> Dict[str, Any]:
    """SQL generator that creates and executes SQL queries."""
    initial_plan = state.get("initial_plan", "")
    validation_result = state.get("validation_result", "")
    query_type = state.get("query_type", "sql")

    messages = sql_generator_prompt.format_messages(
        initial_plan=initial_plan,
        validation_result=validation_result,
        metadata=DATABASE_METADATA
    )

    response = gemini_model.invoke(messages)

    # Extract SQL query from response
    sql_query = response.content.strip()
    # Remove potential markdown backticks
    if sql_query.startswith("```sql"):
        sql_query = sql_query[6:]
    if sql_query.endswith("```"):
        sql_query = sql_query[:-3]
    sql_query = sql_query.strip()

    return {
        **state,
        "sql_query": sql_query
    }

# Vector Generator Agent (Revised: Creates client dynamically using key from state)
def vector_generator(state: Dict[str, Any]) -> Dict[str, Any]:
    """Vector generator: Takes suggested text, generates embedding using key from state."""
    vector_suggestion = state.get("vector_search_query_suggestion")
    query = state.get("query") # Fallback to original query
    # --- Get OpenAI key from state --- 
    openai_api_key_from_state = state.get("openai_api_key")
    # print(f"Debug vector_generator: OpenAI Key from state: {'******' if openai_api_key_from_state else 'None'}")

    search_text = vector_suggestion if vector_suggestion else query
    embedding_vector_str = None

    if search_text:
        # --- Create OpenAI client dynamically --- 
        try:
            # Use key from state if available, otherwise None (client defaults to env var)
            # Note: If state has None AND env var is not set, this WILL fail.
            # Ensure OPENAI_API_KEY env var is set in the container as a fallback.
            dynamic_openai_client = OpenAI(api_key=openai_api_key_from_state)
            
            print(f"Debug: Generating embedding for text: {search_text}")
            response = dynamic_openai_client.embeddings.create(
                model=EMBEDDING_MODEL_NAME,
                input=search_text
            )
            embedding = response.data[0].embedding
            embedding_vector_str = str(embedding)
            print(f"Debug: Generated embedding vector (first few dims): {embedding[:5]}...")
        except Exception as e:
            print(f"Error creating embedding for text '{search_text}': {str(e)}")
            # Log which key source was attempted (if possible without exposing key)
            if openai_api_key_from_state:
                print("Debug: Attempted using OpenAI key provided in the request/state.")
            else:
                print("Debug: Attempted using OpenAI key from environment variable (not provided in request/state).")
            # Decide how to handle embedding errors
            pass 

    return {
        **state,
        "vector_search_text": search_text, 
        "embedding_vector_str": embedding_vector_str 
    }

# Synthesizer Agent
synthesizer_prompt = ChatPromptTemplate.from_template("""
你是一个数据分析和结果综合专家。你的任务是基于所有收集到的信息，为用户提供清晰、全面的回答。

用户原始查询: {query}

请分析以下信息，此部分不是回答：
- **初始计划/分析结果**: {initial_plan}
- **查询意图判断**: {has_query_intent}
- **验证结果** (如果执行了): {validation_result}
- **SQL查询** (如果执行了): {sql_query}
- **SQL查询结果** (如果执行了): {sql_results}
- **向量搜索文本** (如果执行了): {vector_search_text}
- **向量搜索结果** (如果执行了): {vector_results}

根据以上信息，执行以下回复操作：
1. **如果 `has_query_intent` 为 false**: 生成一个友好的回应，解释为什么没有执行查询，并询问用户是否需要数据库相关的帮助。例如："您好！我是一个数据库查询助手。如果您有关于餐厅、排名或相关帖子的查询，请告诉我。" 或者直接使用 `initial_plan` 中的解释。
2. **如果 `has_intent` 为 true**，你的最终回应应该是: 
   a. 综合SQL结果、向量搜索结果。提供对用户查询的直接回答。如果是一列数据，按照用户的查询意图进行sorting。
   b. 如果用户意图有查询图片，请在回复中提供图片链接。
   c. 如果用户想要看特定帖子，请将所有帖子相关的信息展示包括图片。
   d. 指出任何结果的限制或注意事项。
   e. 猜测用户下一步想要提问什么，并给出建议。
   f. 回复中避免使用markdown和*号。
   g. 使用emoji让回答更生动。

回答风格：
1. 第一人称回答用户问题，使用轻松，尊重的口吻。                                                      
2. 不要使用数据库，向量搜索等专业词汇。
""")

def synthesizer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesizer that creates the final response."""
    query = state.get("query", "")
    has_query_intent = state.get("has_query_intent", False)
    initial_plan = state.get("initial_plan", "")
    validation_result = state.get("validation_result", "")
    sql_query = state.get("sql_query", "")
    sql_results = state.get("sql_results", "")
    vector_search_text = state.get("vector_search_text", "")
    vector_results = state.get("vector_results", "")

    # 准备传递给prompt的变量，处理可能不存在的情况
    prompt_input = {
        "query": query,
        "initial_plan": initial_plan,
        "has_query_intent": str(has_query_intent), # Prompt需要字符串
        "validation_result": validation_result if validation_result else "未执行",
        "sql_query": sql_query if sql_query else "未执行",
        "sql_results": sql_results if sql_results else "未执行",
        "vector_search_text": vector_search_text if vector_search_text else "未执行",
        "vector_results": vector_results if vector_results else "未执行",
    }

    messages = synthesizer_prompt.format_messages(**prompt_input)

    response = gemini_model.invoke(messages)

    return {
        **state,
        "final_response": response.content
    } 