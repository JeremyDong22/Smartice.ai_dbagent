# logger.py
# 彩色日志工具，用于显示agent的思考和传递过程
# Created to provide colored log output for agent thinking and communication
# Updated log_agent_end to correctly show changed state

import os
import sys
from typing import Dict, Any

# ANSI颜色代码
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    
    # 前景色
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # 背景色
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

# Agent颜色映射
AGENT_COLORS = {
    "initial_planner": Colors.CYAN,
    "validation_planner": Colors.GREEN,
    "sql_generator": Colors.YELLOW,
    "vector_generator": Colors.MAGENTA,
    "execute_sql": Colors.BLUE,
    "execute_vector_search": Colors.RED,
    "synthesizer": Colors.WHITE
}

_last_state = {}

def log_agent_start(agent_name: str, state: Dict[str, Any] = None) -> None:
    """记录agent开始执行的日志"""
    global _last_state
    _last_state = state.copy() # Store the state before the agent runs
    color = AGENT_COLORS.get(agent_name, Colors.WHITE)
    print(f"\n{Colors.BOLD}{color}[开始] {agent_name.upper()}{Colors.RESET}")
    if state:
        print(f"{color}输入状态 (部分):{Colors.RESET}")
        # Only show relevant input state keys for brevity if needed
        relevant_keys = [k for k, v in state.items() if v is not None and v != ""] 
        for key in relevant_keys:
             print(f"{color}  {key}:{Colors.RESET} {truncate_text(str(state[key]))}")

def log_agent_end(agent_name: str, output_state: Dict[str, Any] = None) -> None:
    """记录agent完成执行的日志，仅显示新增或修改的键值"""
    global _last_state
    color = AGENT_COLORS.get(agent_name, Colors.WHITE)
    print(f"\n{Colors.BOLD}{color}[完成] {agent_name.upper()}{Colors.RESET}")
    if output_state:
        print(f"{color}状态变更:{Colors.RESET}")
        changed_keys = []
        for key, value in output_state.items():
            # Check if key is new or value has changed
            if key not in _last_state or _last_state[key] != value:
                 # Only log if value is meaningful (not None or empty string)
                if value is not None and value != "": 
                    print(f"{color}  {key}:{Colors.RESET} {truncate_text(str(value))}")
                    changed_keys.append(key)
        if not changed_keys:
            print(f"{color}  (状态无明显变化){Colors.RESET}")
    _last_state = {} # Clear last state after logging end

def log_router_decision(router_name: str, decision: str) -> None:
    """记录路由器决策的日志"""
    print(f"\n{Colors.BOLD}{Colors.BG_BLUE}[路由] {router_name}{Colors.RESET}")
    print(f"{Colors.BLUE}决策:{Colors.RESET} {decision}")

def truncate_text(text: str, max_length: int = 300) -> str:
    """截断长文本以便于日志显示"""
    if not text:
        return ""
    # Convert potential non-string types to string
    text_str = str(text)
    if len(text_str) <= max_length:
        return text_str
    return text_str[:max_length] + f"{Colors.BOLD}... (截断){Colors.RESET}"

def print_divider() -> None:
    """打印分隔线"""
    try:
        terminal_width = os.get_terminal_size().columns
    except OSError: # Handle cases where terminal size cannot be determined (e.g., in some CI/CD environments)
        terminal_width = 80
    print(f"\n{Colors.BOLD}{'-' * terminal_width}{Colors.RESET}")

def print_query_start(query: str) -> None:
    """打印查询开始信息"""
    try:
        terminal_width = os.get_terminal_size().columns
    except OSError:
        terminal_width = 80
    print(f"\n{Colors.BOLD}{Colors.BG_GREEN}" + "=" * terminal_width + f"{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}[新查询]{Colors.RESET} {query}")
    print(f"{Colors.BOLD}{Colors.BG_GREEN}" + "=" * terminal_width + f"{Colors.RESET}\n")

def print_response(response: str) -> None:
    """打印最终响应"""
    try:
        terminal_width = os.get_terminal_size().columns
    except OSError:
        terminal_width = 80
    print(f"\n{Colors.BOLD}{Colors.BG_MAGENTA}" + "=" * terminal_width + f"{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}[最终响应]{Colors.RESET}")
    print(f"{response}")
    print(f"{Colors.BOLD}{Colors.BG_MAGENTA}" + "=" * terminal_width + f"{Colors.RESET}\n") 