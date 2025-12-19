import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from src.utils.logger import get_logger
from .rag_graph import build_graph

logger = get_logger(__name__)
load_dotenv()

# LangSmith (optional)
# If user sets these in .env, tracing works automatically.
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))
os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "MultiTool-Agentic-RAG"))

_graph = build_graph()

def run_agent(query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    if history is None:
        history = []

    init_state = {
        "query": query,
        "history": history,
        "tool_plan": None,
        "rag_context": None,
        "web_context": None,
        "merged_context": None,
        "retrieval_score": None,
        "draft_answer": None,
        "critique": None,
        "final_answer": None,
    }

    logger.info(f"[run_agent] query={query}")
    result = _graph.invoke(init_state)
    return result
