from typing import TypedDict, Optional, List, Dict, Any

class GraphState(TypedDict):
    query: str
    history: List[Dict[str, str]]

    # Router output
    tool_plan: Optional[List[str]]     # e.g. ["rag"], ["web"], ["rag","web","summarize"]

    # RAG artifacts
    rag_context: Optional[str]

    # Web artifacts
    web_context: Optional[str]

    # Aggregation
    merged_context: Optional[str]
    retrieval_score: Optional[float]

    # Answering
    draft_answer: Optional[str]
    critique: Optional[str]
    final_answer: Optional[str]

