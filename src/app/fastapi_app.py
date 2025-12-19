from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional

from src.agent.run_agent import run_agent
from src.utils.constants import APP_NAME

app = FastAPI(title=APP_NAME)

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    history: List[Message] = []

class QueryResponse(BaseModel):
    answer: str
    tool_plan: Optional[List[str]] = None
    critique: Optional[str] = None
    history: List[Message]

@app.post("/query", response_model=QueryResponse)
async def query_agent(req: QueryRequest):
    state = run_agent(req.query, history=[m.model_dump() for m in req.history])
    answer = state.get("final_answer") or state.get("draft_answer") or ""
    history = state.get("history", [])
    return QueryResponse(
        answer=answer,
        tool_plan=state.get("tool_plan"),
        critique=state.get("critique"),
        history=[Message(**m) for m in history],
    )

# uvicorn src.app.fastapi_app:app --reload