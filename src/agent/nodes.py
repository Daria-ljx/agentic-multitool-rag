import json
from src.utils.logger import get_logger
from langchain_openai import ChatOpenAI
from src.utils.constants import LLM_MODEL_NAME
from src.tools.retriever_tool import build_retriever_tool
from src.tools.web_search_tool import build_web_search_tool
from src.tools.summarizer_tool import SummarizerTool
from .prompts import ROUTER_PROMPT, ANSWER_PROMPT, CRITIC_PROMPT, REFINE_PROMPT

logger = get_logger(__name__)

llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.0)

rag_tool = build_retriever_tool(k=4)
web_tool = build_web_search_tool()
summ_tool = SummarizerTool(model=LLM_MODEL_NAME, temperature=0.0)


def _history_to_text(history):
    lines = []
    for m in history:
        lines.append(f"{m['role'].upper()}: {m['content']}")
    return "\n".join(lines)


# --- Router Node ---
def route_tools(state):
    query = state["query"]
    raw = llm.invoke(ROUTER_PROMPT.format(query=query)).content.strip()
    logger.info(f"[Router] raw={raw}")

    try:
        plan = json.loads(raw)
        if not isinstance(plan, list):
            plan = ["rag"]
    except Exception:
        plan = ["rag"]

    # basic normalization
    plan = [p.lower() for p in plan]
    allowed = {"rag", "web", "summarize"}
    plan = [p for p in plan if p in allowed]
    if not plan:
        plan = ["rag"]

    state["tool_plan"] = plan
    return state


# --- RAG Node ---
def run_rag(state):
    query = state["query"]
    logger.info("[RAG] running rag_search")
    context = rag_tool.run(query)
    state["rag_context"] = context
    return state


# --- Web Search Node ---
def run_web_search(state):
    query = state["query"]
    logger.info("[Web] running web search")

    try:
        results = web_tool.run(query)

        # Tavily 可能返回 list[dict]
        if isinstance(results, list):
            texts = []
            for r in results:
                if isinstance(r, dict):
                    title = r.get("title", "")
                    content = r.get("content", "")
                    url = r.get("url", "")
                    texts.append(f"- {title}\n{content}\nSource: {url}")
                else:
                    texts.append(str(r))
            web_text = "\n\n".join(texts)
        else:
            web_text = str(results)

    except Exception as e:
        logger.warning(f"[Web] Web search failed: {e}")
        web_text = "Web search unavailable."

    state["web_context"] = web_text
    return state



# --- Merge contexts ---
def merge_context(state):
    rag_ctx = (state.get("rag_context") or "").strip()
    web_ctx = (state.get("web_context") or "").strip()

    parts = []
    if rag_ctx:
        parts.append("## Internal Docs (RAG)\n" + rag_ctx)
    if web_ctx:
        parts.append("## Web Search\n" + web_ctx)

    merged = "\n\n".join(parts).strip()
    state["merged_context"] = merged
    return state


# --- Summarize (optional tool) ---
def summarize_context(state):
    merged = state.get("merged_context") or ""
    if not merged.strip():
        return state

    logger.info("[Summarize] summarizing merged context")
    summary = summ_tool.run(merged, max_bullets=10)
    state["merged_context"] = "## Summarized Context\n" + summary
    return state


# --- Answer Agent ---
def generate_draft(state):
    history = state.get("history", [])
    history_text = _history_to_text(history)
    query = state["query"]
    context = state.get("merged_context") or ""

    logger.info("[Answer] generating draft")
    draft = llm.invoke(
        ANSWER_PROMPT.format(history=history_text, query=query, context=context)
    ).content.strip()

    state["draft_answer"] = draft
    return state


# --- Critic Agent ---
def critic(state):
    query = state["query"]
    context = state.get("merged_context") or ""
    draft = state.get("draft_answer") or ""

    logger.info("[Critic] critiquing draft")
    critique = llm.invoke(
        CRITIC_PROMPT.format(query=query, context=context, draft=draft)
    ).content.strip()

    state["critique"] = critique
    return state


# --- Refiner Agent ---
def refine(state):
    query = state["query"]
    context = state.get("merged_context") or ""
    draft = state.get("draft_answer") or ""
    critique = state.get("critique") or ""

    logger.info("[Refine] refining answer")
    final = llm.invoke(
        REFINE_PROMPT.format(query=query, context=context, draft=draft, critique=critique)
    ).content.strip()

    state["final_answer"] = final
    return state


# --- Memory update ---
def update_memory(state):
    history = state.get("history", [])
    q = state["query"]
    a = state.get("final_answer") or state.get("draft_answer") or ""

    history = history + [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    state["history"] = history
    return state


# --- Conditional helpers for graph routing ---
def needs_rag(state):
    return "rag" in (state.get("tool_plan") or [])

def needs_web(state):
    return "web" in (state.get("tool_plan") or [])

def needs_summarize(state):
    return "summarize" in (state.get("tool_plan") or [])
