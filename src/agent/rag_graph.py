# from langgraph.graph import StateGraph
# from .graph_state import GraphState
# from .nodes import (
#     route_tools,
#     run_rag,
#     run_web_search,
#     merge_context,
#     summarize_context,
#     generate_draft,
#     critic,
#     refine,
#     update_memory,
#     needs_rag,
#     needs_web,
#     needs_summarize,
# )
#
# def build_graph():
#     g = StateGraph(GraphState)
#
#     g.add_node("router", route_tools)
#     g.add_node("rag", run_rag)
#     g.add_node("web", run_web_search)
#     g.add_node("merge", merge_context)
#     g.add_node("summarize", summarize_context)
#     g.add_node("draft", generate_draft)
#     g.add_node("critic", critic)
#     g.add_node("refine", refine)
#     g.add_node("memory", update_memory)
#
#     g.set_entry_point("router")
#
#     # Router -> conditional branches
#     # We'll route to rag/web (possibly both), then merge.
#     def route_after_router(state):
#         # If both needed, go rag first then web then merge (deterministic)
#         if needs_rag(state) and needs_web(state):
#             return "rag"
#         if needs_rag(state):
#             return "rag"
#         if needs_web(state):
#             return "web"
#         # default to rag
#         return "rag"
#
#     g.add_conditional_edges("router", route_after_router, {
#         "rag": "rag",
#         "web": "web",
#     })
#
#     # After rag: if also needs web -> web else merge
#     def after_rag(state):
#         return "web" if needs_web(state) else "merge"
#
#     g.add_conditional_edges("rag", after_rag, {
#         "web": "web",
#         "merge": "merge",
#     })
#
#     # After web -> merge
#     g.add_edge("web", "merge")
#
#     # After merge: summarize optionally
#     def after_merge(state):
#         return "summarize" if needs_summarize(state) else "draft"
#
#     g.add_conditional_edges("merge", after_merge, {
#         "summarize": "summarize",
#         "draft": "draft",
#     })
#
#     g.add_edge("summarize", "draft")
#     g.add_edge("draft", "critic")
#     g.add_edge("critic", "refine")
#     g.add_edge("refine", "memory")
#
#     app = g.compile()
#     return app



from langgraph.graph import StateGraph, START
from src.agent.graph_state import GraphState
from src.agent.nodes import (
    route_tools,
    run_rag,
    run_web_search,
    merge_context,
    summarize_context,
    generate_draft,
    critic,
    refine,
    update_memory,
    needs_rag,
    needs_web,
    needs_summarize,
)

# ========== NEW: Entry / Init Node ==========
def init_state(state: GraphState) -> GraphState:
    """
    Entry node for Agent Chat UI.

    Agent Chat UI will call:
        graph.invoke({ "query": "..." })

    This node is responsible for:
    - initializing missing fields
    - normalizing the input state
    """
    return {
        "query": state.get("query"),
        "history": state.get("history", []),

        "tool_plan": None,
        "rag_context": None,
        "web_context": None,
        "merged_context": None,
        "retrieval_score": None,
        "draft_answer": None,
        "critique": None,
        "final_answer": None,
    }


def build_graph():
    g = StateGraph(GraphState)

    # -------- Nodes --------
    g.add_node("init", init_state)
    g.add_node("router", route_tools)
    g.add_node("rag", run_rag)
    g.add_node("web", run_web_search)
    g.add_node("merge", merge_context)
    g.add_node("summarize", summarize_context)
    g.add_node("draft", generate_draft)
    g.add_node("critic", critic)
    g.add_node("refine", refine)
    g.add_node("memory", update_memory)

    # -------- Entry --------
    g.add_edge(START, "init")
    g.add_edge("init", "router")

    # -------- Router logic --------
    def route_after_router(state: GraphState):
        if needs_rag(state) and needs_web(state):
            return "rag"
        if needs_rag(state):
            return "rag"
        if needs_web(state):
            return "web"
        return "rag"

    g.add_conditional_edges(
        "router",
        route_after_router,
        {
            "rag": "rag",
            "web": "web",
        },
    )

    # After rag
    def after_rag(state: GraphState):
        return "web" if needs_web(state) else "merge"

    g.add_conditional_edges(
        "rag",
        after_rag,
        {
            "web": "web",
            "merge": "merge",
        },
    )

    # Web → merge
    g.add_edge("web", "merge")

    # Merge → summarize or draft
    def after_merge(state: GraphState):
        return "summarize" if needs_summarize(state) else "draft"

    g.add_conditional_edges(
        "merge",
        after_merge,
        {
            "summarize": "summarize",
            "draft": "draft",
        },
    )

    g.add_edge("summarize", "draft")
    g.add_edge("draft", "critic")
    g.add_edge("critic", "refine")
    g.add_edge("refine", "memory")

    return g.compile()

