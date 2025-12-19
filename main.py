# import sys, os, traceback
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if ROOT not in sys.path:
#     sys.path.append(ROOT)
from dotenv import load_dotenv
load_dotenv()
import traceback
from src.agent.run_agent import run_agent


def main():
    history = []
    print("Multi-Tool Agentic RAG (type 'exit' to quit)\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        try:
            state = run_agent(q, history=history)
            history = state.get("history", history)
            ans = state.get("final_answer") or state.get("draft_answer") or ""
            # draw_mermaid()
            print(f"\nAssistant: {ans}\n")
            print(f"[tool_plan] {state.get('tool_plan')}\n")
        except Exception:
            print(traceback.format_exc())

if __name__ == "__main__":
    main()
