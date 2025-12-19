import sys
import os
import traceback
import streamlit as st

# 确保可以 import src.*
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.agent.run_agent import run_agent
from src.utils.constants import APP_NAME

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption(
    "Lijiaxin - Multi-tool Agentic AI System (LangGraph + RAG + Web Search + Summarization)"
)

# 初始化对话历史
if "history" not in st.session_state:
    st.session_state["history"] = []

# === 显示历史对话 ===
for msg in st.session_state["history"]:
    role = msg["role"]
    content = msg["content"]

    with st.chat_message(role):
        st.markdown(content)

# === 输入框（ChatGPT 风格）===
user_query = st.chat_input("Ask something...")

if user_query:
    # 1️⃣ 显示用户消息
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_query)

    try:
        # 2️⃣ 调用 Agent
        with st.spinner("Agent is thinking..."):
            state = run_agent(
                user_query,
                history=st.session_state["history"]
            )

        # 3️⃣ 取结果
        answer = state.get("final_answer") or state.get("draft_answer") or ""
        tool_plan = state.get("tool_plan")

        # 4️⃣ 更新历史
        st.session_state["history"] = state.get("history", [])

        # 5️⃣ 显示 Assistant 回复
        with st.chat_message("assistant", ):
            st.markdown(answer)

    except Exception:
        st.error("Agent failed.")
        st.code(traceback.format_exc())
