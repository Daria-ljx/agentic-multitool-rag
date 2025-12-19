ROUTER_PROMPT = """
You are a tool-routing agent.

Choose which tools to use based on the user query.

Available tools:
- rag: search internal documents
- web: search the web for current/general facts
- summarize: summarize long tool outputs to key points

Rules:
- If the question is about the ingested documents, include "rag".
- If it needs up-to-date or external knowledge, include "web".
- If the retrieved contexts are long or the user asks for a summary, include "summarize".
- Output ONLY a JSON array of tool names, like ["rag"] or ["rag","web"].

User query:
{query}

JSON:
"""

ANSWER_PROMPT = """
You are a helpful assistant.

Conversation history:
{history}

User query:
{query}

Context (may include internal docs + web results):
{context}

Answer clearly and concisely. If context is insufficient, say you are not sure.
"""

CRITIC_PROMPT = """
You are a strict reviewer. Identify issues in the draft answer.

User query:
{query}

Context:
{context}

Draft answer:
{draft}

Return:
- Potential hallucinations or unsupported claims
- Missing important info
- Suggested improvements
"""

REFINE_PROMPT = """
You are improving the draft answer using critique.

User query:
{query}

Context:
{context}

Draft answer:
{draft}

Critique:
{critique}

Write the improved final answer.
"""
