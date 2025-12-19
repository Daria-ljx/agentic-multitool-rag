from typing import Optional
from langchain.chat_models import init_chat_model
from src.utils.constants import LLM_MODEL_NAME

class SummarizerTool:
    """
    Lightweight tool wrapper (not BaseTool) to avoid Pydantic hassles.
    Has a .run(text) method.
    """
    def __init__(self, model: str = LLM_MODEL_NAME, temperature: float = 0.0):
        self.llm = init_chat_model(model=model, temperature=temperature)

    def run(self, text: str, max_bullets: int = 8) -> str:
        prompt = (
            "Summarize the content into concise bullet points.\n"
            f"Max bullets: {max_bullets}\n\n"
            "Content:\n"
            f"{text}"
        )
        return self.llm.invoke(prompt).content.strip()
