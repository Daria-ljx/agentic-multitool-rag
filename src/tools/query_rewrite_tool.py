from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from src.utils.constants import LLM_MODEL_NAME

class QueryRewriteTool(BaseTool):
    name: str = "query_rewriter"
    description: str = "Rewrite user queries to improve retrieval accuracy."

    llm: ChatOpenAI = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.0)

    def _run(self, query: str) -> str:
        prompt = (
            "Rewrite the user query to be clearer and more useful for document retrieval.\n"
            f"Original query: {query}\n"
        )
        return self.llm.invoke(prompt).content.strip()

    async def _arun(self, query: str):
        raise NotImplementedError
