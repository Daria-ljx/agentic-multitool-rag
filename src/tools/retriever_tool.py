from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from src.rag.embeddings import get_embeddings
from src.utils.constants import VECTORSTORE_PATH

def build_retriever_tool(k: int = 4):
    """
    Create a retriever tool using LangChain's built-in factory function.
    This avoids dealing with BaseTool + Pydantic v2 complexities.
    """

    # Load embeddings
    embeddings = get_embeddings()

    # Load FAISS vectorstore
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Convert vectorstore to retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Create built-in LangChain retriever tool
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="document_retriever",
        description=(
            "Use this tool to retrieve relevant context from the FAISS vectorstore. "
            "Input should be a search query."
        )
    )

    return retriever_tool