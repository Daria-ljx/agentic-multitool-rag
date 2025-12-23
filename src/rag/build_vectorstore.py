from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.rag.embeddings import get_embeddings
from langchain_community.vectorstores import FAISS
from src.data.clean_text import clean_text
from src.data.load_docs import load_documents
from src.utils.constants import VECTORSTORE_PATH

def build_vectorstore():
    docs = load_documents()

    # Clean text
    for d in docs:
        d.page_content = clean_text(d.page_content)

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    print(f"Generated {len(chunks)} chunks.")

    # Build vectorstore
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTORSTORE_PATH)
    print("FAISS vectorstore saved.")

    return vectorstore

if __name__ == "__main__":
    build_vectorstore()
