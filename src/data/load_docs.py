import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader

def load_documents(data_dir="src/data/raw"):
    docs = []
    for file in os.listdir(data_dir):
        full_path = os.path.join(data_dir, file)
        if file.endswith(".txt"):
            loader = TextLoader(full_path)
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(full_path)
        else:
            continue

        docs.extend(loader.load())
    return docs

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.")
