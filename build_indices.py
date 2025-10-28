from langchain_community.vectorstores import FAISS


def build_faiss_index(chunked_docs, embedding_model):
    embedding_model = embedding_model

    # FAISS Vector Index
    faiss_store = FAISS.from_documents(chunked_docs, embedding_model)
    faiss_store.save_local("faiss_index")

    # Load FAISS Vector Store
    faiss_store = FAISS.load_local(
        "faiss_index",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    return faiss_store
