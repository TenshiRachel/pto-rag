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

import pickle
from rank_bm25 import BM25Okapi
from typing import List
import re

def simple_tokenize(text: str):
    # Lowercase + split on alphanumeric boundaries
    return re.findall(r"\w+", text.lower())

class BM25Index:
    def __init__(self, docs, ids):
        self.docs = docs                 # list of raw text strings
        self.ids = ids                   # chunk IDs
        self.tokenized_docs = [simple_tokenize(doc) for doc in docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query, top_k=10):
        query_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked

    def save(self, path: str):
        data = {
            "docs": self.docs,
            "ids": self.ids,
            "tokenized_docs": self.tokenized_docs
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(docs=data["docs"], ids=data["ids"])
        obj.tokenized_docs = data["tokenized_docs"]
        obj.bm25 = BM25Okapi(obj.tokenized_docs)
        return obj
