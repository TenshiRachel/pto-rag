from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain.tools import StructuredTool


class RetrieverTool:
    """Wrapper around FAISS retriever for LangChain/OpenAI tools interface."""

    def __init__(self, faiss_store: FAISS, top_k: int = 12):
        self.retriever: BaseRetriever = faiss_store.as_retriever(
            search_kwargs={"k": top_k}
        )

    def forward(self, query: str) -> str:
        """Retrieve top-k documents similar to the query."""
        assert isinstance(query, str), "Your search query must be a string"
        docs = self.retriever.invoke(query)

        if not docs:
            return "No relevant documents found."

        formatted_docs = "\n".join(
            [
                f"===== Document {i+1} =====\n"
                f"{doc.page_content}\n"
                f"Metadata: {doc.metadata}\n"
                for i, doc in enumerate(docs)
            ]
        )
        return formatted_docs

    def as_tool(self) -> StructuredTool:
        """Convert to a LangChain StructuredTool usable by OpenAI agents."""
        return StructuredTool.from_function(
            func=self.forward,
            name="retriever",
            description="Retrieves semantically similar documents from a FAISS vector index given a natural language query.",
        )
