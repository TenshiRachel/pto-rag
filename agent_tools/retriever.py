from langchain.tools import StructuredTool
import time
import json
from typing import Dict, Any, Optional, List


class RetrieverTool:
    def __init__(self, faiss_store, k: int = 12, cache: Optional[Dict[str, Any]] = None):
        """
        faiss_store: your FAISS vector store
        k: how many chunks to retrieve
        cache: shared dict for caching retrieval results per normalized query
        """
        self.retriever = faiss_store.as_retriever(search_kwargs={"k": k})
        self.cache = cache if cache is not None else {}
        self.last_hit = False  # for instrumentation

    def _normalize_query(self, q: str) -> str:
        # normalize so "opex yoy" and "OPEX YOY  " hash the same
        return " ".join(q.strip().lower().split())

    def forward(self, query: str) -> str:
        """
        Returns formatted retrieved context for the LLM.
        Uses cache if available.
        """
        norm_q = self._normalize_query(query)
        start = time.time()
        if norm_q in self.cache:
            # cache hit
            self.last_hit = True
            result = self.cache[norm_q]
        else:
            # cache miss: actually retrieve from FAISS
            self.last_hit = False
            docs = self.retriever.get_relevant_documents(query)
            # format results into a big string with metadata
            chunks = []
            for d in docs:
                meta = d.metadata if hasattr(d, "metadata") else {}
                chunks.append(
                    json.dumps(
                        {
                            "content": d.page_content,
                            "metadata": meta,
                        },
                        ensure_ascii=False
                    )
                )
            result = "\n\n".join(chunks)
            # store in cache
            self.cache[norm_q] = result
        end = time.time()
        # (optional) store how long this call took for debugging
        self.last_call_duration = end - start
        return result

    def as_tool(self) -> StructuredTool:
        """Convert to a LangChain StructuredTool usable by OpenAI agents."""
        return StructuredTool.from_function(
            func=self.forward,
            name="retriever",
            description="Retrieves semantically similar documents from a FAISS vector index given a natural language query.",
        )
