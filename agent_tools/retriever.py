from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain.tools import StructuredTool

class RetrieverTool:
    """
    A FAISS-backed retriever with optional caching and dynamic-K.

    - If `cache` is provided and has the query, returns cached docs and updates diagnostics.
    - If cache-miss:
        * Use the agent-provided `k` if present.
        * Otherwise, infer k heuristically from the query text (e.g., "last 5 quarters" -> 5).
        * Fallback to `default_k` if nothing inferred.
      Then run similarity search, update diagnostics, and store in cache (docs + metadata).
    - Always sets:
        * self.last_pairs: List[(report, page)]
        * self.last_k_used: int
        * self.last_hit: bool  (True when served from cache)
    """

    def __init__(
        self,
        faiss_store,
        default_k: int = 12,
        cache: Optional[Dict[str, List[Document]]] = None,
        use_dynamic_k: bool = False,
        min_k: int = 1,
        max_k: int = 30,
    ):
        self.faiss_store = faiss_store
        self.default_k = default_k
        self.cache = cache
        self.use_dynamic_k = use_dynamic_k
        self.min_k = min_k
        self.max_k = max_k

        # diagnostics for benchmarking
        self.last_pairs: List[Tuple[Optional[str], Optional[int]]] = []
        self.last_k_used: Optional[int] = None
        self.last_hit: bool = False

    # ---------- internal helpers ----------

    def _infer_k_from_query(self, query: str) -> Optional[int]:
        """
        Heuristic: pick up counts from common phrasings, e.g.:
        - "last 5 quarters", "last 3 fiscal years", "last 4 years"
        - "top 8 pages", "top 6"
        """
        patterns = [
            r"last\s+(\d+)\s+(?:fiscal\s+)?(years?|quarters?)",
            r"top\s+(\d+)\b",
            r"last\s+(\d+)\b",
        ]
        for pat in patterns:
            m = re.search(pat, query, flags=re.IGNORECASE)
            if m:
                try:
                    val = int(m.group(1))
                    return max(self.min_k, min(self.max_k, val))
                except Exception:
                    continue
        return None

    def _pairs_from_docs(self, docs: List[Document]) -> List[Tuple[Optional[str], Optional[int]]]:
        pairs = []
        for d in docs:
            md = getattr(d, "metadata", {}) or {}
            report = md.get("report") or md.get("source") or md.get("document_id")
            page = md.get("page")
            # normalize page to int if possible
            try:
                if page is not None:
                    page = int(page)
            except Exception:
                page = None
            pairs.append((report, page))
        # remove duplicates while preserving order
        seen = set()
        deduped = []
        for p in pairs:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        return deduped

    # ---------- main retrieval ----------

    def forward(self, query: str, k: Optional[int] = None) -> List[Document]:
        # cache hit path
        if self.cache is not None and query in self.cache:
            docs = self.cache[query]
            self.last_hit = True
            self.last_pairs = self._pairs_from_docs(docs)
            self.last_k_used = len(docs)
            return docs

        # dynamic K (only when enabled and no explicit k from agent)
        used_k = None
        if self.use_dynamic_k:
            used_k = k if k is not None else self._infer_k_from_query(query)

        if used_k is None:
            used_k = self.default_k  # fallback

        # clamp
        used_k = max(self.min_k, min(self.max_k, int(used_k)))

        # do the actual retrieval
        docs = self.faiss_store.similarity_search(query, k=used_k)

        # update diagnostics
        self.last_hit = False
        self.last_pairs = self._pairs_from_docs(docs)
        self.last_k_used = used_k

        # store to cache (preserve metadata!) for subsequent hits
        if self.cache is not None:
            self.cache[query] = docs

        return docs

    # ---------- tool wrapper ----------

    class _Input(BaseModel):
        request: str = Field(
            ...,
            description=(
                "Full retrieval request. Format it like: "
                "'query=<the finance question>; k=<optional int>'. "
                "If k is not provided, dynamic K will be inferred automatically."
            ),
        )

    def _parse_request(self, request: str) -> Tuple[str, Optional[int]]:
        """
        Expect something like:
        'query=Report the Gross Margin over the last 5 quarters; k=5'
        or just:
        'query=Show Operating Expenses for the last 3 fiscal years'
        We'll pull out the query text and optional k.
        """
        # default
        q_text = request
        k_val = None

        # try to split on ';'
        parts = [p.strip() for p in request.split(";")]

        for p in parts:
            if p.lower().startswith("query="):
                q_text = p[len("query="):].strip()
            elif p.lower().startswith("k="):
                try:
                    k_val = int(p[len("k="):].strip())
                except:
                    k_val = None

        return q_text, k_val

    def _tool_func(self, request: str) -> str:
        query_text, k_hint = self._parse_request(request)
        self.forward(query_text, k=k_hint)
        return f"retrieved_k={self.last_k_used}; pairs={self.last_pairs[:6]}"

    def as_tool(self):
        return StructuredTool.from_function(
            name="retriever",
            func=self._tool_func,
            args_schema=self._Input,
            description=(
                "Retrieve finance evidence snippets from NVIDIA filings. "
                "You MUST call it at most once. "
                "Use format: 'query=<your question>; k=<optional int>'. "
                "If k is omitted, pick k based on the question (e.g. 'last 5 quarters' -> 5)."
            ),
        )
