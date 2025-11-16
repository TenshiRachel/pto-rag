from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain.tools import StructuredTool

from sentence_transformers import CrossEncoder

def infer_period_type(report: str) -> str:
    """
    Infer period type from a report name.
    Returns "year", "quarter", or "unknown".
    """
    if not report:
        return "unknown"

    r = report.lower()

    # Detect quarter-like patterns ANYWHERE (case-insensitive)
    # This matches: q1, q2, q3, q4, fy25q3, 2024q1, q3_2022, q1-fy23, etc.
    if re.search(r"q[1-4]", r) or "quarter" in r:
        return "quarter"

    # Detect year-level patterns:
    # FY23, FY2023, 2021, 2022, Annual
    if (
        re.search(r"fy\d{2}(?!\d)", r)          # FY23, FY24
        or re.search(r"fy20\d{2}", r)     # FY2023
        or re.search(r"\b20\d{2}\b", r)       # 2021, 2022
        or "annual" in r
        or "year" in r
    ):
        return "year"

    return "unknown"

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
        max_k: int = 15,
        search_k: int = 50,
        use_reranking: bool = False,
        reranker_model: str = "BAAI/bge-reranker-base",
        relevance_threshold: float = 0.9,
    ):
        self.faiss_store = faiss_store
        self.default_k = default_k
        self.cache = cache
        self.use_dynamic_k = use_dynamic_k
        self.min_k = min_k
        self.max_k = max_k
        self.search_k = search_k
        self.use_reranking = use_reranking
        self.relevance_threshold = relevance_threshold

        # Initialize reranker if requested
        self.reranker = None
        if self.use_reranking:
            self.reranker = CrossEncoder(reranker_model)

        # diagnostics for benchmarking
        self.last_pairs: List[Tuple[Optional[str], Optional[int]]] = []
        self.last_k_used: Optional[int] = None
        self.last_hit: bool = False
        self.last_scores: List[float] = []  # reranking scores
        self.last_reranked: bool = False  # whether reranking was applied
        self.last_adaptive_expanded: bool = False  # whether adaptive retrieval expanded k
        self.last_initial_k: Optional[int] = None  # k before adaptive expansion
        self.last_query: Optional[str] = None  # the actual query passed to retriever
        self.last_documents: List[Document] = []  # the actual retrieved documents with content
        
        # full diagnostics history (for evaluation)
        self.retrieval_history: List[Dict[str, Any]] = []

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


    def _filter_by_period(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Filter quarters vs years
        """
        if not docs:
            return docs

        ql = query.lower()

        want_year = (
            "fiscal year" in ql
            or "fiscal years" in ql
            or ("last" in ql and "year" in ql)
            or "annual" in ql
        )
        want_quarter = "quarter" in ql or "quarters" in ql

        # no filtering if no specific period inferred
        if not (want_year or want_quarter):
            return docs

        filtered: List[Document] = []
        for d in docs:
            report = (d.metadata or {}).get("report", "") or ""
            period_type = infer_period_type(report)

            if want_year and period_type == "year":
                filtered.append(d)
            elif want_quarter and period_type == "quarter":
                filtered.append(d)
        
        # apply filter if there is data
        return filtered or docs
    

    # ---------- main retrieval ----------

    def forward(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Flow when  use_dynamic_k=True:
        1. Check cache - if hit, return cached docs
        2. Determine dynamic K from query or use provided K
        3. Retrieve initial K documents
        4. Rerank if enabled
        5. If results unsatisfactory (low avg score), expand and retrieve more
        6. Return top K documents (sorted by relevance if reranked)
        """
        # 1. Cache hit path
        if self.cache is not None and query in self.cache:
            docs = self.cache[query]
            self.last_hit = True
            self.last_pairs = self._pairs_from_docs(docs)
            self.last_k_used = len(docs)
            self.last_scores = []  # No reranking on cache hit
            self.last_reranked = False
            self.last_adaptive_expanded = False
            self.last_initial_k = None
            self.last_query = query
            self.last_documents = docs
            
            # Log cache hit
            self._log_retrieval(query, from_cache=True, final_docs=docs)
            return docs

        # 2. dynamic K (only when enabled and no explicit k from agent)
        target_k = None
        if self.use_dynamic_k:
            target_k = k if k is not None else self._infer_k_from_query(query)

        if target_k is None:
            target_k = self.default_k  # fallback

        # clamp
        target_k = max(self.min_k, min(self.max_k, int(target_k)))
        
        # reset diagnostics for this retrieval
        self.last_initial_k = target_k
        self.last_reranked = False
        self.last_adaptive_expanded = False
        self.last_scores = []

        # 3. FAISS retrieval
        docs = self.faiss_store.similarity_search(query, k=target_k)

        # Metadata filter
        docs = self._filter_by_period(query, docs)

        # 4. rerank if enabled
        if self.use_reranking and self.reranker is not None:
            docs = self._rerank_documents(query, docs)
            self.last_reranked = True
            
            # 5. adaptively expand if dynamic_k enabled and results unsatisfactory
            if self.use_dynamic_k and target_k < self.max_k:
                docs = self._adaptive_retrieve(query, docs, target_k)
        
        # 6. final docs - if adaptive expansion happened, scale quantity based on quality gap
        if self.last_adaptive_expanded and self.last_scores:
            # Calculate how far avg score is from threshold
            top_k_scores = self.last_scores[:target_k]
            avg_score = sum(top_k_scores) / len(top_k_scores) if top_k_scores else 0.0
            
            # Quality gap: how much below threshold we are (0.0 to 1.0)
            quality_gap = max(0.0, self.relevance_threshold - avg_score)
            
            # Scale multiplier based on gap:
            multiplier = 2 + (quality_gap / self.relevance_threshold) * 3
            
            # Calculate final_k based on multiplier
            final_k = int(target_k * multiplier)

            final_k = min(final_k, len(docs), self.max_k)  # Cap at available docs or max_k allowed
            final_docs = docs[:final_k]
        else:
            # No expansion needed, return exactly target_k
            final_docs = docs[:target_k]

        final_docs = sorted(
            final_docs,
            key=lambda doc: (
                doc.metadata.get("report", ""),
                doc.metadata.get("page", 0)
            ),
            reverse=True
        )

        # Update diagnostics
        self.last_hit = False
        self.last_pairs = self._pairs_from_docs(final_docs)
        self.last_k_used = len(final_docs)
        self.last_query = query
        self.last_documents = final_docs

        # Log this retrieval with all metrics
        self._log_retrieval(query, from_cache=False, final_docs=final_docs)

        # Store final docs to cache
        if self.cache is not None:
            self.cache[query] = final_docs

        return final_docs
    

    def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Rerank documents using cross-encoder model.
        Returns documents sorted by relevance score (highest first).
        """
        if not docs:
            return docs
        
        # get scores from cross-encoder
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        
        # attach scores to documents and sort
        scored_docs = []
        for doc, score in zip(docs, scores):
            # Convert to Python float for JSON serialization
            score_float = float(score)
            setattr(doc, '_rerank_score', score_float)
            scored_docs.append((doc, score_float))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # update and return scores (already converted to Python float)
        self.last_scores = [score for _, score in scored_docs]
        return [doc for doc, _ in scored_docs]

    def _adaptive_retrieve(
        self, 
        query: str, 
        initial_docs: List[Document], 
        target_k: int
    ) -> List[Document]:
        """
        Adaptively retrieve more documents to increase quantity when quality is lacking.
        Used when dynamic_k is enabled.
        
        Strategy:
        - Keep expanding k until we reach max_k (always expand if avg < threshold)
        - Goal: Get more documents to compensate for low average quality
        - Return all sorted documents (caller decides how many to use)
        """
        if not initial_docs or not self.last_scores:
            return initial_docs
        
        # Check avg score of top target_k documents
        top_k_scores = self.last_scores[:target_k]
        avg_score = sum(top_k_scores) / len(top_k_scores) if top_k_scores else 0.0
        
        # If quality is good enough, no need to expand
        if avg_score >= self.relevance_threshold and len(top_k_scores) >= target_k:
            return initial_docs
        
        # Quality is low - expand to max_k to get more documents
        if target_k >= self.search_k:
            # Already at max, can't expand more
            return initial_docs
        
        self.last_adaptive_expanded = True
        
        # Retrieve with max_k and rerank to get as many docs as possible
        expanded_docs = self.faiss_store.similarity_search(query, k=self.search_k)
        # for d in expanded_docs:
        #     print(d.metadata.get("report"), d.metadata.get("page"))
        
        # Filter by metadata
        expanded_docs = self._filter_by_period(query, expanded_docs)

        docs = self._rerank_documents(query, expanded_docs)
        
        return docs
    
    def _log_retrieval(self, query: str, from_cache: bool, final_docs: List[Document]) -> None:
        """
        Log comprehensive diagnostics for this retrieval for evaluation purposes.
        """
        log_entry = {
            'query': query,
            'from_cache': from_cache,
            'k_requested': self.last_initial_k,
            'k_final': len(final_docs),
            'reranked': self.last_reranked,
            'adaptive_expanded': self.last_adaptive_expanded,
            'scores': self.last_scores.copy() if self.last_scores else [],
            'score_stats': self._compute_score_stats() if self.last_scores else {},
            'retrieved_pairs': self.last_pairs.copy(),
        }
        self.retrieval_history.append(log_entry)
    
    def _compute_score_stats(self) -> Dict[str, float]:
        """Compute statistics on reranking scores."""
        if not self.last_scores:
            return {}
        
        scores = self.last_scores
        return {
            'min': min(scores),
            'max': max(scores),
            'mean': sum(scores) / len(scores),
            'median': sorted(scores)[len(scores) // 2],
        }
    
    def get_retrieval_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report of retrieval performance.
        
        Returns metrics useful for evaluating reranking effectiveness:
        - Cache hit rate
        - Reranking usage
        - Adaptive expansion rate
        - Score distributions
        """
        if not self.retrieval_history:
            return {'error': 'No retrieval history available'}
        
        total = len(self.retrieval_history)
        cache_hits = sum(1 for entry in self.retrieval_history if entry['from_cache'])
        reranked = sum(1 for entry in self.retrieval_history if entry['reranked'])
        expanded = sum(1 for entry in self.retrieval_history if entry['adaptive_expanded'])
        
        # Collect all scores for distribution analysis
        all_scores = []
        for entry in self.retrieval_history:
            all_scores.extend(entry['scores'])
        
        report = {
            'total_retrievals': total,
            'cache_hit_rate': cache_hits / total if total > 0 else 0,
            'reranking_rate': reranked / total if total > 0 else 0,
            'adaptive_expansion_rate': expanded / total if total > 0 else 0,
            'score_distribution': {
                'min': min(all_scores) if all_scores else None,
                'max': max(all_scores) if all_scores else None,
                'mean': sum(all_scores) / len(all_scores) if all_scores else None,
                'count': len(all_scores),
            },
            'history': self.retrieval_history,
        }
        
        return report

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
                # Strip surrounding quotes if present (both single and double)
                if (q_text.startswith('"') and q_text.endswith('"')) or \
                   (q_text.startswith("'") and q_text.endswith("'")):
                    q_text = q_text[1:-1]
            elif p.lower().startswith("k="):
                try:
                    k_val = int(p[len("k="):].strip())
                except:
                    k_val = None

        return q_text, k_val

    def _tool_func(self, request: str) -> str:
        query_text, k_hint = self._parse_request(request)
        docs = self.forward(query_text, k=k_hint)
        
        # Format documents with content for the agent
        result_lines = [f"Retrieved {len(docs)} documents:\n"]
        for i, doc in enumerate(docs, 1):
            report = doc.metadata.get("report") or doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            content = doc.page_content.strip()
            
            result_lines.append(f"[{i}] {report} (Page {page}):")
            result_lines.append(content)
            result_lines.append("")  # blank line between docs
        
        return "\n".join(result_lines)

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
