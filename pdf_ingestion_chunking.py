import os
import re
import json
import glob
import time
import asyncio
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np


# ============================================================================
# PDF PARSING
# ============================================================================
async def parse_pdf_async(pdf_path: str, report_name: str, year: str = None) -> List[Document]:
    """Parse PDF asynchronously using thread pool for I/O-bound operations"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,  # Uses default executor
        parse_pdf_with_tables,
        pdf_path,
        report_name,
        year
    )


async def parse_pdf_cached_async(pdf_path: str, report_name: str, year: str = None) -> List[Document]:
    """Async version with caching"""
    PARSED_DIR = "cache/parsed_pdfs"
    os.makedirs(PARSED_DIR, exist_ok=True)
    cache_file = os.path.join(PARSED_DIR, f"{report_name}.json")

    loop = asyncio.get_event_loop()

    try:
        # Check cache (async file read)
        if os.path.exists(cache_file):
            print(f"Using cached parse for {report_name}")

            def read_cache():
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)

            data = await loop.run_in_executor(None, read_cache)
            return [Document(**d) for d in data]

        # Parse fresh
        print(f"Parsing {os.path.basename(pdf_path)} ...")
        text_docs, table_docs = await parse_pdf_async(pdf_path, report_name, year)
        docs = text_docs + table_docs

        # Save to cache (async write)
        def write_cache():
            serializable_docs = [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in docs
            ]
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(serializable_docs, f, ensure_ascii=False, indent=2)

        await loop.run_in_executor(None, write_cache)
        return docs

    except Exception as e:
        print(f"Error parsing PDF {report_name}: {e}")
        return []


async def process_all_async(pdf_folder: str = 'data/') -> List[Document]:
    """Process all PDFs concurrently"""
    start_total = time.perf_counter()

    pdf_paths = glob.glob(os.path.join(pdf_folder, "*.pdf"))

    # Create tasks for concurrent processing
    tasks = []
    for pdf_path in pdf_paths:
        report_name = os.path.basename(pdf_path).replace(".pdf", "")
        tasks.append(parse_pdf_cached_async(pdf_path, report_name))

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Flatten results
    all_docs = []
    for docs in results:
        all_docs.extend(docs)

    elapsed_total = time.perf_counter() - start_total
    print(f"\n✓ Loaded {len(all_docs)} parsed documents in {elapsed_total:.2f}s (async)")

    return all_docs


def detect_section(page, text, min_size_diff=2.0):
    """Extract section title from page text using regex patterns or font size"""
    patterns = [
        r"(Item\s+\d+[A-Za-z]?.\s*[A-Z][^\n]+)",
        r"(ITEM\s+\d+[A-Z]?.\s*[A-Z][^\n]+)",
        r"(^[A-Z][A-Z\s]{10,})"
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).strip()

    try:
        words = page.extract_words(extra_attrs=["size", "fontname"])
        if not words:
            return None

        from collections import defaultdict
        lines = defaultdict(list)

        for w in words:
            lines[round(w["top"], -1)].append(w)

        avg_font_size = sum(float(w["size"]) for w in words) / len(words)

        def score_line(line):
            avg_size = sum(float(w["size"]) for w in line) / len(line)
            bold_bonus = any("Bold" in w["fontname"] or "Black" in w["fontname"] for w in line)
            return avg_size + (3 if bold_bonus else 0), avg_size

        scored_lines = [(y, line, *score_line(line)) for y, line in lines.items()]
        if not scored_lines:
            return None

        _, best_line, best_score, best_size = max(scored_lines, key=lambda x: x[2])

        if best_size < avg_font_size + min_size_diff or len(best_line) > 20:
            return None

        section_title = " ".join(w["text"] for w in best_line)
        return section_title.strip() if section_title else None

    except Exception:
        return None


def parse_pdf_with_tables(pdf_path, report_name, year=None):
    """Original synchronous parsing function"""
    text_docs, table_docs = [], []
    current_section = None

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []

            section = detect_section(page, text)
            if section:
                current_section = section

            text_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": i,
                        "report": report_name,
                        "year": year,
                        "section": current_section,
                        "type": "text"
                    }
                )
            )

            for t in tables:
                cleaned_table = [
                    [cell if cell is not None else "" for cell in row]
                    for row in t
                ]
                table_text = "\n".join(["\t".join(row) for row in cleaned_table])
                table_docs.append(
                    Document(
                        page_content=table_text,
                        metadata={
                            "page": i,
                            "report": report_name,
                            "year": year,
                            "section": current_section,
                            "type": "table"
                        }
                    )
                )

    return text_docs, table_docs


# ============================================================================
# CHUNKING
# ============================================================================

class ChunkingEvaluator:
    """
    Evaluate chunking strategies using static metrics only.
    Fast evaluation without requiring embedding models or retrieval testing.
    """

    def __init__(
        self,
        documents: List[Document],
        min_chunk_size: int = 500,
        max_chunk_size: int = 2000,
        target_chunk_size: int = 1000
    ):
        """
        Args:
            documents: Documents to evaluate
            min_chunk_size: Minimum acceptable chunk size (penalize below this)
            max_chunk_size: Maximum acceptable chunk size (penalize above this)
            target_chunk_size: Ideal chunk size for your use case
        """
        self.documents = documents
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.target_chunk_size = target_chunk_size

    def evaluate_chunking_strategy(
        self,
        chunk_sizes: List[int],
        chunk_overlaps: List[int]
    ) -> List[Dict]:
        """
        Evaluate different chunking configurations using static metrics.

        Returns metrics for each configuration:
        - avg_chunk_size: Average size of chunks
        - num_chunks: Total number of chunks
        - context_preservation: How well section context is preserved
        - section_coherence: Measure of semantic coherence within chunks
        - boundary_quality: Quality of chunk boundaries
        - information_density: Ratio of meaningful content to filler
        - size_appropriateness: How well chunks match target size
        - semantic_completeness: Whether chunks contain complete thoughts
        """
        results = []

        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                print(f"Evaluating: chunk_size={chunk_size}, overlap={chunk_overlap}")

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )

                chunks = splitter.split_documents(self.documents)

                # Calculate metrics
                metrics = {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "num_chunks": len(chunks),
                    "avg_chunk_length": np.mean([len(c.page_content) for c in chunks]),
                    "std_chunk_length": np.std([len(c.page_content) for c in chunks]),
                    
                    "context_preservation": self._calculate_context_preservation(chunks),
                    "section_coherence": self._calculate_section_coherence(chunks),
                    "boundary_quality": self._calculate_boundary_quality(chunks),
                    
                    "information_density": self._calculate_information_density(chunks),
                    "size_appropriateness": self._calculate_size_appropriateness(chunks),
                    "semantic_completeness": self._calculate_semantic_completeness(chunks),
                }

                results.append(metrics)

        return results

    def _calculate_context_preservation(self, chunks: List[Document]) -> float:
        """
        Measure how well section metadata is preserved across chunks
        Higher is better (0-1 scale)
        """
        if not chunks:
            return 0.0

        chunks_with_section = sum(1 for c in chunks if c.metadata.get("section"))
        return chunks_with_section / len(chunks)

    def _calculate_section_coherence(self, chunks: List[Document]) -> float:
        """
        Measure how often chunks stay within the same section
        Higher is better (0-1 scale)
        """
        if len(chunks) < 2:
            return 1.0

        same_section_transitions = 0
        total_transitions = 0

        for i in range(len(chunks) - 1):
            curr_section = chunks[i].metadata.get("section")
            next_section = chunks[i + 1].metadata.get("section")

            if curr_section and next_section:
                total_transitions += 1
                if curr_section == next_section:
                    same_section_transitions += 1

        return same_section_transitions / total_transitions if total_transitions > 0 else 0.0

    def _calculate_boundary_quality(self, chunks: List[Document]) -> float:
        """
        Measure quality of chunk boundaries (prefer sentence/paragraph boundaries)
        Higher is better (0-1 scale)
        """
        good_boundaries = 0

        for chunk in chunks:
            content = chunk.page_content
            if not content:
                continue

            # Check if chunk ends with natural boundary
            if content.rstrip().endswith(('.', '!', '?', '\n')):
                good_boundaries += 1

        return good_boundaries / len(chunks) if chunks else 0.0

    def _calculate_information_density(self, chunks: List[Document]) -> float:
        """
        Measure information density: ratio of meaningful content to filler.
        Higher density = more useful for retrieval.
        Score: 0-1 (higher is better)
        """
        if not chunks:
            return 0.0
        
        density_scores = []
        
        # Words that indicate meaningful financial content
        meaningful_keywords = {
            'revenue', 'profit', 'loss', 'earnings', 'growth', 'decline',
            'assets', 'liabilities', 'equity', 'cash', 'debt', 'margin',
            'quarter', 'year', 'increased', 'decreased', 'million', 'billion',
            'risk', 'strategy', 'operations', 'management', 'market', 'sales',
            'income', 'expenses', 'dividend', 'shareholder', 'investment',
            'opex', 'operating', 'efficiency', 'ratio', 'gross'
        }
        
        for chunk in chunks:
            content = chunk.page_content.lower()
            words = content.split()
            
            if not words:
                density_scores.append(0.0)
                continue
            
            # Count meaningful words
            meaningful_count = sum(1 for word in words if any(kw in word for kw in meaningful_keywords))
            
            # Count numbers (financial data)
            number_count = sum(1 for word in words if re.search(r'\d', word))
            
            # Density = (meaningful words + numbers) / total words
            density = (meaningful_count + number_count) / len(words)
            density_scores.append(min(density, 1.0))  # Cap at 1.0
        
        return np.mean(density_scores)

    def _calculate_size_appropriateness(self, chunks: List[Document]) -> float:
        """
        Penalize chunks that are too small (insufficient context) or 
        too large (too much noise). Uses a bell curve around target size.
        Score: 0-1 (higher is better)
        
        Prevents bias for small or large chunks
        """
        if not chunks:
            return 0.0
        
        scores = []
        for chunk in chunks:
            length = len(chunk.page_content)
            
            # Heavily penalize chunks outside acceptable range
            if length < self.min_chunk_size:
                penalty = (length / self.min_chunk_size) ** 2  # Quadratic penalty
                scores.append(penalty * 0.5)  # Max 0.5 score if below min
            elif length > self.max_chunk_size:
                penalty = (self.max_chunk_size / length) ** 2
                scores.append(penalty * 0.7)  # Max 0.7 score if above max
            else:
                # Gaussian curve around target size
                distance = abs(length - self.target_chunk_size)
                normalized_distance = distance / (self.target_chunk_size * 0.5)
                score = np.exp(-normalized_distance ** 2)
                scores.append(score)
        
        return np.mean(scores)

    def _calculate_semantic_completeness(self, chunks: List[Document]) -> float:
        """
        Estimate if chunks contain complete thoughts/concepts.
        Checks for: complete sentences, presence of key financial terms,
        balanced structure (not just fragments).
        Score: 0-1 (higher is better)
        """
        if not chunks:
            return 0.0
        
        completeness_scores = []
        
        for chunk in chunks:
            content = chunk.page_content.strip()
            if not content:
                completeness_scores.append(0.0)
                continue
            
            score = 0.0
            
            # 1. Check for complete sentences (ends with punctuation)
            if content[-1] in '.!?':
                score += 0.3
            
            # 2. Check for sentence structure (has multiple sentences)
            sentence_count = len(re.findall(r'[.!?]+', content))
            if sentence_count >= 2:
                score += 0.3
            elif sentence_count == 1:
                score += 0.15
            
            # 3. Check for balanced content (not just a header or fragment)
            word_count = len(content.split())
            if word_count >= 50:  # Reasonable amount of text
                score += 0.2
            elif word_count >= 20:
                score += 0.1
            
            # 4. Check for contextual clues (section info, temporal markers)
            has_context = bool(re.search(r'\b(year|quarter|fiscal|period|ended)\b', content.lower()))
            if has_context:
                score += 0.2
            
            completeness_scores.append(min(score, 1.0))
        
        return np.mean(completeness_scores)

    def find_optimal_config(
        self,
        results: List[Dict],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Find the best chunking configuration based on composite score.
        
        Uses balanced weights that don't systematically favor small or large chunks.
        
        Args:
            results: List of metric dictionaries from evaluate_chunking_strategy
            weights: Custom weights for metrics (optional)
        """
        if weights is None:
            # Optimized weights to avoid bias toward any chunk size
            weights = {
                # Context metrics (20%) - Secondary but important
                "context_preservation": 0.08,
                "section_coherence": 0.07,
                "boundary_quality": 0.05,
                
                # Retrieval metrics (80%) - Primary drivers
                "information_density": 0.30,      # Financial data presence
                "size_appropriateness": 0.20,     # Balanced size penalty
                "semantic_completeness": 0.30,    # Complete thoughts
            }
        
        for r in results:
            score = 0.0
            for metric, weight in weights.items():
                if metric in r:
                    score += weight * r[metric]
            r["composite_score"] = score
        
        optimal = max(results, key=lambda x: x["composite_score"])
        return optimal
    
    def print_evaluation_report(self, results: List[Dict], top_n: int = 5):
        """
        Print a detailed comparison report of top configurations.
        """
        # Calculate composite scores
        self.find_optimal_config(results)
        
        # Sort by composite score
        sorted_results = sorted(results, key=lambda x: x["composite_score"], reverse=True)
        
        print("\n" + "="*80)
        print(f"CHUNKING EVALUATION REPORT (Top {top_n} Configurations)")
        print("="*80)
        
        for idx, r in enumerate(sorted_results[:top_n], 1):
            print(f"\n#{idx} CONFIG: size={r['chunk_size']}, overlap={r['chunk_overlap']}")
            print(f"  Composite Score: {r['composite_score']:.3f}")
            print(f"\n  Stats:")
            print(f"    • Chunks: {r['num_chunks']}")
            print(f"    • Avg Length: {r['avg_chunk_length']:.0f} ± {r['std_chunk_length']:.0f}")
            print(f"\n  Context Metrics:")
            print(f"    • Context Preservation: {r['context_preservation']:.2%}")
            print(f"    • Section Coherence: {r['section_coherence']:.2%}")
            print(f"    • Boundary Quality: {r['boundary_quality']:.2%}")
            print(f"\n  Retrieval Metrics:")
            print(f"    • Information Density: {r['information_density']:.2%}")
            print(f"    • Size Appropriateness: {r['size_appropriateness']:.2%}")
            print(f"    • Semantic Completeness: {r['semantic_completeness']:.2%}")
            
            if idx < top_n:
                print(f"  {'-'*76}")
        
        print("\n" + "="*80)
        print("RECOMMENDED: Use configuration #1")
        print("="*80)


# ============================================================================
# ENHANCED CHUNKING WITH SMART SECTION-AWARE SPLITTING
# ============================================================================

class SmartFinancialChunker:
    """Section-aware chunking optimized for financial documents"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Smart chunking that preserves section context and handles tables separately
        """
        text_docs = [d for d in documents if d.metadata.get("type") == "text"]
        table_docs = [d for d in documents if d.metadata.get("type") == "table"]

        # Chunk text documents normally
        text_chunks = self.base_splitter.split_documents(text_docs)

        # Keep tables intact (don't split them)
        table_chunks = []
        for table_doc in table_docs:
            # If table is too large, still split but prefer row boundaries
            if len(table_doc.page_content) > self.chunk_size:
                rows = table_doc.page_content.split("\n")
                current_chunk = []
                current_size = 0

                for row in rows:
                    row_size = len(row) + 1  # +1 for newline
                    if current_size + row_size > self.chunk_size and current_chunk:
                        # Create chunk from accumulated rows
                        chunk_content = "\n".join(current_chunk)
                        table_chunks.append(Document(
                            page_content=chunk_content,
                            metadata={**table_doc.metadata, "chunked": True}
                        ))
                        current_chunk = [row]
                        current_size = row_size
                    else:
                        current_chunk.append(row)
                        current_size += row_size

                # Add remaining rows
                if current_chunk:
                    chunk_content = "\n".join(current_chunk)
                    table_chunks.append(Document(
                        page_content=chunk_content,
                        metadata={**table_doc.metadata, "chunked": True}
                    ))
            else:
                table_chunks.append(table_doc)

        return text_chunks + table_chunks
