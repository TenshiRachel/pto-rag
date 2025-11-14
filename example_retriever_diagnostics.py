"""
Example: How to use RetrieverTool diagnostics to evaluate reranking effectiveness
"""
import json
from agent_tools.retriever import RetrieverTool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from build_indices import BM25Index

load_dotenv()

# Perform some retrievals
# docs1 = retriever.forward("What was the revenue in Q1 2024?")
# docs2 = retriever.forward("What are the operating expenses?")
# docs3 = retriever.forward("What was the revenue in Q1 2024?")  # Cache hit

# Check diagnostics after each retrieval
def print_last_retrieval_info(retriever):
    """Print diagnostics from the last retrieval"""
    print(f"Cache hit: {retriever.last_hit}")
    print(f"K used: {retriever.last_k_used}")
    print(f"Initial K: {retriever.last_initial_k}")
    print(f"Reranked: {retriever.last_reranked}")
    print(f"Adaptive expanded: {retriever.last_adaptive_expanded}")
    print(f"Scores: {retriever.last_scores[:5]}...")  # First 5 scores
    print(f"Retrieved pairs: {retriever.last_pairs[:3]}...")  # First 3 pairs
    print()

# Get comprehensive evaluation report
def save_retrieval_report(retriever, output_file="retrieval_evaluation.json"):
    """Generate and save evaluation report"""
    report = retriever.get_retrieval_report()
    
    print("=== Retrieval Performance Report ===")
    print(f"Total retrievals: {report['total_retrievals']}")
    print(f"Cache hit rate: {report['cache_hit_rate']:.2%}")
    print(f"Reranking rate: {report['reranking_rate']:.2%}")
    print(f"Adaptive expansion rate: {report['adaptive_expansion_rate']:.2%}")
    print()
    print("Score distribution:")
    print(f"  Min: {report['score_distribution']['min']}")
    print(f"  Max: {report['score_distribution']['max']}")
    print(f"  Mean: {report['score_distribution']['mean']:.4f}")
    print(f"  Total scores: {report['score_distribution']['count']}")
    print()
    
    # Save detailed history for analysis
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Detailed report saved to {output_file}")

# Analyze specific retrieval from history
def analyze_retrieval(retriever, query_text):
    """Analyze all retrievals for a specific query"""
    matching = [
        entry for entry in retriever.retrieval_history 
        if entry['query'] == query_text
    ]
    
    print(f"=== Analysis for query: '{query_text}' ===")
    print(f"Times retrieved: {len(matching)}")
    
    for i, entry in enumerate(matching, 1):
        print(f"\nRetrieval #{i}:")
        print(f"  From cache: {entry['from_cache']}")
        print(f"  Reranked: {entry['reranked']}")
        print(f"  K requested: {entry['k_requested']}")
        print(f"  K final: {entry['k_final']}")
        if entry['score_stats']:
            print(f"  Score mean: {entry['score_stats']['mean']:.4f}")
            print(f"  Score range: [{entry['score_stats']['min']:.4f}, {entry['score_stats']['max']:.4f}]")

# Compare three configurations: baseline, dynamic K only, and dynamic K + reranking
def compare_retrieval_strategies(faiss_store, bm25_index, test_queries, ground_truth_items=None):
    """
    Run the same queries with three different configurations:
    1. Baseline: No dynamic K, no reranking
    2. Dynamic K only: Adaptive retrieval without reranking
    3. Full: Dynamic K + reranking + adaptive retrieval
    
    Args:
        faiss_store: FAISS vector store
        test_queries: List of query strings
        ground_truth_items: Optional list of dicts with 'query' and 'expected_citations'
    """
    results = {
        'baseline': [],
        'dynamic_k_only': [],
        'dynamic_k_with_reranking': []
    }
    
    # Helper function to compute precision and recall
    def compute_metrics(retrieved_pairs, gt_citations):
        """Compute precision@K and recall@K against ground truth."""
        # Normalize ground truth as set of (report, page) tuples
        gt_set = set(
            (c.get("report"), c.get("page"))
            for c in gt_citations
            if c.get("report") is not None and c.get("page") is not None
        )
        
        # Normalize retrieved pairs
        retrieved_set = set(
            (tuple(r) if isinstance(r, list) else r)
            for r in retrieved_pairs
            if r[0] is not None and r[1] is not None
        )
        
        intersection = gt_set.intersection(retrieved_set)
        
        precision_at_k = len(intersection) / len(retrieved_set) if len(retrieved_set) > 0 else 0.0
        recall_at_k = len(intersection) / len(gt_set) if len(gt_set) > 0 else 0.0
        f1_at_k = (2 * precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) > 0 else 0.0
        
        return {
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k,
            'f1_at_k': f1_at_k,
            'relevant_retrieved': len(intersection),
            'total_retrieved': len(retrieved_set),
            'total_relevant': len(gt_set)
        }
    
    # Create ground truth lookup if provided
    gt_lookup = {}
    if ground_truth_items:
        for item in ground_truth_items:
            gt_lookup[item['query']] = item.get('expected_citations', [])
    
    print("=" * 70)
    print("COMPARING RETRIEVAL STRATEGIES")
    print("=" * 70)
    
    # 1. Baseline: No dynamic K, no reranking
    print("\n[1/3] Running baseline (no dynamic K, no reranking)...")
    retriever_baseline = RetrieverTool(
        faiss_store=faiss_store,
        bm25_store=bm25_index,
        default_k=12,
        use_dynamic_k=False,
        use_reranking=False,
        cache={}
    )
    
    for query in test_queries:
        docs = retriever_baseline.forward(query)
        entry = {
            'query': query,
            'k': len(docs),
            'pairs': retriever_baseline.last_pairs,
            'initial_k': retriever_baseline.last_initial_k,
            'expanded': False
        }
        
        # Add ground truth metrics if available
        if query in gt_lookup:
            metrics = compute_metrics(retriever_baseline.last_pairs, gt_lookup[query])
            entry.update(metrics)
        
        results['baseline'].append(entry)
    
    # 2. Dynamic K only (adaptive retrieval without reranking)
    # Note: Without reranking, there's no quality score to trigger expansion
    # So this will just infer K from query but won't expand
    print("[2/3] Running with dynamic K only (no reranking)...")
    retriever_dynamic_k = RetrieverTool(
        faiss_store=faiss_store,
        bm25_store=bm25_index,
        default_k=12,
        use_dynamic_k=True,
        use_reranking=False,
        cache={}
    )
    
    for query in test_queries:
        docs = retriever_dynamic_k.forward(query)
        entry = {
            'query': query,
            'k': len(docs),
            'pairs': retriever_dynamic_k.last_pairs,
            'initial_k': retriever_dynamic_k.last_initial_k,
            'expanded': retriever_dynamic_k.last_adaptive_expanded
        }
        
        # Add ground truth metrics if available
        if query in gt_lookup:
            metrics = compute_metrics(retriever_dynamic_k.last_pairs, gt_lookup[query])
            entry.update(metrics)
        
        results['dynamic_k_only'].append(entry)
    
    # 3. Full: Dynamic K + reranking + adaptive retrieval
    print("[3/3] Running with dynamic K + reranking + adaptive retrieval...")
    retriever_full = RetrieverTool(
        faiss_store=faiss_store,
        bm25_store=bm25_index,
        default_k=12,
        use_dynamic_k=True,
        use_reranking=True,
        relevance_threshold=0.5,
        cache={}
    )
    
    for query in test_queries:
        docs = retriever_full.forward(query)
        entry = {
            'query': query,
            'k': len(docs),
            'pairs': retriever_full.last_pairs,
            'scores': retriever_full.last_scores,
            'initial_k': retriever_full.last_initial_k,
            'expanded': retriever_full.last_adaptive_expanded
        }
        
        # Add ground truth metrics if available
        if query in gt_lookup:
            metrics = compute_metrics(retriever_full.last_pairs, gt_lookup[query])
            entry.update(metrics)
        
        results['dynamic_k_with_reranking'].append(entry)
    
    # Save comparison
    with open('retrieval_strategy_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Print summary for each query
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")
        print(f"  Baseline:        k={results['baseline'][i]['k']:2d} | expanded={results['baseline'][i]['expanded']}")
        print(f"  Dynamic K only:  k={results['dynamic_k_only'][i]['k']:2d} | initial_k={results['dynamic_k_only'][i]['initial_k']} | expanded={results['dynamic_k_only'][i]['expanded']}")
        print(f"  Full (D+R):      k={results['dynamic_k_with_reranking'][i]['k']:2d} | initial_k={results['dynamic_k_with_reranking'][i]['initial_k']} | expanded={results['dynamic_k_with_reranking'][i]['expanded']}")
        
        # Ground truth metrics if available
        if 'precision_at_k' in results['baseline'][i]:
            print(f"  Ground Truth Metrics:")
            print(f"    Baseline:   P={results['baseline'][i]['precision_at_k']:.3f} | R={results['baseline'][i]['recall_at_k']:.3f} | F1={results['baseline'][i]['f1_at_k']:.3f}")
            print(f"    Dynamic K:  P={results['dynamic_k_only'][i]['precision_at_k']:.3f} | R={results['dynamic_k_only'][i]['recall_at_k']:.3f} | F1={results['dynamic_k_only'][i]['f1_at_k']:.3f}")
            print(f"    Full (D+R): P={results['dynamic_k_with_reranking'][i]['precision_at_k']:.3f} | R={results['dynamic_k_with_reranking'][i]['recall_at_k']:.3f} | F1={results['dynamic_k_with_reranking'][i]['f1_at_k']:.3f}")
        
        # Show if documents changed
        baseline_set = set(map(tuple, results['baseline'][i]['pairs']))
        dynamic_set = set(map(tuple, results['dynamic_k_only'][i]['pairs']))
        full_set = set(map(tuple, results['dynamic_k_with_reranking'][i]['pairs']))
        
        baseline_vs_dynamic = len(baseline_set & dynamic_set)
        baseline_vs_full = len(baseline_set & full_set)
        dynamic_vs_full = len(dynamic_set & full_set)
        
        print(f"  Document overlap:")
        print(f"    Baseline vs Dynamic K:  {baseline_vs_dynamic}/{results['baseline'][i]['k']}")
        print(f"    Baseline vs Full:       {baseline_vs_full}/{results['baseline'][i]['k']}")
        print(f"    Dynamic K vs Full:      {dynamic_vs_full}/{results['dynamic_k_only'][i]['k']}")
    
    # Overall average metrics if ground truth available
    if ground_truth_items and len(gt_lookup) > 0:
        print("\n" + "=" * 70)
        print("AVERAGE METRICS ACROSS ALL QUERIES")
        print("=" * 70)
        
        def avg_metric(strategy, metric):
            values = [r[metric] for r in results[strategy] if metric in r]
            return sum(values) / len(values) if values else 0.0
        
        print(f"\nBaseline:")
        print(f"  Avg Precision@K: {avg_metric('baseline', 'precision_at_k'):.3f}")
        print(f"  Avg Recall@K:    {avg_metric('baseline', 'recall_at_k'):.3f}")
        print(f"  Avg F1@K:        {avg_metric('baseline', 'f1_at_k'):.3f}")
        
        print(f"\nDynamic K Only:")
        print(f"  Avg Precision@K: {avg_metric('dynamic_k_only', 'precision_at_k'):.3f}")
        print(f"  Avg Recall@K:    {avg_metric('dynamic_k_only', 'recall_at_k'):.3f}")
        print(f"  Avg F1@K:        {avg_metric('dynamic_k_only', 'f1_at_k'):.3f}")
        
        print(f"\nFull (D+R):")
        print(f"  Avg Precision@K: {avg_metric('dynamic_k_with_reranking', 'precision_at_k'):.3f}")
        print(f"  Avg Recall@K:    {avg_metric('dynamic_k_with_reranking', 'recall_at_k'):.3f}")
        print(f"  Avg F1@K:        {avg_metric('dynamic_k_with_reranking', 'f1_at_k'):.3f}")
        
        # Show improvement
        baseline_f1 = avg_metric('baseline', 'f1_at_k')
        dynamic_f1 = avg_metric('dynamic_k_only', 'f1_at_k')
        full_f1 = avg_metric('dynamic_k_with_reranking', 'f1_at_k')
        
        print(f"\nF1 Improvement:")
        if baseline_f1 > 0:
            print(f"  Dynamic K vs Baseline: {((dynamic_f1 - baseline_f1) / baseline_f1 * 100):+.1f}%")
            print(f"  Full vs Baseline:      {((full_f1 - baseline_f1) / baseline_f1 * 100):+.1f}%")
        if dynamic_f1 > 0:
            print(f"  Full vs Dynamic K:     {((full_f1 - dynamic_f1) / dynamic_f1 * 100):+.1f}%")
    
    print("\n" + "=" * 70)
    print("Results saved to retrieval_strategy_comparison.json")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    print("Open json")
    with open("qa/nvda_ground_truth3.json", "r") as f:
        gt_items = json.load(f)
    benchmark_questions = [item["query"] for item in gt_items[:3]]
    
    print("Open faiss index")
    faiss_store = FAISS.load_local(
        "faiss_index",
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        allow_dangerous_deserialization=True
    )

    print("Open BM25 Index")
    bm25_index = BM25Index.load("bm25_index.pkl")

    print("Start retrievals")
    # Pass ground truth items for evaluation
    compare_retrieval_strategies(faiss_store, bm25_index, benchmark_questions, ground_truth_items=gt_items[:3])