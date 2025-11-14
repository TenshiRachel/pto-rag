import time
import json
import os
from pyinstrument import Profiler
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import initialize_agent, AgentType
# from langchain_community.vectorstores import FAISS
from pdf_ingestion_chunking import process_all, chunk_documents
from build_indices import build_faiss_index
from agent_tools.calculator import CalculatorTool
from agent_tools.comparison import ComparisonTool
from agent_tools.retriever import RetrieverTool
from utilities.extract_json import extract_json_and_prose
from utilities.timing import TimingCallback


MAX_CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def create_system_message(question):
    return f"""
        You are a financial analyst agent that can use tools.

        Use the retriever tool when you need to fetch financial data. You can only call this tool ONCE.
        Use the comparison tool when you need to compute YoY or QoQ comparisons.
        Use the calculator tool to compute or derive financial ratios or custom metrics (e.g., Opex ÷ Operating Income, Gross Margin %, Net Margin).

        Once done, return
        1. The **final structured JSON output** in this format:
        2. **Prose explanation**, converting the JSON output into a formatted table

        {{
        "query": "...",
        "data_values": [...],
        "computed_values": [...],
        "citations": [{{"report": "...", "page": ..., "section": "..."}}],
        "tools": ["<list the tools you actually used>"],
        "tools_count": <total number of tools used>
        }}

        Guidelines:
        - `data_values` contain the raw financial figures, corresponding fiscal years, and units retrieved directly from reports before any calculations.
        - `computed_values` include the calculated results (e.g., YoY or QoQ changes) together with the corresponding values from data_values.
        - Always include every period in `computed_values`, even if the change value is null.
        - When calling the retriever tool, choose and pass an appropriate k (3–15).

        Now, handle this query:
        {question}
    """


def create_agent(faiss_store, retr_tool, comp_tool, calc_tool):
    openai_llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

    agent = initialize_agent(
        tools=[
            retr_tool.as_tool(),
            comp_tool.as_tool(),
            calc_tool.as_tool()
        ],
        llm=openai_llm,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent

def normalize_relevant_groups(relevant_groups):
    if not relevant_groups:
        return {}

    # dict case: either group->dict or single dict of report->weight
    if isinstance(relevant_groups, dict):
        if all(isinstance(v, dict) for v in relevant_groups.values()):
            out = {}
            for g, d in relevant_groups.items():
                sub = {}
                for rep, w in (d or {}).items():
                    try:
                        sub[str(rep)] = int(w)
                    except Exception:
                        sub[str(rep)] = 1
                out[str(g)] = sub
            return out
        else:
            sub = {}
            for rep, w in relevant_groups.items():
                try:
                    sub[str(rep)] = int(w)
                except Exception:
                    sub[str(rep)] = 1
            return {"__all__": sub}

    # list case
    if isinstance(relevant_groups, list):
        if all(isinstance(x, str) for x in relevant_groups):
            return {"__all__": {rep: 2 for rep in relevant_groups}}

        out = {}
        for el in relevant_groups:
            if not isinstance(el, dict):
                continue

            # NEW: year-grouped shape: {"year": "...", "docs": {...}}
            if "year" in el and "docs" in el and isinstance(el["docs"], dict):
                gname = str(el["year"])
                out[gname] = {}
                for rep, w in el["docs"].items():
                    try:
                        out[gname][str(rep)] = int(w)
                    except Exception:
                        out[gname][str(rep)] = 1
                continue

            # existing: explicit {"report","weight","group"}
            rep = el.get("report")
            if isinstance(rep, str):
                group = el.get("group", "__all__")
                w = el.get("weight", el.get("w", 2))
                try:
                    w = int(w)
                except Exception:
                    w = 2
                out.setdefault(str(group), {})[rep] = w
                continue

            # mapping dict like {"FY25_10K": 2, ...} -> fold into single group
            merged = {k: (int(v) if isinstance(v, (int, float, str)) else 1)
                      for k, v in el.items() if isinstance(k, str)}
            if merged:
                out.setdefault("__all__", {}).update(merged)

        return out

    return {}


def run_benchmark(output_json_path: str, use_cache: bool, use_dynamic_k: bool):
    load_dotenv()

    # 1. ingestion / chunking timing
    ingest_start = time.time()
    all_docs = process_all()
    chunked_docs = chunk_documents(all_docs, chunk_size=800, chunk_overlap=100)
    ingest_end = time.time()
    ingest_time_s = ingest_end - ingest_start

    # 2. build FAISS + embeddings timing
    embed_start = time.time()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    faiss_store = build_faiss_index(chunked_docs, embeddings)
    embed_end = time.time()
    index_build_time_s = embed_end - embed_start

    # load 3 benchmark queries
    with open("qa/nvda_ground_truth3.json", "r") as f:
        gt_items = json.load(f)
    benchmark_questions = [item["query"] for item in gt_items[:3]]

    # shared retrieval cache across questions if optimized
    retrieval_cache = {} if use_cache else None

    all_results = []
    overall_start = time.time()

    for idx, q in enumerate(benchmark_questions, start=1):
        profiler = Profiler()
        profiler.start()

        timing_callback = TimingCallback()
        timing_callback.start_total_timer()

        # decide if this run is optimized (caching + dynamic k) or baseline
        is_optimized = use_cache  # same meaning in your CLI

        retr_tool = RetrieverTool(
            faiss_store=faiss_store,
            default_k=12,
            cache=retrieval_cache,  # cache toggled only by use_cache
            use_dynamic_k=use_dynamic_k,  # dynamic-K toggled separately
        )

        comp_tool = ComparisonTool()
        calc_tool = CalculatorTool()
        agent = create_agent(faiss_store, retr_tool, comp_tool, calc_tool)

        # ask one question
        prompt = create_system_message(q)
        agent_start = time.time()
        response = agent.invoke(prompt, config={"callbacks": [timing_callback]})
        agent_end = time.time()

        timing_callback.end_total_timer()
        profiler.stop()

        # mark whether FIRST retrieval was served from cache
        first_hit = getattr(retr_tool, "last_hit", False)
        timing_callback.register_cache_hit(first_hit)

        # measure "warm cache" speed separately WITHOUT affecting agent timing
        warm_cache_latency = None
        if use_cache:
            t0 = time.time()
            _ = retr_tool.forward(q)  # same query string again
            t1 = time.time()
            warm_cache_latency = (t1 - t0) * 1000.0  # ms

        # --- ground truth relevant citations OR grouped relevance ---
        gt_entry = gt_items[idx - 1]

        retrieved_pairs = list(getattr(retr_tool, "last_pairs", []))
        retrieved_reports = [r[0] for r in retrieved_pairs if r and r[0]]
        k_used = getattr(retr_tool, "last_k_used", None)
        denom = max(len(retrieved_reports), 1)

        relevant_groups_raw = gt_entry.get("relevant_docs")
        group_docs = normalize_relevant_groups(relevant_groups_raw)

        if group_docs:
            MAX_W = 2  # full relevance

            # --- precision@K (binary) ---
            hits = 0
            for rep in retrieved_reports:
                if any(rep in docmap for docmap in group_docs.values()):
                    hits += 1
            precision_at_k = hits / denom

            # --- recall@K (binary, group coverage) ---
            covered = 0
            for _, docmap in group_docs.items():
                if any(rep in docmap for rep in retrieved_reports):
                    covered += 1
            recall_at_k = covered / max(len(group_docs), 1)

            # --- graded precision@K ---
            graded_hits = 0.0
            for rep in retrieved_reports:
                best = 0
                for docmap in group_docs.values():
                    best = max(best, docmap.get(rep, 0))
                graded_hits += 1.0 if best == 2 else (0.5 if best == 1 else 0.0)
            graded_precision_at_k = graded_hits / denom

            # --- graded recall@K ---
            graded_recall_sum = 0.0
            for _, docmap in group_docs.items():
                best_w = 0
                for rep in retrieved_reports:
                    best_w = max(best_w, docmap.get(rep, 0))
                graded_recall_sum += (best_w / MAX_W)
            graded_recall_at_k = graded_recall_sum / max(len(group_docs), 1)

        else:
            gt_citations = gt_entry.get("expected_citations", [])
            gt_set = set((c.get("report"), c.get("page"))
                         for c in gt_citations
                         if c.get("report") is not None and c.get("page") is not None)
            retrieved_set = set((r[0], r[1]) for r in retrieved_pairs
                                if r[0] is not None and r[1] is not None)
            inter = gt_set.intersection(retrieved_set)

            precision_at_k = (len(inter) / len(retrieved_set)) if retrieved_set else 0.0
            recall_at_k = (len(inter) / len(gt_set)) if gt_set else 0.0
            graded_precision_at_k = precision_at_k
            graded_recall_at_k = recall_at_k


        # profiler artifacts
        profile_base = f"{'opt' if use_cache else 'base'}_Agent_q{idx}"
        profile_session_path = profile_base + ".txt"
        profile_html_path = profile_base + ".html"

        with open(profile_session_path, "w", encoding="utf-8") as f_txt:
            f_txt.write(profiler.output_text(unicode=True, color=False))

        with open(profile_html_path, "w", encoding="utf-8") as f_html:
            f_html.write(profiler.output_html())

        # extract structured JSON + prose
        raw_response = str(response)
        parsed_json, prose = extract_json_and_prose(raw_response)

        per_question_record = {
            "record_type": "question_result",
            "run_type": "optimized" if use_cache else "baseline",
            "question_number": idx,
            "question": q,
            "raw_response": raw_response,
            "data": parsed_json if parsed_json else None,
            "prose": prose,

            # retrieval diagnostics
            "retrieved_pairs": retrieved_pairs,
            "k_used": k_used,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "graded_precision_at_k": graded_precision_at_k,
            "graded_recall_at_k": graded_recall_at_k,

            # timing
            "elapsed_time": agent_end - agent_start,
            "callback_timing": timing_callback.get_summary(),
            "first_retrieve_cached": first_hit,
            "warm_cache_retrieval_time_ms": warm_cache_latency,

            # system-level (repeated for convenience)
            "ingest_time_s": ingest_time_s,
            "index_build_time_s": index_build_time_s,

            # profiler paths
            "profile_session_path": profile_session_path,
            "profile_html_path": profile_html_path,
        }
        all_results.append(per_question_record)

        # write partial progress
        with open(output_json_path, "w") as wf:
            json.dump(all_results, wf, indent=2)

    overall_end = time.time()
    total_runtime_s = overall_end - overall_start
    avg_per_q_s = total_runtime_s / len(benchmark_questions)

    print(f"[{('optimized' if use_cache else 'baseline')}] Total runtime: {total_runtime_s:.2f} s")
    print(f"[{('optimized' if use_cache else 'baseline')}] Avg per question: {avg_per_q_s:.2f} s")

    # meta summary (1 row)
    run_meta = {
        "record_type": "run_meta",
        "run_type": "optimized" if use_cache else "baseline",
        "summary_total_runtime_s": total_runtime_s,
        "summary_avg_per_question_s": avg_per_q_s,
        "summary_questions": len(benchmark_questions),
        "summary_qps": len(benchmark_questions) / total_runtime_s,
        "ingest_time_s": ingest_time_s,
        "index_build_time_s": index_build_time_s,
    }

    # final JSON we save:
    final_payload = [run_meta] + all_results
    with open(output_json_path, "w") as wf:
        json.dump(final_payload, wf, indent=2)

    return final_payload



# ---- CLI entrypoint so you can call this from PowerShell ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write JSON results (e.g. agent_results_baseline.json)"
    )
    parser.add_argument("--use_cache", action="store_true", help="Enable retrieval caching")
    parser.add_argument("--use_dynamic_k", action="store_true", help="Enable dynamic-K retrieval")
    args = parser.parse_args()

    run_benchmark(args.output, use_cache=args.use_cache, use_dynamic_k=args.use_dynamic_k)