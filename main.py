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


def run_benchmark(output_json_path: str, use_cache: bool):
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

        # set up tools
        retr_tool = RetrieverTool(faiss_store, k=12, cache=retrieval_cache)
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
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Enable retrieval caching optimization"
    )
    args = parser.parse_args()

    run_benchmark(args.output, use_cache=args.use_cache)