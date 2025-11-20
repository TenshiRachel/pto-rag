import time
import json
import os
import asyncio
from pyinstrument import Profiler
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import initialize_agent, AgentType, AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
# from langchain_community.vectorstores import FAISS
from pdf_ingestion_chunking import process_all_async, ChunkingEvaluator, SmartFinancialChunker
from build_indices import build_faiss_index, BM25Index
from agent_tools.calculator import CalculatorTool
from agent_tools.comparison import ComparisonTool
from agent_tools.retriever import RetrieverTool
from utilities.json import extract_json_and_prose, strip_json_fences
from utilities.timing import TimingCallback
from utilities.retrieval import RetrievalConfig
import pandas as pd
from retrieval_summarizer import RetrievalSummarizer
from summarization_metrics import compute_summarization_metrics


def create_system_message(question, summaries=None):
    summary_block = ""
    if summaries:
        formatted = "\n".join(
            f"[Summary from {s['report']} p{s['page']}] {s['summary']}"
            for s in summaries
        )
        summary_block = f"\nRetrieved Summaries:\n{formatted}\n"

    return f"""
        You are a financial analyst agent that must reason step-by-step BEFORE using any tools.

        Use Chain-of-Thought planning:
        1. Understand what specific financial data is required
        2. From the retrieved chunks, identify which fiscal periods and report types are available
        3. List the exact fiscal periods needed for the question (only using periods you have seen in retrieved metadata)
        4. Decide which tools are needed and in what order
        5. Run the tools
        6. Verify retrieved numbers and periods match the requested ones

        Period selection rules:
        - Your knowledge of time periods is limited to what appears in the retrieved chunks.
        - Use report names and metadata (e.g. FY26Q2_10Q, FY25_10K, FY25Q4_QuarterlyPresentation) to infer fiscal year and quarter.
        - When the question says "latest quarter" or "most recent quarter":
          Choose the quarter with the highest fiscal year and quarter among reports that are 10-Qs or QuarterlyPresentations.
        - When the question says "latest fiscal year" or "most recent year":
          Choose the highest fiscal year among 10-K reports (and matching annual presentations if needed).
        - Never use a fiscal year or quarter that does NOT appear in the retrieved report names or metadata.
        - If a requested period is not present in the retrieved data, state this explicitly instead of guessing.

        Rules:
        - Never guess numbers or fiscal periods
        - All figures must be retrieved using the retriever tool
        - Provide citations for every value used
        - Use values from **GAAP** "Financial Summary" tables. Avoid Non-GAAP reconciliation tables unless the question explicitly asks for Non-GAAP metrics.
        - Ignore and do NOT extract values from any section titled "Reconciliation of Non-GAAP to GAAP Financial Measures"

        Use tools at most ONCE per tool type.

        Once done, return
        1. The **final structured JSON output** in this format:
        2. **Prose explanation**, converting the JSON output into a formatted table INSIDE the JSON fields only
        3. **NO** markdown, code fences, headings, backslashes, lists, tables, LaTeX, or narrative outside the JSON object

        {{
        "query": "...",
        "data_values": [...],
        "computed_values": [...],
        "citations": [{{"report": "...", "page": ..., "section": "..."}}],
        "tools": ["<list the tools you actually used>"],
        "tools_count": <total number of tools used>,
        "explanation": "<put formatted prose here, NOT outside JSON>"
        }}

        Guidelines:
        - `data_values` contain the raw financial figures, corresponding fiscal years, and units retrieved directly from reports before any calculations.
        - `computed_values` include the calculated results (e.g., YoY or QoQ changes) together with the corresponding values from data_values.
        - Always include every period in `computed_values`, even if the change value is null.

        {summary_block}

        Now answer:
        {question}
        """

def build_eval_prompt(agent_output, question, ground_truth=""):
    return f"""
    You are an expert financial analyst evaluator.

    The `ground_truth` provided below already contains the correct and most recent data.
    Do NOT attempt to infer, update, or verify it externally.
    Your task is only to compare the agent's output to the provided ground truth 
    and score the evaluation criteria based on consistency.

    Scoring Rules:
    - All metric scores are percentages from 0 to 100.
    - 100% = perfectly correct, compliant, or aligned
    - 0% = completely incorrect or missing

    You MUST respond strictly in JSON format:
    {{
      "accuracy_pct": number,     # % match between agent values and ground truth (content correctness)
      "format_pct": number,       # % adherence to expected output format and structure
      "tool_use_pct": number,     # % appropriateness and correctness of tool usage vs expected tools
      "citation_pct": number,     # % of claims properly supported by cited ground truth
      "final_pct": number,        # weighted or averaged overall % score
      "comments": "Brief explanation and key issues only"
    }}

    IMPORTANT:
    - Evaluate ONLY relative to the ground truth below.
    - Keep comments concise.
    - Do NOT provide any reasoning or explanation outside the JSON.
    - JSON must be valid and parseable.
    - Do NOT use markdown or code fences.

    Agent Response:
    {agent_output}

    Question:
    {question}

    Ground Truth:
    {ground_truth}
    """


def create_agent(faiss_store, retriever_config):
    # Create Tools
    print("=== RETRIEVAL CONFIG ===")
    print("output:", retriever_config.output)
    print("use_cache:", retriever_config.use_cache)
    print("use_dynamic_k:", retriever_config.use_dynamic_k)
    print("use_rerank:", retriever_config.use_rerank)
    print("========================")

    retriever = RetrieverTool(
        faiss_store=faiss_store,
        default_k=12,
        cache=retriever_config.cache if getattr(retriever_config, "use_cache", False) else None,
        use_dynamic_k=retriever_config.use_dynamic_k,
        use_reranking=retriever_config.use_rerank,
        reranker_model="BAAI/bge-reranker-base",
        relevance_threshold=0.8,
    )

    retriever_tool = retriever.as_tool()
    comparison_tool = ComparisonTool().as_tool()
    calculator_tool = CalculatorTool().as_tool()
    tools = [retriever_tool, comparison_tool, calculator_tool]

    # Initialize OpenAI LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

    # ChatPromptTemplate for chat history and user input
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a financial analysis assistant capable of using tools to answer complex financial questions."
        ),
        SystemMessagePromptTemplate.from_template(
            "{input}"  # To pass in query
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the agent using OpenAI Tools
    agent = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Wrap in AgentExecutor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,  # 6 for now
    )

    return agent_executor, retriever


def evaluate_agent_output(agent_output, question, ground_truth=""):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    eval_prompt = build_eval_prompt(
        agent_output=agent_output,
        question=question,
        ground_truth=ground_truth
    )

    eval_response = llm.invoke(eval_prompt)

    raw_text = eval_response.content
    clean_text = strip_json_fences(raw_text)

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON returned by judge", "raw": raw_text}

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
        # simple list of reports -> single group with weight 2
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
            merged = {
                k: (int(v) if isinstance(v, (int, float, str)) else 1)
                for k, v in el.items()
                if isinstance(k, str)
            }
            if merged:
                out.setdefault("__all__", {}).update(merged)

        return out

    return {}


def compute_retrieval_metrics(ground_truth_entry, retrieved_pairs):
    # Extract just report ids
    retrieved_reports = [r[0] for r in retrieved_pairs if r and r[0]]
    denom = max(len(retrieved_reports), 1)

    relevant_groups_raw = ground_truth_entry.get("relevant_docs")
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
        # Fallback: citation-level metrics when no grouped relevance is provided
        gt_citations = ground_truth_entry.get("expected_citations", [])
        gt_set = {
            (c.get("report"), c.get("page"))
            for c in gt_citations
            if c.get("report") is not None and c.get("page") is not None
        }
        retrieved_set = {
            (r[0], r[1])
            for r in retrieved_pairs
            if r[0] is not None and r[1] is not None
        }
        inter = gt_set.intersection(retrieved_set)

        precision_at_k = (len(inter) / len(retrieved_set)) if retrieved_set else 0.0
        recall_at_k = (len(inter) / len(gt_set)) if gt_set else 0.0
        graded_precision_at_k = precision_at_k
        graded_recall_at_k = recall_at_k

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "graded_precision_at_k": graded_precision_at_k,
        "graded_recall_at_k": graded_recall_at_k,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write JSON results (e.g. agent_results_baseline.json)",
    )
    parser.add_argument("--use_cache", action="store_true", help="Enable retrieval caching")
    parser.add_argument("--use_dynamic_k", action="store_true", help="Enable dynamic-K retrieval")
    parser.add_argument("--use_rerank", action="store_true", help="Enable re-ranking and adaptive retrieval")

    args = parser.parse_args()

    retrievalConfig = RetrievalConfig(
        output=args.output,
        use_cache=args.use_cache,
        use_dynamic_k=args.use_dynamic_k,
        use_rerank=args.use_rerank
    )

    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Async ingestion & chunking evaluation
    all_docs = asyncio.run(process_all_async("data/"))
    chunking_evaluator = ChunkingEvaluator(all_docs)

    results = chunking_evaluator.evaluate_chunking_strategy(
        chunk_sizes=[600, 700, 800],
        chunk_overlaps=[100, 150, 200]
    )

    chunking_evaluator.print_evaluation_report(results)

    # Find optimal configuration
    optimal_chunk_config = chunking_evaluator.find_optimal_config(results)
    print(f"\nOPTIMAL CONFIG: size={optimal_chunk_config['chunk_size']}, overlap={optimal_chunk_config['chunk_overlap']}")
    print(f"  Composite Score: {optimal_chunk_config['composite_score']:.3f}")

    chunker = SmartFinancialChunker(
        chunk_size=optimal_chunk_config['chunk_size'],
        chunk_overlap=optimal_chunk_config['chunk_overlap']
    )

    chunked_docs = chunker.chunk_documents(all_docs)

    print(f"Chunked into {len(chunked_docs)} total segments.")
    print(f"Parsed and chunked {len(all_docs)} total text segments.")

    EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")  # Can be 'text-embedding-3-large'
    faiss_store = build_faiss_index(chunked_docs, EMBEDDING_MODEL)

    # Load the questions from JSON file
    with open('qa/nvda_ground_truth3.json', 'r') as f:
        test_questions = json.load(f)

    # Extract just the queries
    queries = [item['query'] for item in test_questions]

    # Extract the ground truth
    ground_truth_map = {item['query']: item for item in test_questions}

    all_results = []

    # Agent created outside the loop instead of inside
    agent_executor, retriever_tool_instance = create_agent(faiss_store, retrievalConfig)

    for i, question in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print('='*60)

        # Create NEW profiler for each question
        prof = Profiler(async_mode="disabled")
        timing_callback = TimingCallback()

        start = time.perf_counter()
        prof.start()
        
        prompt_initial = create_system_message(question)
        response_initial = agent_executor.invoke(
            {"input": prompt_initial},
            config={"callbacks": [timing_callback]}
        )
        
        last_documents = getattr(retriever_tool_instance, "last_documents", [])
        
        # Build a JSON-serialisable view of retrieved chunks
        retrieved_chunks = [
            {
                "report": doc.metadata.get("report")
                or doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "content": doc.page_content,
            }
            for doc in last_documents
        ]

        # -------------------------------------------------------------
        # SUMMARIZE RETRIEVED CHUNKS BEFORE CREATING PROMPT
        # -------------------------------------------------------------
        summarizer = RetrievalSummarizer()
        summary_output = summarizer.summarize_chunks(retrieved_chunks)

        summaries = summary_output["summaries"]
        # Compute summarization metrics for each chunk
        summarization_metrics = [
            compute_summarization_metrics(chunk["content"], s["summary"])
            for chunk, s in zip(retrieved_chunks, summaries)
        ]
        
        # ========== AVERAGE SUMMARIZATION METRICS ==========
        def avg(values):
            return sum(values) / len(values) if values else 0

        avg_orig_tokens = avg([m["orig_tokens"] for m in summarization_metrics])
        avg_summary_tokens = avg([m["summary_tokens"] for m in summarization_metrics])
        avg_compression = avg([m["compression_ratio"] for m in summarization_metrics])
        avg_num_retention = avg([m["number_preservation_pct"] for m in summarization_metrics])
        avg_kw_before = avg([m["keyword_density_before"] for m in summarization_metrics])
        avg_kw_after = avg([m["keyword_density_after"] for m in summarization_metrics])

        summarization_avg = {
            "avg_orig_tokens": avg_orig_tokens,
            "avg_summary_tokens": avg_summary_tokens,
            "avg_compression_ratio": avg_compression,
            "avg_numeric_retention": avg_num_retention,
            "avg_keyword_density_before": avg_kw_before,
            "avg_keyword_density_after": avg_kw_after,
        }

        # -------------------------------------------------------------
        # BUILD SYSTEM MESSAGE WITH SUMMARIES INCLUDED
        # -------------------------------------------------------------
        prompt = create_system_message(question, summaries=summaries)

        # -------------------------------------------------------------
        # RUN AGENT WITH SUMMARIZED CONTEXT
        # -------------------------------------------------------------
        response = agent_executor.invoke({"input": prompt}, config={"callbacks": [timing_callback]})

        timing_callback.finalize()

        prof.stop()
        elapsed = time.perf_counter() - start

        # Save individual profile files
        os.makedirs("./results_log", exist_ok=True)
        profile_session_path = f"./results_log/03_Agent_q{i}.pyisession"
        profile_html_path = f"./results_log/03_Agent_q{i}.html"

        # Save individual profile
        print(f"\n--- Profile for Question {i} ---")
        print(prof.output_text(unicode=True, color=True))
        prof.last_session.save(profile_session_path)

        with open(profile_html_path, "w", encoding="utf-8") as f:
            f.write(prof.output_html())

        # Extract results
        output_text = response.get("output", "")
        data, prose = extract_json_and_prose(output_text)

        # Ground truth for this question
        ground_truth_entry = ground_truth_map.get(question, {})
        ground_truth_text = json.dumps(ground_truth_entry, indent=2)

        # --- Retrieval diagnostics & metrics (from RetrieverTool) ---
        retrieved_pairs = list(
            getattr(retriever_tool_instance, "last_pairs", [])
        )
        k_used = getattr(retriever_tool_instance, "last_k_used", None)
        initial_k = getattr(
            retriever_tool_instance, "last_initial_k", None
        )
        adaptive_expanded = getattr(
            retriever_tool_instance, "last_adaptive_expanded", False
        )
        reranked = getattr(
            retriever_tool_instance, "last_reranked", False
        )
        rerank_scores = getattr(
            retriever_tool_instance, "last_scores", []
        )
        last_query = getattr(
            retriever_tool_instance, "last_query", None
        )
        last_documents = getattr(
            retriever_tool_instance, "last_documents", []
        )

        # Compute graded / binary precision & recall
        retrieval_metrics = compute_retrieval_metrics(
            ground_truth_entry, retrieved_pairs
        )

        # Store result
        result = {
            "question_number": i,
            "question": question,
            "raw_response": f"{response}",
            "raw_output": output_text,
            "data": data,
            "prose": prose,
            "elapsed_time": elapsed,
            "callback_timing": timing_callback.get_summary(),
            "profile_session_path": profile_session_path,
            "profile_html_path": profile_html_path,
            # retrieval diagnostics
            "retriever_query": last_query,
            "retrieved_pairs": retrieved_pairs,
            "retrieved_chunks": retrieved_chunks,
            "k_used": k_used,
            "initial_k": initial_k,
            "adaptive_expanded": adaptive_expanded,
            "reranked": reranked,
            "rerank_scores": rerank_scores,
            # retrieval metrics
            "precision_at_k": retrieval_metrics["precision_at_k"],
            "recall_at_k": retrieval_metrics["recall_at_k"],
            "graded_precision_at_k": retrieval_metrics[
                "graded_precision_at_k"
            ],
            "graded_recall_at_k": retrieval_metrics[
                "graded_recall_at_k"
            ],
        }
        all_results.append(result)

        raw_output = result["raw_output"]

        eval_result = evaluate_agent_output(
            raw_output,
            question,
            ground_truth=ground_truth_text
        )

        result["ground_truth"] = ground_truth_entry
        result["evaluation"] = eval_result
        result["summaries"] = summaries
        result["summarization_metrics"] = summarization_metrics
        result["summarization_avg"] = summarization_avg

        # Print summary
        print(f"\n--- Profile for Question {i} ---")
        print(prof.output_text(unicode=True, color=True))
        print(f"\nResponse: {response}")
        print(f"Elapsed time: {elapsed:.2f}s")

    with open("agent_results_with_eval.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary statistics
    total_time = sum(r['elapsed_time'] for r in all_results)
    avg_time = total_time / len(all_results)
    print(f"\nTotal questions: {len(all_results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per question: {avg_time:.2f}s")

    df = pd.read_json("agent_results_with_eval.json")
    df["final_score"] = df["evaluation"].apply(
        lambda x: x.get("final_score", None)
        if isinstance(x, dict)
        else None
    )