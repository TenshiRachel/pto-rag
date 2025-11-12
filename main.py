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
from build_indices import build_faiss_index
from agent_tools.calculator import CalculatorTool
from agent_tools.comparison import ComparisonTool
from agent_tools.retriever import RetrieverTool
from utilities.json import extract_json_and_prose, strip_json_fences
from utilities.timing import TimingCallback
import pandas as pd


def get_nvidia_fiscal_year(today=None):
    """
    Returns NVIDIA's current fiscal year label (e.g., 'FY26').
    NVIDIA's fiscal year ends in January.
    """
    import datetime
    if today is None:
        today = datetime.date.today()

    fiscal_year_end_month = 1  # January
    year = today.year
    if today.month > fiscal_year_end_month:
        year += 1

    return f"FY{str(year)[-2:]}"


def create_system_message(question):
    return f"""
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


def build_eval_prompt(agent_output, question, ground_truth=""):
    from datetime import date
    current_fy = get_nvidia_fiscal_year(today=date.today())
    latest_complete_fy = f"FY{int(current_fy[-2:]) - 1}"

    return f"""
    You are an expert financial analyst evaluating a generated agent response.

    Please assess the following aspects of the agent output, and respond strictly in JSON format:
    {{
      "accuracy_score": integer,
      "format_score": integer,
      "tool_use_score": integer,
      "citation_score": integer,
      "final_score": integer,
      "comments": "Brief explanation and key issues or strengths"
    }}

    ### Evaluation Criteria

    - **Accuracy:**  
      Check that financial values, calculations, and time periods align with the most recent available data.  
      The fiscal year ends in January; current FY = {current_fy}, latest complete FY = {latest_complete_fy}.  
      Penalize outdated or incomplete data but do not mention missing years explicitly.

    - **Format:**  
      Is the structure clear, complete, and compliant with the expected JSON + prose format?

    - **Tool Use:**  
      Were the correct tools (retriever, calculator, comparison) used logically and efficiently?

    - **Citations:**  
      Are citations present, relevant, and tied to the reported figures?

    Respond **only in valid JSON** — no markdown, text, or code fences.

    Agent Response:
    {agent_output}

    Question:
    {question}

    (If applicable) Ground Truth:
    {ground_truth}

    Please write a step-by-step explanation of your score before presenting the final JSON.
    Evaluate honestly and critically.
    """


def create_agent(faiss_store):
    # Create Tools
    retriever_tool = RetrieverTool(faiss_store=faiss_store, top_k=12).as_tool()
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

    return agent_executor


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


if __name__ == '__main__':
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    all_docs = asyncio.run(process_all_async('data/'))
    chunking_evaluator = ChunkingEvaluator(all_docs)

    results = chunking_evaluator.evaluate_chunking_strategy(
        chunk_sizes=[700, 800, 900, 1000],
        chunk_overlaps=[100, 200, 300]
    )

    print("\n--- Evaluation Results ---")
    for r in results:
        print(f"\nConfig: size={r['chunk_size']}, overlap={r['chunk_overlap']}")
        print(f"  Chunks: {r['num_chunks']}")
        print(f"  Avg Length: {r['avg_chunk_length']:.0f} ± {r['std_chunk_length']:.0f}")
        print(f"  Context Preservation: {r['context_preservation']:.2%}")
        print(f"  Section Coherence: {r['section_coherence']:.2%}")
        print(f"  Boundary Quality: {r['boundary_quality']:.2%}")

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

    all_results = []
    # system_prompt = create_system_message()

    # Agent created outside the loop instead of inside or else it will create new agent every iteration
    agent_executor = create_agent(faiss_store)

    for i, question in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print('='*60)

        # Create NEW profiler for each question
        prof = Profiler(async_mode="disabled")
        timing_callback = TimingCallback()

        start = time.perf_counter()
        prof.start()

        prompt = create_system_message(question)

        response = agent_executor.invoke(
            {"input": create_system_message(question)},
            config={"callbacks": [timing_callback]}
        )
        timing_callback.finalize()

        prof.stop()
        elapsed = time.perf_counter() - start

        # Save individual profile files
        profile_session_path = f"./results_log/03_Agent_q{i}.pyisession"
        profile_html_path = f"./results_log/03_Agent_q{i}.html"

        # Save individual profile
        print(f"\n--- Profile for Question {i} ---")
        print(prof.output_text(unicode=True, color=True))
        prof.last_session.save(profile_session_path)

        with open(profile_html_path, "w", encoding="utf-8") as f:
            f.write(prof.output_html())

        # Extract results
        # data, prose = extract_json_and_prose(response['output'])
        output_text = response.get("output", "")
        data, prose = extract_json_and_prose(output_text)

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
            "profile_html_path": profile_html_path
        }
        all_results.append(result)

        raw_output = result['raw_output']
        eval_result = evaluate_agent_output(raw_output, question)
        result["evaluation"] = eval_result

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
    df["final_score"] = df["evaluation"].apply(lambda x: x.get("final_score", None))
