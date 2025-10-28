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


def create_agent(faiss_store):
    # Create Tools
    retriever_tool = RetrieverTool(faiss_store=faiss_store, top_k=12).as_tool()
    comparison_tool = ComparisonTool().as_tool()
    calculator_tool = CalculatorTool().as_tool()

    # Initialize OpenAI LLM
    openai_llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

    # Create OpenAI-compatible agent that can use tools
    agent = initialize_agent(
        tools=[retriever_tool, comparison_tool, calculator_tool],
        llm=openai_llm,
        agent_type=AgentType.OPENAI_FUNCTIONS,  # enables OpenAI’s function/tool calling
        verbose=True,
        handle_parsing_errors=True
    )
    return agent


if __name__ == '__main__':
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    all_docs = process_all()
    chunked_docs = chunk_documents(all_docs, MAX_CHUNK_SIZE, CHUNK_OVERLAP)

    print(f"Chunked into {len(chunked_docs)} total segments.")
    print(f"Parsed and chunked {len(all_docs)} total text segments.")

    EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small") # Can be 'text-embedding-3-large'
    faiss_store = build_faiss_index(chunked_docs, EMBEDDING_MODEL)

    # Load the questions from JSON file
    with open('qa/nvda_ground_truth3.json', 'r') as f:
        test_questions = json.load(f)

    # Extract just the queries
    queries = [item['query'] for item in test_questions]

    all_results = []
    # system_prompt = create_system_message()

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

        # Load FAISS Vector Store
        # faiss_store = FAISS.load_local(
        #     "faiss_index",
        #     embeddings=embedding_model,
        #     allow_dangerous_deserialization=True
        # )

        agent = create_agent(faiss_store)

        response = agent.invoke(prompt, config={"callbacks": [timing_callback]})
        timing_callback.finalize()

        prof.stop()
        elapsed = time.perf_counter() - start

        # Save individual profile files
        profile_session_path = f"03_Agent_q{i}.pyisession"
        profile_html_path = f"03_Agent_q{i}.html"

        # Save individual profile
        print(f"\n--- Profile for Question {i} ---")
        print(prof.output_text(unicode=True, color=True))
        prof.last_session.save(profile_session_path)

        with open(profile_html_path, "w", encoding="utf-8") as f:
            f.write(prof.output_html())

        # Extract results
        data, prose = extract_json_and_prose(response['output'])

        # Store result
        result = {
            "question_number": i,
            "question": question,
            "raw_response": f"{response}",
            "raw_output": response['output'],
            "data": data,
            "prose": prose,
            "elapsed_time": elapsed,
            "callback_timing": timing_callback.get_summary(),
            "profile_session_path": profile_session_path,
            "profile_html_path": profile_html_path
        }
        all_results.append(result)

        # Print summary
        print(f"\n--- Profile for Question {i} ---")
        print(prof.output_text(unicode=True, color=True))
        print(f"\nResponse: {response}")
        print(f"Elapsed time: {elapsed:.2f}s")

        # Save all results to a single JSON file
        results_filename = f"agent_results.json"

        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"All results saved to: {results_filename}")
        print(f"{'='*60}")

    # Print summary statistics
    total_time = sum(r['elapsed_time'] for r in all_results)
    avg_time = total_time / len(all_results)
    print(f"\nTotal questions: {len(all_results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per question: {avg_time:.2f}s")
