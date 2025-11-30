## Agent CFO Assistant for NVIDIA using Retrieval-Augmented Generation (RAG)

### Why NVIDIA?

NVIDIA is a leading semiconductor and AI technology company whose investor-facing documents contain rich, technically dense information. Its frequent public disclosures (quarterly/annual reports, investor presentations, press releases) make it suitable for a Retrieval-Augmented Generation (RAG) system designed to answer financial and strategic queries.

### Data ingested

1. Annual Reports - 5
2. Quarterly Reports - 8
3. Investor Presentations - 5
4. Press Releases - 3

Total - 21 Documents, ranging from FY2021 to FY2026

## Setup

### Requirement Installation

Run the following:

```
pip install -r requirements.txt
```

### Environment variables

Make sure the base folder contains a .env file with the following variables:

```
OPENAI_API_KEY=your-key-here
```

## Running the agent

### Before optimizations

Please run the Baseline Implementation notebook.

### After optimizations

#### Arguments

Arguments in [] are optional.

Run with all 3 enabled for best results.

- [use_cache] - Use ratio caching (Reuse previously retrieved documents for answers)
- [use_dynamic_k] - Update k (Num of retrieved documents) for retriever during generation
- [use_rerank] - Reorder retrieved documents

#### Run Command
```
python main.py [--use_cache] [--use_dynamic_k] [--use_rerank]
```

**Do take note that agent results produced from after optimization uses LLM as a judge while accuracy analysis is more accurate**

### Analysis

Please run ingestion_benchmark.ipynb and accuracy_result_analysis.ipynb