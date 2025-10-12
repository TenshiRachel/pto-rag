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