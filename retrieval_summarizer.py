import re
from typing import List, Dict
from langchain_openai import ChatOpenAI
import tiktoken


FINANCIAL_KEYWORDS = [
    "revenue", "income", "profit", "loss", "operating", "margin",
    "cash", "liabilities", "assets", "expense", "opex", "cost",
    "fiscal", "quarter", "guidance", "eps", "earnings"
]


def count_tokens(text: str, model: str = "gpt-4o-mini"):
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


class RetrievalSummarizer:
    """
    Two-pass summarizer:
    - PASS 1: Extract all numeric lines (kept 100% intact)
    - PASS 2: Summarize non-numeric lines only
    - Final Output = numeric lines + compressed narrative
    """

    def __init__(self, model_name="gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    # ---------------------------------------------
    # PASS 1 — Extract all numeric-containing lines
    # ---------------------------------------------
    def extract_numeric_lines(self, text: str) -> List[str]:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

        numeric_lines = []
        non_numeric_lines = []

        for line in lines:
            if re.search(r"\d", line):  # contains any digit
                numeric_lines.append(line)
            else:
                non_numeric_lines.append(line)

        return numeric_lines, non_numeric_lines

    # ---------------------------------------------
    # PASS 2 — Summarize *only* the non-numeric text
    # ---------------------------------------------
    def summarize_narrative(self, non_numeric_lines: List[str]) -> str:
        if not non_numeric_lines:
            return ""  # nothing to summarize

        narrative_text = "\n".join(non_numeric_lines)

        prompt = f"""
        Summarize the following financial narrative text.
        
        RULES:
        - DO NOT add or change any numbers.
        - DO NOT restate specific numeric values.
        - Only summarize qualitative statements and relationships.
        - Keep the summary under 80 tokens.
        - Focus on trends, direction, and key qualitative insights.
        
        Text:
        {narrative_text}

        Summary:
        """

        return self.llm.invoke(prompt).content.strip()

    # ---------------------------------------------
    # FINAL COMPOSITION: numeric (untouched) + summary
    # ---------------------------------------------
    def summarize_chunk(self, chunk: Dict) -> Dict:
        original_text = chunk["content"]

        numeric_lines, non_numeric_lines = self.extract_numeric_lines(original_text)
        narrative_summary = self.summarize_narrative(non_numeric_lines)

        # Combine numeric lines first, summary after
        final_summary = "\n".join(numeric_lines[:12])  # cap for safety
        if narrative_summary:
            final_summary += "\n\n" + narrative_summary

        return {
            "report": chunk["report"],
            "page": chunk["page"],
            "summary": final_summary,
        }

    def summarize_chunks(self, retrieved_chunks: List[Dict]) -> Dict:
        summaries = []
        for chunk in retrieved_chunks:
            summaries.append(self.summarize_chunk(chunk))

        return {
            "summaries": summaries,
            "metrics": []  # main.py handles metrics
        }
