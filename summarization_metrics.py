import re
from typing import Dict, List
import tiktoken


# -----------------------------
# TOKEN COUNTING
# -----------------------------
def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


# -----------------------------
# NUMBER EXTRACTION
# -----------------------------
def extract_numbers(text: str) -> List[str]:
    pattern = r"-?\d[\d,]*(?:\.\d+)?%?"
    return re.findall(pattern, text)


# -----------------------------
# FINANCIAL KEYWORD DENSITY
# -----------------------------
FINANCIAL_KEYWORDS = {
    "revenue", "income", "profit", "loss", "earnings", "margin",
    "cash", "equity", "liabilities", "assets", "expense", "opex",
    "cost", "operating", "quarter", "fiscal", "growth", "decline",
}


def compute_keyword_density(text: str) -> float:
    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return 0.0

    keyword_matches = sum(1 for w in words if w in FINANCIAL_KEYWORDS)
    return keyword_matches / len(words)


# -----------------------------
# MAIN METRICS FUNCTION
# -----------------------------
def compute_summarization_metrics(original_text: str, summary_text: str) -> Dict:
    orig_tokens = count_tokens(original_text)
    summary_tokens = count_tokens(summary_text)

    numbers_before = extract_numbers(original_text)
    numbers_after = extract_numbers(summary_text)

    keyword_density_before = compute_keyword_density(original_text)
    keyword_density_after = compute_keyword_density(summary_text)

    return {
        "orig_tokens": orig_tokens,
        "summary_tokens": summary_tokens,
        "compression_ratio": round(summary_tokens / orig_tokens, 4)
            if orig_tokens else 0.0,

        "numbers_before": len(numbers_before),
        "numbers_after": len(numbers_after),
        "number_preservation_pct":
            round(len(numbers_after) / len(numbers_before), 4)
            if numbers_before else 1.0,

        "keyword_density_before": round(keyword_density_before, 4),
        "keyword_density_after": round(keyword_density_after, 4),
    }
