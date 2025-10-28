import re
import json
from langchain.tools import StructuredTool


class ComparisonTool:
    """Tool to compute Year-over-Year (YoY) or Quarter-over-Quarter (QoQ) percentage changes from numeric data."""

    @staticmethod
    def forward(input_str: str) -> str:
        # Clean and sanitize LLM formatting
        input_str = re.sub(r"^```(?:json)?|```$", "", input_str.strip(), flags=re.IGNORECASE).strip()

        try:
            params = json.loads(input_str)
        except Exception as e:
            return json.dumps({"error": f"Invalid input JSON: {str(e)}", "received": input_str})

        # --- Parse parameters ---
        data = params.get("data", [])
        metric_key = params.get("metric_key", "opex")
        period_key = params.get("period_key", "fiscal_year")
        comparison_type = params.get("comparison_type", "yoy")

        # --- Handle dict input ---
        if isinstance(data, dict):
            data = [{period_key: k, metric_key: v} for k, v in data.items()]

        if not isinstance(data, list):
            return json.dumps({"error": "Expected 'data' to be a list or dict of period:value pairs."})

        # --- Sort chronologically if possible ---
        try:
            data = sorted(data, key=lambda x: x[period_key])
        except Exception:
            pass

        # --- Compute percentage changes ---
        results = []
        for i, record in enumerate(data):
            if i == 0:
                pct_change = None
            else:
                prev_value = data[i - 1][metric_key]
                curr_value = record[metric_key]
                pct_change = None if prev_value == 0 else ((curr_value - prev_value) / prev_value) * 100

            results.append({
                period_key: record[period_key],
                metric_key: record[metric_key],
                f"{comparison_type}_change (%)": round(pct_change, 2) if pct_change is not None else None,
                "units": record.get("units", "millions USD"),
            })

        return json.dumps(results, indent=2)

    def as_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.forward,
            name="comparison_tool",
            description=(
                "Computes Year-over-Year (YoY) or Quarter-over-Quarter (QoQ) percentage changes "
                "for a given financial metric across time periods. Input must be valid JSON with 'data', "
                "'metric_key', 'period_key', and 'comparison_type'."
            ),
        )