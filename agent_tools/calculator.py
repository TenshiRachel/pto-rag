import re
import json
from langchain.tools import StructuredTool


FINANCIAL_FORMULAS = {
    "operating_efficiency_ratio": {
        "formula": "operating_expenses / operating_income",
        "description": "Measures how much operating expense is spent per unit of operating income."
    },
    "gross_margin": {
        "formula": "(gross_profit / revenue) * 100",
        "description": "Percentage of revenue remaining after cost of goods sold (COGS)."
    },
    "net_margin": {
        "formula": "(net_income / revenue) * 100",
        "description": "Percentage of revenue retained as net income after expenses."
    },
    "r&d_to_revenue": {
        "formula": "(r_and_d / revenue) * 100",
        "description": "R&D expenses as a percentage of revenue."
    },
}


def resolve_formula_from_query(query: str):
    """Tries to resolve a natural language query to a known formula."""
    q = query.lower()
    for key, entry in FINANCIAL_FORMULAS.items():
        if key.replace("_", " ") in q:
            return entry["formula"]
        if any(kw in q for kw in key.split("_")):
            return entry["formula"]
    return None


# --- Calculator Tool ---
class CalculatorTool:
    """Dynamic Calculator Tool for RAG agents to compute arbitrary financial metrics."""

    @staticmethod
    def forward(input_str: str) -> str:
        """
        Accepts JSON input like:
        {
          "query": "Calculate Operating Efficiency Ratio for the last 3 fiscal years",
          "data": [...],
          "period_key": "fiscal_year",
          "formula": "opex / operating_income"
        }
        Returns: JSON string of computed results.
        """
        # Clean formatting
        input_str = re.sub(r"^```(?:json)?|```$", "", input_str.strip(), flags=re.IGNORECASE).strip()

        try:
            params = json.loads(input_str)
        except Exception as e:
            return json.dumps({"error": f"Invalid input JSON: {str(e)}"})

        query = params.get("query", "").lower()
        data = params.get("data", [])
        period_key = params.get("period_key", "fiscal_year")
        formula = params.get("formula")

        if isinstance(data, dict):
            data = [{period_key: k, **v} for k, v in data.items()]

        if not isinstance(data, list):
            return json.dumps({"error": "Expected 'data' to be a list or dict of period:value mappings."})

        # If no formula is provided, attempt to infer it from the query
        if not formula:
            formula = resolve_formula_from_query(query)
        if not formula:
            return json.dumps({"error": "No formula found or inferred from query."})

        # Identify variables referenced in the formula
        variables = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", formula)

        results = []
        for record in data:
            local_vars = {var: record.get(var) for var in variables}
            if None in local_vars.values():
                results.append({
                    period_key: record.get(period_key),
                    "error": f"Missing data for: {', '.join([k for k,v in local_vars.items() if v is None])}"
                })
                continue

            try:
                # Safe evaluation environment
                value = eval(formula, {"__builtins__": {}}, local_vars)
            except Exception as e:
                results.append({
                    period_key: record.get(period_key),
                    "error": f"Computation failed: {str(e)}"
                })
                continue

            results.append({
                period_key: record[period_key],
                "computed_value": round(value, 4),
                "formula_used": formula,
                "working": f"{formula.replace('/', ' ÷ ')} = {round(value, 4)}"
            })

        return json.dumps(results, indent=2)

    def as_tool(self):
        """Expose this as a LangChain-compatible tool."""
        return StructuredTool.from_function(
            func=self.forward,
            name="calculator_tool",
            description=(
                "Computes arbitrary financial ratios or derived metrics from structured data. "
                "Accepts a JSON input with fields 'query', 'data', 'period_key', and optionally 'formula'. "
                "Understands user queries like 'Calculate operating efficiency ratio' or custom formulas like "
                "'(gross_profit - opex) / revenue'."
            ),
        )
