import re
import json
import math
from langchain.tools import StructuredTool

# Canonical variable alias mapping — fixes SEC naming mismatches
VARIABLE_ALIASES = {
    "operating_income": [
        "operating_income", "operating_income_(loss)", "operating_income_loss"
    ],
    "operating_expenses": [
        "total_operating_expenses", "operating_expenses", "opex"
    ],
    "gross_profit": ["gross_profit"],
    "revenue": ["revenue", "total_revenue"],
    "net_income": ["net_income"],
    "r_and_d": [
        "research_and_development_expenses",
        "research_and_development",
        "research_development",
        "r&d", "rd"
    ]
}

FINANCIAL_FORMULAS = {
    "operating_efficiency_ratio": {
        "formula": "operating_expenses / operating_income",
        "description": "Measures how much operating expense is spent per unit of operating income."
    },
    "gross_margin": {
        "formula": "(gross_profit / revenue) * 100",
        "description": "Percentage of revenue remaining after cost of goods sold."
    },
    "net_margin": {
        "formula": "(net_income / revenue) * 100",
        "description": "Net income as % of revenue."
    },
    "r&d_to_revenue": {
        "formula": "(r_and_d / revenue) * 100",
        "description": "R&D expenses as % of revenue."
    },
}


def resolve_formula_from_query(query: str):
    """Infer a known formula from query text."""
    q = query.lower()
    for key, entry in FINANCIAL_FORMULAS.items():
        if key.replace("_", " ") in q:
            return entry["formula"]
        if any(kw in q for kw in key.split("_")):
            return entry["formula"]
    return None


class CalculatorTool:
    """Financial calculator with period enforcement and alias resolution."""

    @staticmethod
    def forward(input_str: str) -> str:
        """Executes financial metric computations from structured JSON input."""

        # Strip markdown formatting if present
        input_str = re.sub(r"^```(?:json)?|```$", "", input_str.strip(),
                           flags=re.IGNORECASE).strip()

        try:
            params = json.loads(input_str)
        except Exception as e:
            return json.dumps({"error": f"Invalid input JSON: {str(e)}"})

        # Allow multiple input styles
        query = params.get("query", "").lower()
        if not query:
            query = params.get("metric_key", "").lower()

        data = params.get("data", [])
        period_key = params.get("period_key", "fiscal_year")

        # Fix huge agent error: missing conversion dict → list
        if isinstance(data, dict):
            data = [{period_key: k, **v} for k, v in data.items()]

        if not isinstance(data, list):
            return json.dumps({"error": "'data' must be a list of records."})

        if not data:
            return json.dumps({"error": "Empty data list provided."})

        if period_key not in data[0]:
            return json.dumps({"error": f"Missing required period_key '{period_key}' in data."})

        # Formula detection priority
        formula = params.get("formula")
        if not formula:
            formula = params.get("calculation")  # backward compatibility
        if not formula:
            formula = resolve_formula_from_query(query)
        if not formula:
            return json.dumps({"error": "No formula provided or inferred from query."})

        variables = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", formula)
        results = []

        for record in data:
            # Normalize SEC fields
            normalized = {
                re.sub(r'[\s\-()]+', '_', k.strip().lower()): v
                for k, v in record.items()
            }

            # Alias mapping for canonical financial names
            for canonical, aliases in VARIABLE_ALIASES.items():
                for alias in aliases:
                    if alias in normalized:
                        normalized[canonical] = normalized[alias]

            local_vars = {var: normalized.get(var) for var in variables}
            missing = [k for k, v in local_vars.items() if v is None]

            if missing:
                results.append({
                    period_key: record.get(period_key),
                    "error": f"Missing data for: {', '.join(missing)}"
                })
                continue

            try:
                value = eval(
                    formula,
                    {"__builtins__": {}},
                    {
                        **local_vars,
                        **{k: getattr(math, k) for k in dir(math)
                           if not k.startswith("_")}
                    }
                )
            except Exception as e:
                results.append({
                    period_key: record.get(period_key),
                    "error": f"Computation failed: {str(e)}"
                })
                continue

            result_entry = {
                period_key: record[period_key],
                "computed_value": round(value, 4),
                "formula_used": formula
            }

            # Preserve any metadata for citations
            for tag in ["unit", "metadata", "source"]:
                if tag in record:
                    result_entry[tag] = record[tag]

            results.append(result_entry)

        return json.dumps(results, indent=2)

    def as_tool(self):
        return StructuredTool.from_function(
            func=self.forward,
            name="calculator_tool",
            description=(
                "Computes financial ratios from structured data. "
                "Accepts keys 'formula', 'calculation', 'query', or 'metric_key'. "
                "Normalizes SEC financial field names for reliability."
            ),
        )
