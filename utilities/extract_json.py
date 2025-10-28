import re
import json


def extract_json_and_prose(response_text: str):
    """
    Extracts JSON (fenced or unfenced) and prose explanation.
    Captures the full JSON, even with nested braces.
    """
    # Match fenced or unfenced JSON
    pattern = re.compile(
        r"```(?:json)?\s*(\{[\s\S]*\})\s*```"
        r"(?:\s*\*\*Prose Explanation:\*\*\s*([\s\S]*))?"
        r"|(\{[\s\S]*\})\s*(?:\*\*Prose Explanation:\*\*\s*([\s\S]*))?",
        re.DOTALL
    )

    match = pattern.search(response_text)
    if not match:
        print("No JSON block found.")
        return None, None

    # Select the matched JSON group (fenced or unfenced)
    json_str = (match.group(1) or match.group(3) or "").strip()
    prose = (match.group(2) or match.group(4) or "").strip()

    # Trim trailing junk after the last closing brace
    last_brace = json_str.rfind("}")
    if last_brace != -1:
        json_str = json_str[:last_brace + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print("Captured JSON preview:\n", json_str[:500])
        data = None

    return data, prose
