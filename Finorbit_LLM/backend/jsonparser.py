import re
import json
from typing import Union, Any

def clean_json_block(text: str) -> str:
    """Clean up a JSON string by removing Markdown fences and other noise."""
    # Remove triple-fences and leading "json"
    s = re.sub(r'^\s*```(?:json)?\s*', '', text, flags=re.I)
    s = re.sub(r'\s*```\s*$', '', s)
    s = re.sub(r'^\s*json[:\s]*', '', s, flags=re.I)
    
    # Convert literal "\n" to real newlines
    s = s.replace('\\n', '\n')
    
    return s

def extract_json_block(text: str) -> str:
    """Extract a JSON-like block (object or array) from text."""
    match = re.search(r'(\{.*\}|\[.*\])', text, flags=re.S)
    if not match:
        raise ValueError("No JSON-like block found")
    return match.group(1)

def clean_js_syntax(text: str) -> str:
    """Clean JavaScript-style syntax to make it valid JSON."""
    # Remove JS comments and trailing commas
    s = re.sub(r'//.*?$|/\*.*?\*/', '', text, flags=re.S | re.M)
    s = re.sub(r',\s*(?=[}\]])', '', s)
    
    # Convert single-quoted keys to double-quoted
    s = re.sub(r"'\s*([^']+?)\s*'\s*:", r'"\1":', s)
    
    # Convert single-quoted string values to double-quoted
    s = re.sub(r":\s*'([^']*?)'(?=\s*[,\}\]])", 
               lambda m: ':"' + m.group(1).replace('"', '\\"') + '"', 
               s)
    return s

def fix_json_str(text: Union[str, dict, list]) -> str:
    """
    Return a clean JSON string parsed from `text`.
    
    Args:
        text: Input text that contains JSON-like content, or a dict/list to be JSON-encoded.
            Handles common agent/markdown noise:
            - ```json ... ``` fences
            - leading "json" or "json\\n"
            - literal escaped newlines ("\\n")
            - JS comments (// and /* */)
            - trailing commas
            - single-quoted keys/values
    
    Returns:
        A properly formatted JSON string.
    
    Raises:
        ValueError: If no valid JSON-like block can be found or parsed.
    """
    # If already a dict/list, just encode it
    if isinstance(text, (dict, list)):
        return json.dumps(text, separators=(",", ":"), ensure_ascii=False)

    s = str(text)
    # Clean and extract JSON block
    s = clean_json_block(s)
    s = extract_json_block(s)
    s = clean_js_syntax(s)
    # Remove invalid control characters except for allowed whitespace
    # JSON allows: tab (\t), newline (\n), carriage return (\r)
    # Remove other control chars (0x00-0x1F except \t, \n, \r)
    s = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', s)
    # Escape literal newlines inside quoted string values
    def escape_newlines_in_strings(match):
        value = match.group(0)
        # Replace literal newlines with escaped \n
        return value.replace('\n', '\\n').replace('\r', '\\r')
    # This regex matches quoted string values in JSON
    s = re.sub(r'"(.*?)(?<!\\)"', escape_newlines_in_strings, s, flags=re.S)
    # Try to parse the cleaned JSON
    try:
        obj = json.loads(s)
    except Exception:
        try:
            # Last resort: try simple single->double quote replacement
            obj = json.loads(s.replace("'", '"'))
        except Exception as e:
            # Log more context for debugging
            print("[fix_json_str] Failed to parse JSON. Raw snippet:", s[:500])
            raise ValueError(
                f"Could not parse JSON block. Raw snippet: {s[:200]!r}"
            ) from e
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

# Example usage and tests
if __name__ == "__main__":
    # Test 1: Markdown-fenced JSON
    print(fix_json_str('''
    ```json
{
    "safe": true,
    "issues": null,
    "rewritten_response": "I'd be happy to help you create a retirement plan! To provide you with personalized recommendations, I'll need some basic information about your financial situation:\n\n**Please share:**\n- Your current age\n- Your annual income\n- How much you currently have saved for retirement\n- Your target retirement age\n- Any specific retirement goals (e.g., desired monthly income in retirement, travel plans, etc.)\n\nThe more details you can provide, the more accurate and tailored my retirement planning advice will be. \n\n*This is educational and not financial advice.*"
}
```
'''))
    
    # Test 2: Escaped newlines
    print(fix_json_str(' okay here is the output required for you json\\n{\\n "allowed": true,\\n "blocked_category": null,\n "blocked_reason": null\\n}  hope this helped'))
    
    # Test 3: Single quotes and trailing comma
    print(fix_json_str("sfldls lsjdflj sdlfjo {'a': 'b', 'c': null,}"))
