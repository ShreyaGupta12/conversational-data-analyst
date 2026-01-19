# agent_main.py
"""
Ollama-driven tool-calling agent (robust).
The assistant MUST return a single JSON object describing which tool to call.
This script calls the LLM, extracts the JSON, dispatches to tools.py, and prints results.
"""

import os
import json
import re
import time
from typing import Any, Dict, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_models import ChatOllama

from tools import clean_feature, plot_histogram, calculate_correlation

# Model name (set OLLAMA_MODEL env var or change default)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

# Create ChatOllama with low temperature for deterministic output
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)

SYSTEM_PROMPT = """
You are a data analyst assistant with access to these tools:
1) plot_histogram -> args: {"feature_name": string, "bins": int (optional)}
2) calculate_correlation -> args: {"feature_a": string, "feature_b": string}
3) clean_feature -> args: {"feature_name": string, "strategy": "mean"|"median"|"drop"}

When the user requests analysis, return EXACTLY one JSON object (no markdown, no explanation):

- If a tool is needed:
  {"tool":"<tool_name>", "args": { ... }}

- If no tool is needed:
  {"tool": null, "answer": "short answer text"}

EXAMPLES (exact JSON only):
User: Plot a histogram for 'monthly_active_users'
Assistant JSON:
{"tool":"plot_histogram","args":{"feature_name":"monthly_active_users","bins":20}}

User: What's the correlation between 'revenue' and 'marketing_spend'?
Assistant JSON:
{"tool":"calculate_correlation","args":{"feature_a":"revenue","feature_b":"marketing_spend"}}

User: Fill missing values in 'funding' using median
Assistant JSON:
{"tool":"clean_feature","args":{"feature_name":"funding","strategy":"median"}}
"""

LOG_PATH = "llm_raw_output.log"


def log_raw(text: str):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(text + "\n")


def extract_json_from_text(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Extract the first balanced JSON object from text using a stack-based parser.
    Returns a dict or None.
    """
    if not text:
        return None

    start = text.find("{")
    if start == -1:
        return None

    stack = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                stack += 1
            elif ch == "}":
                stack -= 1
                if stack == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        jt = candidate.replace("'", '"')
                        jt = re.sub(r",\s*}", "}", jt)
                        jt = re.sub(r",\s*]", "]", jt)
                        try:
                            return json.loads(jt)
                        except Exception:
                            return None
    return None


def _extract_text_from_generate_response(resp: Any) -> Optional[str]:
    """Extract text from common .generate(...) response shapes."""
    try:
        gens = getattr(resp, "generations", None)
        if gens:
            first = gens[0][0]
            txt = getattr(first, "text", None)
            if txt:
                return txt
            msg = getattr(first, "message", None)
            if msg:
                return getattr(msg, "content", str(msg))
        return None
    except Exception:
        return None


def _extract_text_from_predict_messages(resp: Any) -> Optional[str]:
    """Extract text from common .predict_messages or similar response shapes."""
    try:
        if hasattr(resp, "content"):
            return resp.content
        if isinstance(resp, str):
            return resp
        msg = getattr(resp, "message", None)
        if msg and hasattr(msg, "content"):
            return msg.content
        return None
    except Exception:
        return None


def call_llm_once(user_prompt: str) -> Optional[str]:
    """
    Try several ChatOllama call styles and extract text:
      1) llm.generate([...])
      2) llm.predict_messages([...])
      3) llm.generate([[SystemMessage, HumanMessage]])
      4) direct call (if callable)
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]

    # 1) try .generate(list[list[msg]])
    try:
        resp = llm.generate([[messages[0], messages[1]]])
        txt = _extract_text_from_generate_response(resp)
        if txt:
            return txt
    except Exception:
        pass

    # 2) try .predict_messages(messages)
    try:
        if hasattr(llm, "predict_messages"):
            resp2 = llm.predict_messages(messages)
            txt2 = _extract_text_from_predict_messages(resp2)
            if txt2:
                return txt2
    except Exception:
        pass

    # 3) try alternative generate shape
    try:
        resp3 = llm.generate([messages])
        txt3 = _extract_text_from_generate_response(resp3)
        if txt3:
            return txt3
    except Exception:
        pass

    # 4) try direct call if supported
    try:
        if callable(llm):
            out = llm([messages[0], messages[1]])
            if isinstance(out, str):
                return out
            txt4 = _extract_text_from_predict_messages(out)
            if txt4:
                return txt4
    except Exception:
        pass

    return None


def ask_for_json_retry(bad_raw: str) -> Optional[Dict[str, Any]]:
    retry_prompt = (
        "You previously returned:\n\n"
        + bad_raw
        + "\n\nThat output could not be parsed as JSON. Please now return EXACTLY the JSON object "
          "that describes which tool to call (no explanation, no markdown). Follow the schema in the system prompt."
    )
    raw2 = call_llm_once(retry_prompt)
    if raw2:
        log_raw(raw2)
    return extract_json_from_text(raw2) if raw2 else None


def dispatch_tool(call: Dict[str, Any]) -> str:
    if not isinstance(call, dict) or "tool" not in call:
        return "Error: invalid JSON object."

    tool_name = call.get("tool")
    args = call.get("args", {})

    if tool_name is None:
        return call.get("answer", "Okay.")

    if tool_name == "plot_histogram":
        feature = args.get("feature_name")
        bins = args.get("bins", None)
        try:
            return plot_histogram(feature, bins=bins) if bins else plot_histogram(feature)
        except Exception as e:
            return f"Error running plot_histogram: {e}"

    if tool_name == "calculate_correlation":
        a = args.get("feature_a")
        b = args.get("feature_b")
        try:
            return calculate_correlation(a, b)
        except Exception as e:
            return f"Error running calculate_correlation: {e}"

    if tool_name == "clean_feature":
        feature = args.get("feature_name")
        strategy = args.get("strategy", "mean")
        try:
            return clean_feature(feature, strategy)
        except Exception as e:
            return f"Error running clean_feature: {e}"

    return f"Unknown tool: {tool_name}"


def main():
    print("Ollama-driven tool-calling agent (robust). Type 'exit' to quit.")
    print(f"Using model: {OLLAMA_MODEL}")
    print("If parsing fails, the agent will try one auto-retry. Raw outputs are logged to llm_raw_output.log\n")

    examples = [
        "Plot a histogram for 'Total Funding (USD; In Millions)'",
        "What is the correlation between 'Total Funding (USD; In Millions)' and 'Valuation (USD; In Millions)'?",
        "Clean 'Total Funding (USD; In Millions)' using median"
    ]
    print("Examples:\n - " + "\n - ".join(examples) + "\n")

    while True:
        try:
            user = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        raw = call_llm_once(user)
        if raw is None:
            print("Agent: LLM call failed. Check Ollama is running and the model name (OLLAMA_MODEL).")
            print("I tried multiple call styles (.generate, .predict_messages, direct), but none returned text.")
            continue

        log_raw(raw)
        print("\n--- RAW MODEL OUTPUT ---\n")
        print(raw)
        print("\n--- END RAW OUTPUT ---\n")

        parsed = extract_json_from_text(raw)
        if parsed is None:
            parsed = ask_for_json_retry(raw)

        if parsed is None:
            print("Agent: Could not parse valid JSON from model. I logged the raw output to llm_raw_output.log.")
            print("Please paste the RAW MODEL OUTPUT (the block printed above) here so I can debug the exact shape.\n")
            continue

        result = dispatch_tool(parsed)
        print("Agent:", result, "\n")


if __name__ == "__main__":
    main()