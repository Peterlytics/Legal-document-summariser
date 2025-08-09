import argparse, sys, os, json
from openai import OpenAI

SYSTEM_PROMPT = """You are a Legal Document Summariser.
Produce a concise, structured brief for a lay reader and a lawyer.

Return JSON with these keys:
- title
- parties
- purpose
- key_terms
- obligations
- risks
- dates_deadlines
- termination
- governing_law
- red_flags
Keep it faithful, neutral, and quote clauses when necessary.
"""

def read_input(path: str | None) -> str:
    if path and path != "-":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if sys.stdin.isatty():
        print("No input provided. Pass a file path or pipe text.", file=sys.stderr)
        print('  python -m legal_summariser.cli contract.txt', file=sys.stderr)
        print('  type contract.txt | python -m legal_summariser.cli -', file=sys.stderr)
        sys.exit(2)
    return sys.stdin.read()

def summarise_offline(text: str) -> str:
    # Deterministic stub so you can develop without any API key
    data = {
        "title": "Stub Summary (offline)",
        "parties": ["Alpha Ltd", "Beta LLC"],
        "purpose": "Demonstration-only summary for development without API access.",
        "key_terms": ["confidentiality", "3-year term"],
        "obligations": ["Do not disclose confidential information"],
        "risks": ["Stub mode may miss real risks"],
        "dates_deadlines": ["N/A"],
        "termination": "N/A",
        "governing_law": "N/A",
        "red_flags": ["This is a stub; run in online mode for real output."]
    }
    return json.dumps(data, ensure_ascii=False)

def summarise_online(text: str, model: str, max_output_tokens: int) -> str:
    # Support local servers via OPENAI_BASE_URL (e.g., LM Studio http://localhost:1234/v1 or Ollama http://localhost:11434/v1)
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or "not-needed-for-some-local-servers"
    client = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Summarise the following legal text:\n\n{text}"},
    ]
    # Try JSON mode first
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=max_output_tokens,
        )
        content = resp.choices[0].message.content
        return json.dumps(json.loads(content), ensure_ascii=False)
    except Exception:
        # Fallback: no JSON mode; ask for JSON and best-effort parse
        resp = client.chat.completions.create(
            model=model,
            messages=messages + [{"role": "system", "content": "Return only valid minified JSON."}],
            temperature=0.2,
            max_tokens=max_output_tokens,
        )
        content = resp.choices[0].message.content
        try:
            return json.dumps(json.loads(content), ensure_ascii=False)
        except Exception:
            return content

def main():
    p = argparse.ArgumentParser(description="Legal Document Summariser (CLI)")
    p.add_argument("path", nargs="?", default="-", help="File path or '-' for stdin")
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--offline", action="store_true", help="Run without any API (stub output)")
    args = p.parse_args()

    try:
        text = read_input(args.path)
        if args.offline:
            print(summarise_offline(text)); return
        print(summarise_online(text, args.model, args.max_tokens))
    except FileNotFoundError:
        print(f"File not found: {args.path}", file=sys.stderr); sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()
