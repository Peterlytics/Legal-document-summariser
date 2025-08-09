import argparse, sys, os
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

def summarise(text: str, model: str, max_output_tokens: int) -> str:
    client = OpenAI()
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarise the following legal text:\n\n{text}"},
        ],
        max_output_tokens=max_output_tokens,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return resp.output_text

def main():
    parser = argparse.ArgumentParser(description="Legal Document Summariser (CLI)")
    parser.add_argument("path", nargs="?", default="-", help="File path or '-' for stdin")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--max-tokens", type=int, default=1200)
    args = parser.parse_args()
    try:
        text = read_input(args.path)
        summary_json = summarise(text, args.model, args.max_tokens)
        print(summary_json)
    except FileNotFoundError:
        print(f"File not found: {args.path}", file=sys.stderr); sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()
