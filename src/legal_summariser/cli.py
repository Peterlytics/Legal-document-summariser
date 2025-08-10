import argparse, sys, os, json, re
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
        print("  python -m legal_summariser.cli contract.txt", file=sys.stderr)
        print("  type contract.txt | python -m legal_summariser.cli -", file=sys.stderr)
        sys.exit(2)
    return sys.stdin.read()

# --------- detection ----------
LEGAL_HINTS = re.compile(r"agreement|clause|party|parties|effective date|governing law|termination|warranty|indemnif|liability|confidential|hereby|whereas|section \d", re.I)

def offline_detect(text: str):
    score = 0
    if LEGAL_HINTS.search(text): score += 0.6
    if len(text.split()) > 120: score += 0.2
    if re.search(r"\bLtd|LLC|PLC|Inc\.\b", text): score += 0.2
    is_legal = score >= 0.6
    return {"is_legal": is_legal, "type": "contract" if is_legal else "other", "confidence": round(min(score,1.0),2), "reason": "heuristic"}

def classify_online(client: OpenAI, model: str, text: str):
    msg = [
        {"role":"system","content":"Classify if the text is a legal document (e.g., contract, policy, statute, terms). Respond ONLY with JSON: {\"is_legal\": bool, \"type\": string, \"confidence\": number, \"reason\": string}"},
        {"role":"user","content":text[:12000]},
    ]
    try:
        resp = client.chat.completions.create(
            model=model, messages=msg, temperature=0,
            response_format={"type":"json_object"}, max_tokens=200
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        # best-effort fallback without JSON mode
        resp = client.chat.completions.create(
            model=model, messages=msg+[{"role":"system","content":"Return valid JSON only."}],
            temperature=0, max_tokens=200
        )
        return json.loads(resp.choices[0].message.content)

# --------- summary ----------
def summarise_offline(text: str) -> str:
    data = {
        "title": "Stub Summary (offline)",
        "parties": ["Alpha Ltd", "Beta LLC"],
        "purpose": "Demonstration-only summary without API.",
        "key_terms": ["confidentiality", "term"],
        "obligations": ["Do not disclose confidential information"],
        "risks": ["Stub mode may miss real risks"],
        "dates_deadlines": ["N/A"],
        "termination": "N/A",
        "governing_law": "N/A",
        "red_flags": ["This is a stub; run online for real output."]
    }
    return json.dumps(data, ensure_ascii=False)

def summarise_online(client: OpenAI, model: str, text: str, max_output_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Summarise the following legal text:\n\n{text}"},
    ]
    try:
        r = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2,
            response_format={"type":"json_object"}, max_tokens=max_output_tokens
        )
        return json.dumps(json.loads(r.choices[0].message.content), ensure_ascii=False)
    except Exception:
        r = client.chat.completions.create(
            model=model, messages=messages+[{"role":"system","content":"Return only valid minified JSON."}],
            temperature=0.2, max_tokens=max_output_tokens
        )
        try:
            return json.dumps(json.loads(r.choices[0].message.content), ensure_ascii=False)
        except Exception:
            return r.choices[0].message.content

def main():
    p = argparse.ArgumentParser(description="Legal Document Summariser (CLI)")
    p.add_argument("path", nargs="?", default="-", help="File path or '-' for stdin")
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--offline", action="store_true", help="Run without any API (stub output)")
    p.add_argument("--allow-non-legal", action="store_true", help="Force summarise even if text looks non-legal")
    args = p.parse_args()

    try:
        text = read_input(args.path)
        if args.offline:
            det = offline_detect(text)
            if not det["is_legal"] and not args.allow_non_legal:
                print(json.dumps({"error":"not_legal","detector":det}, ensure_ascii=False)); sys.exit(3)
            print(summarise_offline(text)); return

        # Online path (supports local servers via OPENAI_BASE_URL)
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key  = os.getenv("OPENAI_API_KEY") or "not-needed-for-some-local-servers"
        client = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI()

        det = classify_online(client, args.model, text)
        if not det.get("is_legal", False) and not args.allow_non_legal:
            print(json.dumps({"error":"not_legal","detector":det}, ensure_ascii=False)); sys.exit(3)

        print(summarise_online(client, args.model, text, args.max_tokens))
    except FileNotFoundError:
        print(f"File not found: {args.path}", file=sys.stderr); sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()
