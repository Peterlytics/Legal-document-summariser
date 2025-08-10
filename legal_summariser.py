#!/usr/bin/env python3
"""
CLI that reuses ldsum.core for summarisation.
"""

import os
import sys
import argparse
from ldsum.core import is_probably_legal, get_client, summarise

def read_input(source: str) -> str:
    if source == "-":
        return sys.stdin.read()
    if not os.path.exists(source):
        raise FileNotFoundError(f"File not found: {source}")
    if not os.path.isfile(source):
        raise IsADirectoryError(f"Path is not a file: {source}")
    with open(source, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser(description="Summarise a legal document using OpenAI.")
    parser.add_argument("input", help="Path to a text file, or '-' to read from stdin.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name (default: gpt-4o-mini).")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Max tokens (default: 1000).")
    args = parser.parse_args()

    try:
        text = read_input(args.input)
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        sys.exit(1)

    if not text.strip():
        print("Error: Input is empty.", file=sys.stderr)
        sys.exit(1)

    if not is_probably_legal(text):
        print("The input does not appear to be a legal document. Provide a contract/policy/etc.", file=sys.stderr)
        sys.exit(2)

    try:
        client = get_client()
        summary = summarise(client, text, model=args.model, max_tokens=args.max_tokens, temperature=0.3)
        print(summary)
    except Exception as e:
        print(f"Failed to generate summary: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
