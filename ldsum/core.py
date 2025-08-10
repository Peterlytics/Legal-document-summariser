import os
from typing import Optional

try:
    from openai import OpenAI
    from openai import APIError  # type: ignore
except Exception:
    try:
        import openai  # type: ignore
        OpenAI = None  # type: ignore
        APIError = Exception  # type: ignore
    except Exception as e:
        raise RuntimeError("Install openai via: pip install openai") from e

SYSTEM_PROMPT = (
    "You are an expert legal document summarizer. Your role is to provide clear, "
    "concise, and objective summaries of legal documents based solely on the content "
    "provided. Do not add any external information, interpretations, legal advice, "
    "or opinions—stick strictly to summarizing what's in the document. Always include "
    'a disclaimer at the top: "This is an AI-generated summary and not legal advice. '
    'Consult a qualified attorney for professional guidance."\\n\\n'
    "For each summary, structure your output as follows:\\n\\n"
    "1. *Overview*: A high-level summary of the document's purpose, type (e.g., contract, "
    "agreement, will), parties involved, and effective dates in 2-4 sentences.\\n\\n"
    "2. *Key Provisions*: Bullet points listing the main clauses, obligations, rights, and "
    "responsibilities of each party. Include any conditions, timelines, or contingencies.\\n\\n"
    "3. *Financial Aspects*: If applicable, summarize payments, fees, penalties, or "
    "economic terms.\\n\\n"
    "4. *Risks and Liabilities*: Highlight any disclaimers, limitations of liability, "
    "dispute resolution mechanisms, or potential risks mentioned.\\n\\n"
    "5. *Termination and Amendments*: Details on how the document can end, be changed, or renewed.\\n\\n"
    "6. *Other Notable Clauses*: Any unique or miscellaneous sections (e.g., governing law, "
    "confidentiality, non-compete).\\n\\n"
    "7. *Full Summary Length*: Aim for a total summary of 300-600 words unless specified otherwise. "
    "Use neutral, professional language.\\n\\n"
    "If the document is too long or complex, focus on key sections. If the input is not a legal "
    "document, politely decline and explain why."
)

LEGAL_KEYWORDS = [
    "agreement","contract","party","parties","shall","hereby","warranty","representations",
    "liability","indemnify","indemnification","governing law","jurisdiction","venue",
    "arbitration","mediation","term","termination","confidentiality","non-disclosure","nda",
    "consideration","assignment","force majeure","severability","entire agreement","amendment",
    "effective date","dispute resolution","penalty","fees","payment","license","licence",
    "obligation","clause","herein","whereas"
]

def is_probably_legal(text: str) -> bool:
    if not text or len(text.strip()) < 50:
        return False
    low = text.lower()
    score = sum(1 for kw in LEGAL_KEYWORDS if kw in low)
    return score >= 2

def get_client(explicit_key: Optional[str] = None):
    api_key = explicit_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if OpenAI:
        return OpenAI(api_key=api_key)
    else:
        import openai  # type: ignore
        openai.api_key = api_key
        return openai

def summarise(client, text: str, model: str, max_tokens: int, temperature: float = 0.3) -> str:
    # Modern client
    if OpenAI and isinstance(client, OpenAI):
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content.strip()
    # Legacy client
    import openai  # type: ignore
    resp = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return resp["choices"][0]["message"]["content"].strip()
