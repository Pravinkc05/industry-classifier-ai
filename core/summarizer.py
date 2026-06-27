"""
core/summarizer.py

Condenses a raw, often marketing-heavy company description into a clean,
comparable one-paragraph business profile -- the text that actually gets
embedded and matched against industry codes.

Supports two interchangeable providers, switched via core.config.LLM_PROVIDER:
  - "openai": calls the OpenAI chat completions API
  - "ollama": calls a local model via Ollama (no API key, no cost, offline)

Both code paths return the same shape of output, so callers
(matcher.py, app.py) don't need to know which one is active.
"""

from __future__ import annotations

from core import config
from core.ner import ExtractedEntities

_PROMPT_TEMPLATE = """Summarize the following company description into a single,
clear paragraph (3-4 sentences) describing what the business actually does,
its industry, and its primary products or services. Strip out marketing
language, slogans, and filler. Be concrete and factual.

Detected organization name(s): {orgs}
Detected industry-related terms: {terms}

Company description:
\"\"\"
{text}
\"\"\"

Business profile summary:"""


def _build_prompt(entities: ExtractedEntities) -> str:
    return _PROMPT_TEMPLATE.format(
        orgs=", ".join(entities.organizations) or "none detected",
        terms=", ".join(entities.industry_terms) or "none detected",
        text=entities.raw_text[:4000],
    )


def _summarize_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=config.OPENAI_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def _summarize_ollama(prompt: str) -> str:
    import ollama

    response = ollama.generate(model=config.OLLAMA_MODEL, prompt=prompt, options={"temperature": 0})
    return response["response"].strip()


def summarize_business(entities: ExtractedEntities) -> str:
    """
    Returns a clean business-profile summary string, using whichever
    LLM provider is configured (core.config.LLM_PROVIDER).
    """
    prompt = _build_prompt(entities)

    if config.LLM_PROVIDER == "openai":
        return _summarize_openai(prompt)
    elif config.LLM_PROVIDER == "ollama":
        return _summarize_ollama(prompt)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}'. Use 'openai' or 'ollama'."
        )
