"""
core/ner.py

Pulls organization names and industry-relevant noun phrases out of raw
business description text using spaCy. This is a cheap pre-filter step
that runs before the (more expensive) LLM summarization -- it helps
keep the LLM prompt focused on the relevant parts of noisy text.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import spacy

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


@dataclass
class ExtractedEntities:
    organizations: list[str] = field(default_factory=list)
    industry_terms: list[str] = field(default_factory=list)
    raw_text: str = ""


# Generic business/industry nouns worth keeping even though spaCy's default
# NER model doesn't have an "industry term" label of its own.
_INDUSTRY_KEYWORDS = {
    "manufacturing", "retail", "wholesale", "construction", "logistics",
    "consulting", "software", "healthcare", "insurance", "finance",
    "transportation", "agriculture", "hospitality", "warehousing",
    "distribution", "services", "production", "fabrication",
}


def extract_entities(text: str) -> ExtractedEntities:
    """Extract organization names and industry-relevant terms from text."""
    doc = _get_nlp()(text)

    orgs = sorted({ent.text for ent in doc.ents if ent.label_ == "ORG"})

    industry_terms = sorted({
        token.lemma_.lower()
        for token in doc
        if token.lemma_.lower() in _INDUSTRY_KEYWORDS
    })

    return ExtractedEntities(organizations=orgs, industry_terms=industry_terms, raw_text=text)
