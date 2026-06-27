"""
app.py

Streamlit UI: paste in a company's business description, and get back
the top-k ranked industry code matches with similarity scores, for a
human underwriter to review and confirm.
"""

import streamlit as st

from core import config
from core.ner import extract_entities
from core.summarizer import summarize_business
from core.embeddings import embed_text, load_cached_code_embeddings
from core.matcher import top_k_matches

st.set_page_config(page_title="Industry Classifier AI", layout="centered")

st.title("Industry Classifier AI")
st.caption(
    "Paste a company's business description below to get ranked industry "
    "code suggestions. This tool ranks candidates for human review -- it "
    "doesn't make the final call."
)

with st.sidebar:
    st.subheader("Configuration")
    st.write(f"LLM provider: `{config.LLM_PROVIDER}`")
    st.write(f"Embedding provider: `{config.EMBEDDING_PROVIDER}`")
    st.caption("Change these via the LLM_PROVIDER / EMBEDDING_PROVIDER environment variables.")

description = st.text_area(
    "Company business description",
    height=180,
    placeholder=(
        "e.g. Acme Builders has been constructing commercial office "
        "buildings and retail spaces across the region for over a decade, "
        "specializing in ground-up construction and tenant improvements..."
    ),
)

k = st.slider("Number of matches to show", min_value=3, max_value=10, value=5)

if st.button("Classify", type="primary") and description.strip():
    with st.spinner("Extracting entities..."):
        entities = extract_entities(description)

    with st.spinner(f"Summarizing with {config.LLM_PROVIDER}..."):
        summary = summarize_business(entities)

    st.subheader("Business profile summary")
    st.write(summary)

    with st.spinner("Embedding and matching against industry codes..."):
        profile_embedding = embed_text(summary)
        code_embeddings, codes = load_cached_code_embeddings()
        matches = top_k_matches(profile_embedding, code_embeddings, codes, k=k)

    st.subheader("Top industry code matches")
    for match in matches:
        st.write(f"**{match.code}** — {match.description}  \nConfidence: {match.score:.2%}")

elif description.strip() == "":
    st.info("Enter a business description above and click Classify.")
