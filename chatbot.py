"""
chatbot.py

Streamlit app that lets a user ask questions like:
  - "What kind of person is this user?"
  - "What are their habits?"
  - "How do they talk?"

It answers using BOTH:
  1. The RAG system (rag_store.py) — topic summaries + raw message chunks
  2. The persona JSON (persona_extractor.py) — structured habits/facts/
     traits/communication style

Run with:
    streamlit run chatbot.py

Prereqs (run once before first launch):
    python build_index.py
This builds the ChromaDB index and persona JSON the chatbot reads from.
"""

import json
import streamlit as st

from rag_store import RagStore
from checkpoint_summarizer import _call_ollama

PERSONA_PATH = "outputs/persona.json"

st.set_page_config(page_title="Conversation Persona Chatbot", layout="wide")
st.title("Conversation Persona Chatbot")
st.caption("Ask about the user's habits, facts, personality, or communication style. "
           "Answers combine RAG retrieval over the conversation log with extracted persona data.")


@st.cache_resource
def load_store():
    return RagStore()


@st.cache_resource
def load_persona():
    try:
        with open(PERSONA_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


store = load_store()
persona = load_persona()

if persona is None:
    st.warning(
        "No persona.json found at outputs/persona.json. "
        "Run `python build_index.py` first to generate the index and persona profile."
    )

with st.sidebar:
    st.header("Persona snapshot")
    if persona:
        st.json(persona)
    else:
        st.write("Not built yet.")

query = st.text_input("Ask a question about the user:", placeholder="What are their habits?")

if st.button("Ask") and query:
    with st.spinner("Retrieving context..."):
        retrieved = store.retrieve(query)
        context_block = RagStore.build_context_block(retrieved)

    persona_block = json.dumps(persona, indent=2) if persona else "No persona data available."

    prompt = (
        "You are answering a question about a person based ONLY on the evidence below. "
        "Do not invent facts not supported by the context or persona data.\n\n"
        f"PERSONA DATA:\n{persona_block}\n\n"
        f"RETRIEVED CONVERSATION CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER (concise, grounded in the evidence above):"
    )

    with st.spinner("Generating answer..."):
        try:
            answer = _call_ollama(prompt)
        except Exception as e:
            answer = f"[Local LLM unavailable: {e}]\n\nRaw retrieved context:\n{context_block}"

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Show retrieved context used"):
        st.text(context_block)
