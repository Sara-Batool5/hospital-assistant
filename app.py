import os
import pickle

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# ---------- Config ----------
INDEX_DIR = "faiss_index"
TOP_K = 4
GROQ_MODEL = "openai/gpt-oss-120b"

st.set_page_config(page_title="Hospital Policy Assistant", page_icon="🏥", layout="centered")


# ---------- Cached loaders ----------
@st.cache_resource
def load_index_and_metadata():
    index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"], data["model_name"]


@st.cache_resource
def load_embed_model(model_name):
    return SentenceTransformer(model_name)


@st.cache_resource
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in secrets. Add it to .streamlit/secrets.toml.")
        st.stop()
    return Groq(api_key=api_key)


# ---------- Retrieval ----------
def retrieve(query, index, chunks, embed_model, k=TOP_K):
    query_vec = embed_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, ids = index.search(query_vec.astype(np.float32), k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({**chunk, "score": float(score)})
    return results


def build_prompt(query, retrieved_chunks):
    context_blocks = []
    for i, c in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"[{i}] Source: {c['source']} (page {c['page']}, category: {c['category']})\n{c['text']}"
        )
    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a hospital policy assistant. Answer the user's question using ONLY the "
        "provided context excerpts from hospital policy documents. If the answer isn't in "
        "the context, say you don't have enough information in the knowledge base. "
        "Be concise and precise, and refer to the excerpt numbers (e.g. [1], [2]) that "
        "support each part of your answer."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    return system_prompt, user_prompt


def generate_answer(client, system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------- UI ----------
st.title("🏥 Hospital Policy Assistant")
st.caption("Ask a question about hospital policy. Answers are grounded in your knowledge base.")

index, chunks, embed_model_name = load_index_and_metadata()
embed_model = load_embed_model(embed_model_name)
groq_client = load_groq_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}** — page {s['page']} ({s['category']})")

query = st.chat_input("Ask about a hospital policy...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            retrieved = retrieve(query, index, chunks, embed_model)
            system_prompt, user_prompt = build_prompt(query, retrieved)
            answer = generate_answer(groq_client, system_prompt, user_prompt)

        st.markdown(answer)

        seen = set()
        unique_sources = []
        for c in retrieved:
            key = (c["source"], c["page"])
            if key not in seen:
                seen.add(key)
                unique_sources.append(c)

        if unique_sources:
            with st.expander("Sources"):
                for s in unique_sources:
                    st.markdown(f"- **{s['source']}** — page {s['page']} ({s['category']})")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": unique_sources}
    )
