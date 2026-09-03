import os
import requests
import psycopg2
from pgvector.psycopg2 import register_vector
from google import genai
from google.genai import types
import streamlit as st

st.set_page_config(page_title="Project Hail Mary Agent", page_icon="🚀", layout="wide")

# --- Configuration ---
DB_URL = os.getenv("DATABASE_URL", "postgresql://admin:maryHAIL1990@our-hail-mary-db-service:5432/hail_mary_db")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://embedding-server:8080/v1/embeddings")
RERANKER_API_URL = os.getenv("RERANKER_API_URL", "http://reranker-server:8080/v1/rerank")
SCRATCHPAD_FILE = "scratchpad.md"

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieved_context" not in st.session_state:
    st.session_state.retrieved_context = []
    
if not os.path.exists(SCRATCHPAD_FILE):
    with open(SCRATCHPAD_FILE, "w") as f:
        f.write("# Agent Investigation Scratchpad\n- [ ] Initialized session\n")

def read_scratchpad():
    with open(SCRATCHPAD_FILE, "r") as f:
        return f.read()

def update_scratchpad(new_note):
    with open(SCRATCHPAD_FILE, "a") as f:
        f.write(f"- {new_note}\n")

# --- Core RAG Functions ---
def rewrite_query(raw_query, history):
    """Resolves conversational context into a standalone query."""
    if not history:
        return raw_query
        
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
    prompt = f"Given this chat history:\n{history_text}\n\nRewrite this user query to be completely standalone, replacing vague pronouns with specific names: {raw_query}\n\nOutput ONLY the rewritten query."
    
    response = gemini_client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt
    )
    return response.text.strip()

import re

def retrieve_and_rerank(query):
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    
    # 1. Check for exact Chapter queries (e.g. "Chapter 17", "chapter 5")
    chapter_match = re.search(r'\bchapter\s+(\d+)\b', query, re.IGNORECASE)
    
    if chapter_match:
        chapter_num = chapter_match.group(1)
        target_header = f"[Chapter {chapter_num}"
        
        with conn.cursor() as cur:
            # Fetch ALL chunks belonging to this specific chapter in sequential order
            cur.execute("""
                SELECT chunk_text FROM book_chunks 
                WHERE chunk_text ILIKE %s 
                ORDER BY id ASC 
                LIMIT 30;
            """, (f"%{target_header}%",))
            chapter_chunks = [row[0] for row in cur.fetchall()]
            
        conn.close()
        
        if chapter_chunks:
            # Return the sequential chapter content directly
            return chapter_chunks

    # 2. Standard Semantic Retrieval (for regular conceptual questions)
    res = requests.post(EMBEDDING_API_URL, json={"input": query, "model": "bge-m3"})
    query_vector = res.json()["data"][0]["embedding"]
    
    with conn.cursor() as cur:
        # Increased candidate pool from 15 to 40
        cur.execute("SELECT chunk_text FROM book_chunks ORDER BY embedding <=> %s::vector LIMIT 40;", (query_vector,))
        candidates = [row[0] for row in cur.fetchall()]
    conn.close()
    
    if not candidates:
        return []

    # 3. Cross-encoder scoring
    rerank_res = requests.post(RERANKER_API_URL, json={
        "query": query, 
        "documents": candidates, 
        "model": "bge-reranker-v2-m3"
    })
    results = rerank_res.json().get("results", [])
    sorted_results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
    
    # Return top 10 high-signal chunks instead of 3
    return [candidates[r['index']] for r in sorted_results[:10]]

# --- UI Layout ---
st.title("🚀 Project Hail Mary: Context Engine")

# Sidebar for Transparency
with st.sidebar:
    st.header("🧠 Agent Memory")
    st.markdown("### Scratchpad")
    st.text_area("Internal Notes", value=read_scratchpad(), height=200, disabled=True)
    
    st.markdown("### Active Context Stack")
    for i, chunk in enumerate(st.session_state.retrieved_context):
        with st.expander(f"Retrieved Excerpt {i+1}"):
            st.write(chunk)

# Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the book..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        # 1. Rewrite Query
        with st.spinner("Rewriting query for context..."):
            standalone_query = rewrite_query(prompt, st.session_state.messages[:-1])
            st.caption(f"*(Resolved query: {standalone_query})*")
            
        # 2. Retrieve Facts
        with st.spinner("Searching and Reranking..."):
            retrieved_facts = retrieve_and_rerank(standalone_query)
            st.session_state.retrieved_context = retrieved_facts
            update_scratchpad(f"Queried: '{standalone_query}' -> Found {len(retrieved_facts)} chunks.")
            
        # 3. Generate Answer
        scratchpad = read_scratchpad()
        system_content = "You are a precise literary assistant for Project Hail Mary. Answer accurately using ONLY the provided retrieved facts and scratchpad. If data is missing, state it clearly.\n\n"
        system_content += f"--- SCRATCHPAD ---\n{scratchpad}\n\n--- RETRIEVED EXCERPTS ---\n"
        for i, fact in enumerate(retrieved_facts):
            system_content += f"Excerpt {i+1}:\n{fact}\n\n"

        contents = []
        for msg in st.session_state.messages[-5:]: 
            role = "user" if msg["role"] == "user" else "model"
            contents.append({'role': role, 'parts': [{'text': msg['content']}]})
            
        response = gemini_client.models.generate_content_stream(
            model='gemini-3.6-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_content,
                temperature=0.1,
            )
        )
        
        def stream_generator():
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        full_response = st.write_stream(stream_generator())
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()