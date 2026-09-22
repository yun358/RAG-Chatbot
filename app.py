import os
import re
import requests
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
import streamlit as st
from ingest import ingest_document
import math

st.set_page_config(page_title="RAG Context Engine", page_icon="📄", layout="wide")

# --- Configuration ---
DB_URL = os.getenv("DATABASE_URL", "postgresql://admin:securePass123@rag-db-service:5432/rag_db")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://embedding-server:8080/v1/embeddings")
RERANKER_API_URL = os.getenv("RERANKER_API_URL", "http://reranker-server:8080/v1/rerank")

SCRATCHPAD_FILE = "scratchpad.md"

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieved_context" not in st.session_state:
    st.session_state.retrieved_context = []
    
if not os.path.exists(SCRATCHPAD_FILE):
    with open(SCRATCHPAD_FILE, "w") as f:
        f.write("# Agent Investigation Scratchpad\n- [ ] Initialized session\n")

def read_scratchpad(max_lines=10):
    """Reads the scratchpad, returning only the most recent entries to prevent prompt bloat."""
    if not os.path.exists(SCRATCHPAD_FILE):
        return ""
    with open(SCRATCHPAD_FILE, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Return header plus the last N operational lines
    header = "# Agent Investigation Scratchpad"
    recent_lines = [l for l in lines if not l.startswith("#")][-max_lines:]
    return f"{header}\n" + "\n".join(recent_lines)

def update_scratchpad(new_note):
    with open(SCRATCHPAD_FILE, "a") as f:
        f.write(f"- {new_note}\n")

# --- Database & Metadata Helpers ---
def extract_metadata(chunk_text):
    """Parses injected metadata tags to extract document name, chapter, page, and raw chunk text."""
    doc_name, chapter, page, clean_text = "Unknown", "Unknown", "Unknown", chunk_text
    meta_match = re.match(r'\[(.*?)\s*\|\s*Page\s*(\d+)\]\s*(.*)', chunk_text, re.DOTALL)
    
    if meta_match:
        full_section = meta_match.group(1).strip()
        page = meta_match.group(2).strip()
        clean_text = meta_match.group(3).strip()
        
        if " ⏵ " in full_section:
            parts = full_section.split(" ⏵ ", 1)
            doc_name = parts[0]
            chapter = parts[1]
        else:
            doc_name = full_section
            chapter = "Base Document"
            
    return doc_name, chapter, page, clean_text

def get_unique_documents():
    """Scans the database to find all unique base document names."""
    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            cur.execute("SELECT substring(chunk_text from 1 for 200) FROM book_chunks WHERE chunk_text LIKE '[%';")
            rows = cur.fetchall()
        conn.close()
        
        docs = set()
        for row in rows:
            doc_name, _, _, _ = extract_metadata(row[0])
            if doc_name != "Unknown":
                docs.add(doc_name)
        return sorted(list(docs))
    except Exception:
        return []

def count_tokens_exact(text: str) -> int:
    """Queries the local llama.cpp server to get the exact token count for a string."""
    try:
        # Strip '/v1' from the API URL to hit the root /tokenize endpoint
        base_url = os.getenv("LLM_API_URL", "http://llm-server:8080/v1").replace("/v1", "")
        res = requests.post(
            f"{base_url}/tokenize",
            json={"content": text},
            timeout=2
        )
        return len(res.json().get("tokens", []))
    except Exception as e:
        # Fallback heuristic if the server is temporarily unreachable or cloud API is used
        return int(len(text.split()) * 1.35)

def get_document_chunks(doc_name):
    """Fetches all chunks for a specific document, ordered sequentially by vector insertion ID."""
    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, chunk_text FROM book_chunks 
                WHERE chunk_text LIKE %s OR chunk_text LIKE %s
                ORDER BY id ASC;
            """, (f"[{doc_name} |%", f"[{doc_name} ⏵%"))
            rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return []

# --- Core RAG Functions ---
def rewrite_query(raw_query, history, client, model_name):
    """Resolves conversational context into a standalone query using the active LLM."""
    if not history:
        return raw_query
        
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
    prompt = f"Given this chat history:\n{history_text}\n\nRewrite this user query to be completely standalone, replacing vague pronouns with specific names: {raw_query}\n\nOutput ONLY the rewritten query."
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Rewrite error: {e}")
        return raw_query

def retrieve_and_rerank(query, top_k):
    # 1. Connect and ensure BM25 Schema Exists (Auto-Migration)
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE book_chunks 
            ADD COLUMN IF NOT EXISTS tsv tsvector 
            GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED;
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_book_chunks_tsv ON book_chunks USING GIN(tsv);")
    conn.autocommit = False
    register_vector(conn)
    
    # 2. Embed query for dense search
    res = requests.post(EMBEDDING_API_URL, json={"input": query, "model": "bge-m3"})
    query_vector = res.json()["data"][0]["embedding"]
    
    # 3. Hybrid Search: Reciprocal Rank Fusion (BM25 + Vector)
    with conn.cursor() as cur:
        cur.execute("""
            WITH vector_candidates AS (
                SELECT id, chunk_text,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS v_rank
                FROM book_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT 20
            ),
            keyword_candidates AS (
                SELECT id, chunk_text,
                       ROW_NUMBER() OVER (ORDER BY ts_rank(tsv, plainto_tsquery('english', %s)) DESC) AS k_rank,
                       ts_rank(tsv, plainto_tsquery('english', %s)) AS k_score
                FROM book_chunks
                WHERE tsv @@ plainto_tsquery('english', %s)
                ORDER BY k_score DESC
                LIMIT 20
            )
            SELECT 
                COALESCE(v.chunk_text, k.chunk_text) AS chunk_text,
                v.v_rank,
                k.k_rank,
                (COALESCE(1.0 / (60 + v.v_rank), 0.0) + COALESCE(1.0 / (60 + k.k_rank), 0.0)) AS rrf_score
            FROM vector_candidates v
            FULL OUTER JOIN keyword_candidates k ON v.id = k.id
            ORDER BY rrf_score DESC
            LIMIT 20;
        """, (query_vector, query_vector, query, query, query))
        candidates = cur.fetchall()  
    conn.close()
    
    if not candidates:
        return []

    candidate_texts = [c[0] for c in candidates]

    # 4. Cross-encoder scoring
    rerank_res = requests.post(RERANKER_API_URL, json={
        "query": query, 
        "documents": candidate_texts, 
        "model": "bge-reranker-v2-m3"
    })
    results = rerank_res.json().get("results", [])
    sorted_results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
    
    # 5. Package results with Hybrid Metadata
    top_chunks = []
    for r in sorted_results[:top_k]:
        idx = r["index"]
        text, v_rank, k_rank, rrf_score = candidates[idx]
        doc_name, chapter, page, clean_text = extract_metadata(text)
        
        # Convert raw logit to a percentage using the sigmoid function
        raw_score = float(r["relevance_score"])
        normalized_score = (1 / (1 + math.exp(-raw_score))) * 100
        
        top_chunks.append({
            "text": text,
            "clean_text": clean_text,
            "doc_name": doc_name,
            "chapter": chapter,
            "page": page,
            "vector_rank": v_rank,
            "keyword_rank": k_rank,
            "rrf_score": round(float(rrf_score), 4),
            "rerank_score": f"{round(normalized_score, 1)}%", # Now outputs a percentage
            "match_type": "Hybrid BM25 + Vector + Reranked"
        })
        
    return top_chunks

# --- Global Sidebar (Navigation, Model Selection, Ingestion) ---
with st.sidebar:
    app_mode = st.radio("🧭 Navigation", ["💬 Chat & Retrieval", "📚 Document Explorer"])
    
    # Inference Backend Switcher
    st.divider()
    st.header("🤖 Inference Provider")
    inference_mode = st.radio("LLM Backend", ["Local (Qwen3-8B-Q5_K_M)", "Cloud API"])
    
    if inference_mode == "Cloud API":
        st.caption("Works with Groq, OpenRouter, DeepSeek, or OpenAI.")
        api_base_url = st.text_input("Base URL", value="https://api.groq.com/openai/v1")
        api_key_input = st.text_input("API Key", type="password", help="Enter your secret API key")
        api_model_name = st.text_input("Model Name", value="llama-3.3-70b-versatile")
        
        if not api_key_input:
            st.warning("Please provide an API key to run queries.")
            
        active_api_key = api_key_input if api_key_input.strip() else "dummy-key"
    else:
        api_base_url = os.getenv("LLM_API_URL", "http://llm-server:8080/v1")
        active_api_key = "sk-no-key-required"
        api_key_input = active_api_key
        api_model_name = "gemma-4-E4B"

    llm_client = OpenAI(base_url=api_base_url, api_key=active_api_key)

    st.divider()
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF to vectorize", type=["pdf"])
    
    col1, col2 = st.columns(2)
    with col1:
        chunk_size_setting = st.number_input("Words per Chunk", min_value=50, max_value=2000, value=250, step=50)
    with col2:
        overlap_setting = st.number_input("Overlap", min_value=0, max_value=max(0, chunk_size_setting-1), value=50, step=10)

    clear_db = st.checkbox("Clear existing database on upload", value=False)
    
    if uploaded_file and st.button("Vectorize Document", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current_batch, total_batches):
            pct = current_batch / total_batches
            progress_bar.progress(pct)
            status_text.text(f"Processing batch {current_batch} of {total_batches}...")

        try:
            with st.spinner("Extracting text and generating embeddings..."):
                count = ingest_document(
                    uploaded_file,
                    doc_name=uploaded_file.name.replace(".pdf", ""),
                    clear_existing=clear_db,
                    progress_callback=update_progress,
                    words_per_chunk=chunk_size_setting,
                    overlap=overlap_setting
                )
            st.success(f"Successfully vectorized and stored {count} chunks!")
            st.rerun()
        except Exception as e:
            st.error(f"Ingestion failed: {e}")

# ==========================================
# PAGE 1: Chat & Retrieval Interface
# ==========================================
if app_mode == "💬 Chat & Retrieval":
    st.title("📄 Document Intelligence & Context Engine")
    
    with st.sidebar:
        st.divider()
        st.header("⚙️ Retrieval Settings")
        top_k_setting = st.number_input("Top-K Chunks to Retrieve", min_value=1, max_value=100, value=10, step=1)
        st.caption("Using Hybrid (Dense + BM25 Lexical) via RRF.")

        st.divider()
        st.header("🧠 Agent Memory")
        st.markdown("### Scratchpad")
        st.text_area("Internal Notes", value=read_scratchpad(), height=150, disabled=True)
        
        st.markdown("### Active Context & Scores")
        for i, item in enumerate(st.session_state.retrieved_context):

            # 1. Inject an invisible HTML anchor point
            st.markdown(f"<div id='chunk-{i+1}'></div>", unsafe_allow_html=True)

            with st.expander(f"Chunk {i+1} | Re-Rank: {item['rerank_score']}"):
                c1, c2 = st.columns(2)
                c1.metric("Rerank Score", f"{item['rerank_score']}")
                c2.metric("Hybrid RRF", f"{item['rrf_score']}")
                
                c3, c4 = st.columns(2)
                v_rank_display = f"#{item['vector_rank']}" if item['vector_rank'] else "N/A"
                k_rank_display = f"#{item['keyword_rank']}" if item['keyword_rank'] else "N/A"
                c3.metric("Vector Rank", v_rank_display)
                c4.metric("BM25 Rank", k_rank_display)
                
                st.caption(f"Type: `{item['match_type']}`")
                st.markdown(f"**Doc:** {item['doc_name']} &nbsp;&nbsp;|&nbsp;&nbsp; **Sec:** {item['chapter']} &nbsp;&nbsp;|&nbsp;&nbsp; **Page:** {item['page']}")
                st.markdown("---")
                st.write(item["clean_text"])

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the documents..."):
        
        if inference_mode == "Cloud API" and (not api_key_input or api_key_input == "dummy-key"):
            st.error("⚠️ Please enter a valid API Key in the left sidebar before submitting queries.")
            st.stop()
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Resolving conversational context..."):
                standalone_query = rewrite_query(prompt, st.session_state.messages[:-1], llm_client, api_model_name)
                st.caption(f"*(Resolved Query: {standalone_query})*")
                
            with st.spinner("Searching and Reranking..."):
                retrieved_facts = retrieve_and_rerank(standalone_query, int(top_k_setting))
                st.session_state.retrieved_context = retrieved_facts
                update_scratchpad(f"Queried: '{standalone_query}' -> Retrieved {len(retrieved_facts)} chunks.")
                
            # --- Dynamic Prompt Budgeting ---
            MAX_PROMPT_TOKENS = 6500  # Leaves ~1,692 tokens free for generation on an 8192 context window
            
            # 1. Truncate scratchpad to prevent unbounded growth
            scratchpad_lines = read_scratchpad().strip().split("\n")
            recent_scratchpad = "\n".join(scratchpad_lines[-5:])
            
            # 2. Build the static foundation of the prompt
            base_system_content = (
                "You are a precise document assistant. Answer accurately using ONLY the provided retrieved chunks. "
                "If data is missing, state it clearly.\n"
                "CRITICAL INSTRUCTION: You MUST cite your sources inline for every claim you make using Markdown hyperlinks. "
                "You must include BOTH the brackets and the parentheses. "
                "Format the links exactly like this to match the UI anchors: [Chunk 1: Page 42](#chunk-1) or [Chunk 2: Page 9](#chunk-2). \n\n"
                "Keep your synthesized response concise, focused, and under 300 words. "
                "Do not use external URLs, only use the '#' anchor.\n\n"
                f"--- RECENT SCRATCHPAD ---\n{recent_scratchpad}\n\n--- RETRIEVED CHUNKS ---\n"
            )
            
            # 3. Calculate initial token baseline (System Prompt + Chat History)
            system_content = base_system_content
            current_tokens = count_tokens_exact(system_content)
            
            history_contents = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages[-5:]]
            for msg in history_contents:
                current_tokens += count_tokens_exact(msg["content"])
            
            # 4. Iteratively add chunks only if they fit the budget
            included_chunks = 0
            for i, item in enumerate(retrieved_facts):
                chunk_text = (
                    f"Chunk {i+1}:\n"
                    f"Metadata: Document '{item['doc_name']}', Section '{item['chapter']}', Page {item['page']}\n"
                    f"Content: {item['clean_text']}\n\n"
                )
                chunk_tokens = count_tokens_exact(chunk_text)
                
                if current_tokens + chunk_tokens > MAX_PROMPT_TOKENS:
                    st.warning(f"⚠️ Token budget reached. Truncated {len(retrieved_facts) - included_chunks} chunks to prevent server crash.")
                    break
                    
                system_content += chunk_text
                current_tokens += chunk_tokens
                included_chunks += 1
                
            # 5. Assemble final payload
            contents = [{"role": "system", "content": system_content}] + history_contents
                
            try:
                response = llm_client.chat.completions.create(
                    model=api_model_name,
                    messages=contents,
                    temperature=0.1,
                    max_tokens=1200,
                    stream=True
                )
                
                def stream_generator():
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                            
                full_response = st.write_stream(stream_generator())
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.rerun()
                
            except Exception as e:
                st.error(f"Generation failed: {e}")

# ==========================================
# PAGE 2: Document Explorer Interface
# ==========================================
elif app_mode == "📚 Document Explorer":
    st.title("📚 Document Explorer")
    st.markdown("Inspect vectorized chunks for each document, ordered by their sequential insertion into the database.")
    
    docs = get_unique_documents()
    
    if not docs:
        st.info("The database is currently empty. Upload a document in the sidebar to get started!")
    else:
        selected_doc = st.selectbox("Select a Document to Inspect", docs)
        
        chunks = get_document_chunks(selected_doc)
        st.metric(f"Total Chunks in '{selected_doc}'", len(chunks))
        st.divider()
        
        for chunk_id, chunk_text in chunks:
            doc_name, chapter, page, clean_text = extract_metadata(chunk_text)
            
            with st.expander(f"Chunk ID: {chunk_id} | {chapter} (Page {page})"):
                st.write(clean_text)