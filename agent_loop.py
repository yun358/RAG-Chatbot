import os
import requests
import psycopg2
from pgvector.psycopg2 import register_vector
from google import genai
from google.genai import types

# Initialize Gemini Client
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- Configuration ---
DB_URL = os.getenv("DATABASE_URL", "postgresql://admin:maryHAIL1990@our-hail-mary-db-service:5432/hail_mary_db")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://embedding-server:8080/v1/embeddings")
RERANKER_API_URL = os.getenv("RERANKER_API_URL", "http://reranker-server:8080/v1/rerank")

SCRATCHPAD_FILE = "scratchpad.md"

def init_scratchpad():
    """WRITE STEP: Initialize an external scratchpad if it doesn't exist."""
    if not os.path.exists(SCRATCHPAD_FILE):
        with open(SCRATCHPAD_FILE, "w") as f:
            f.write("# Agent Investigation Scratchpad\n- [ ] Initialized session\n")

def read_scratchpad():
    """WRITE STEP: Read current external notes."""
    if os.path.exists(SCRATCHPAD_FILE):
        with open(SCRATCHPAD_FILE, "r") as f:
            return f.read()
    return "No active notes."

def update_scratchpad(new_note):
    """WRITE STEP: Append state externally to preserve working memory."""
    with open(SCRATCHPAD_FILE, "a") as f:
        f.write(f"- {new_note}\n")
    print(f"[Scratchpad Updated]: {new_note}")

def compress_history(chat_history):
    """COMPRESS STEP: Keep signal, drop the rest."""
    if len(chat_history) <= 4:
        return chat_history
    
    older_messages = chat_history[:-4]
    recent_messages = chat_history[-4:]
    
    summary_text = f"[System Compression: Summary of previous {len(older_messages)} turns regarding plot progress.]"
    return [{"role": "system", "content": summary_text}] + recent_messages

def retrieve_and_rerank(query):
    """SELECT STEP: Vector retrieval + Cross-Encoder reranking."""
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    
    # 1. Embed query
    res = requests.post(EMBEDDING_API_URL, json={"input": query, "model": "bge-m3"})
    query_vector = res.json()["data"][0]["embedding"]
    
    # 2. Candidate retrieval from PostgreSQL
    with conn.cursor() as cur:
        # THE FIX: Added ::vector cast right after %s
        cur.execute("SELECT chunk_text FROM book_chunks ORDER BY embedding <=> %s::vector LIMIT 15;", (query_vector,))
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
    
    # Return top 3 high-signal chunks
    return [candidates[r['index']] for r in sorted_results[:3]]

def synthesize_answer(context_stack):
    """GENERATION STEP: Feed the isolated context stack to Gemini."""
    system_content = f"{context_stack['instructions']}\n\n"
    system_content += f"--- SCRATCHPAD (Internal Notes) ---\n{context_stack['scratchpad_notes']}\n\n"
    system_content += "--- RETRIEVED BOOK EXCERPTS ---\n"
    for i, fact in enumerate(context_stack['retrieved_facts']):
        system_content += f"Excerpt {i+1}:\n{fact}\n\n"

    # Build conversation contents
    contents = []
    for msg in context_stack['active_history']:
        if msg["role"] == "system":
            continue
        role = "user" if msg["role"] == "user" else "model"
        contents.append({'role': role, 'parts': [{'text': msg['content']}]})
        
    contents.append({'role': 'user', 'parts': [{'text': context_stack['user_input']}]})
    
    print("\n[Gemini is typing...]\n")
    
    response = gemini_client.models.generate_content_stream(
        model="gemini-3.6-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_content,
            temperature=0.1,
        ),
    )
    
    final_answer = ""
    for chunk in response:
        if chunk.text:
            print(chunk.text, end="", flush=True)
            final_answer += chunk.text
            
    print("\n")
    return final_answer

def run_agent_turn(query, chat_history):
    print(f"\nUser Query: {query}")
    init_scratchpad()
    
    active_history = compress_history(chat_history)
    print("Selecting relevant context chunks...")
    retrieved_facts = retrieve_and_rerank(query)
    scratchpad_notes = read_scratchpad()
    
    context_stack = {
        "instructions": "You are a precise literary assistant for Project Hail Mary. Answer accurately using ONLY the provided retrieved facts and scratchpad. If data is missing, state it clearly.",
        "retrieved_facts": retrieved_facts,
        "scratchpad_notes": scratchpad_notes,
        "active_history": active_history,
        "user_input": query
    }
    
    # Generate the actual streaming response
    agent_response = synthesize_answer(context_stack)
    
    # Update scratchpad state externally
    update_scratchpad(f"Queried: '{query}' -> Retrieved {len(retrieved_facts)} high-signal segments.")
    
    # Append to active history
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": agent_response})
    
    return chat_history

if __name__ == "__main__":
    history = []
    history = run_agent_turn("What happens in chapter 17?", history)
    history = run_agent_turn("What chapter does that happen in?", history)