import os
import requests
import psycopg2
from psycopg2.extras import execute_batch
from pgvector.psycopg2 import register_vector
from pypdf import PdfReader

# --- Configuration ---
PDF_PATH = "Project_Hail_Mary.pdf"
DB_URL = os.getenv("DATABASE_URL", "postgresql://admin:maryHAIL1990@our-hail-mary-db-service:5432/hail_mary_db")
API_URL = os.getenv("EMBEDDING_API_URL", "http://embedding-server:8080/v1/embeddings")

# Chunking parameters
WORDS_PER_CHUNK = 250
OVERLAP = 50
BATCH_SIZE = 8

def build_chapter_map(reader):
    """Maps the exact zero-indexed page number to the chapter title using the PDF's outline."""
    chapter_map = {}
    if reader.outline:
        for item in reader.outline:
            # Skip nested lists (sub-headings)
            if isinstance(item, list):
                continue
            
            page_idx = reader.get_destination_page_number(item)
            if page_idx is not None:
                chapter_map[page_idx] = item.title 
                
    return chapter_map

def extract_and_chunk_with_metadata(filepath):
    print(f"Reading {filepath} and extracting structural metadata...")
    reader = PdfReader(filepath)
    chapter_map = build_chapter_map(reader)
    
    current_chapter_title = "Intro/Front Matter"
    enriched_chunks = []
    
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1 # Human-readable page number
        
        # Update chapter if we've crossed a mapped boundary
        if page_idx in chapter_map:
            current_chapter_title = chapter_map[page_idx]
            print(f"Entered {current_chapter_title} on Page {page_num}")
            
        text = page.extract_text()
        if not text:
            continue
            
        words = text.split()
        start = 0
        
        # Sliding window chunking with overlap
        while start < len(words):
            end = start + WORDS_PER_CHUNK
            chunk_body = " ".join(words[start:end])
            
            # Context Header Injection
            enriched_text = f"[{current_chapter_title} | Page {page_num}] {chunk_body}"
            enriched_chunks.append(enriched_text)
            
            start += (WORDS_PER_CHUNK - OVERLAP)
            
    print(f"Generated {len(enriched_chunks)} context-enriched chunks.")
    return enriched_chunks

def setup_database(conn):
    print("Preparing PostgreSQL schema...")
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS book_chunks (
                id SERIAL PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                embedding VECTOR(1024)
            );
        """)
        # Clear the table so we can re-run this script safely
        cur.execute("TRUNCATE TABLE book_chunks RESTART IDENTITY;")
        
    conn.autocommit = False 
    register_vector(conn)

def fetch_embeddings(chunks_batch):
    payload = {
        "input": chunks_batch,
        "model": "bge-m3" 
    }
    
    response = requests.post(API_URL, json=payload)
    response.raise_for_status() 
    
    data = response.json()
    embeddings = [item["embedding"] for item in data["data"]]
    return embeddings

def ingest_data():
    chunks = extract_and_chunk_with_metadata(PDF_PATH)
    
    print("Connecting to database...")
    conn = psycopg2.connect(DB_URL)
    setup_database(conn)
    
    print(f"Connecting to embedding API at {API_URL}...")
    total_batches = (len(chunks) // BATCH_SIZE) + 1
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"Processing batch {batch_num} of {total_batches}...")
        
        try:
            embeddings = fetch_embeddings(batch)
            query = "INSERT INTO book_chunks (chunk_text, embedding) VALUES (%s, %s)"
            data_to_insert = list(zip(batch, embeddings))
            
            with conn.cursor() as cur:
                execute_batch(cur, query, data_to_insert)
            conn.commit()
            
        except Exception as e:
            print(f"Failed on batch {batch_num}. Error: {e}")
            conn.rollback()
            break
            
    conn.close()
    print("Ingestion completely finished! Data is ready for RAG.")

if __name__ == "__main__":
    ingest_data()