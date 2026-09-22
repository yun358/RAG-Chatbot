import os
import requests
import psycopg2
from psycopg2.extras import execute_batch
from pgvector.psycopg2 import register_vector
from pypdf import PdfReader

# --- Configuration ---
DB_URL = os.getenv("DATABASE_URL", "postgresql://admin:securePass123@rag-db-service:5432/rag_db")
API_URL = os.getenv("EMBEDDING_API_URL", "http://embedding-server:8080/v1/embeddings")

BATCH_SIZE = 8

def build_chapter_map(reader):
    """Maps page numbers to chapter/section titles if an outline exists in the PDF."""
    chapter_map = {}
    try:
        if reader.outline:
            for item in reader.outline:
                if isinstance(item, list):
                    continue
                page_idx = reader.get_destination_page_number(item)
                if page_idx is not None:
                    chapter_map[page_idx] = getattr(item, "title", str(item))
    except Exception:
        pass
    return chapter_map

def extract_and_chunk_pdf(pdf_source, default_doc_name="Document", words_per_chunk=250, overlap=50):
    """Extracts text and generates context-tagged chunks from a file path or file stream."""
    reader = PdfReader(pdf_source)
    chapter_map = build_chapter_map(reader)
    
    current_section = default_doc_name
    enriched_chunks = []
    
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        
        if page_idx in chapter_map:
            # Prepend the Document Name to the Chapter Title to prevent UI collisions
            current_section = f"{default_doc_name} ⏵ {chapter_map[page_idx]}"
            
        text = page.extract_text()
        if not text:
            continue
            
        words = text.split()
        start = 0
        
        # Apply the dynamically injected chunking parameters
        while start < len(words):
            end = start + words_per_chunk
            chunk_body = " ".join(words[start:end])
            enriched_text = f"[{current_section} | Page {page_num}] {chunk_body}"
            enriched_chunks.append(enriched_text)
            start += (words_per_chunk - overlap)
            
    return enriched_chunks

def setup_database(conn, clear_existing=False):
    """Prepares the PostgreSQL schema with vector extension support."""
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
        if clear_existing:
            cur.execute("TRUNCATE TABLE book_chunks RESTART IDENTITY;")
    conn.autocommit = False
    register_vector(conn)

def fetch_embeddings(chunks_batch):
    """Generates 1024-dimensional embeddings from the local embedding server."""
    payload = {
        "input": chunks_batch,
        "model": "bge-m3"
    }
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]

def ingest_document(pdf_source, doc_name="Document", clear_existing=False, progress_callback=None, words_per_chunk=250, overlap=50):
    """Runs end-to-end chunking, embedding generation, and vector insertion."""
    
    # Pass the parameters to the extraction function
    chunks = extract_and_chunk_pdf(
        pdf_source, 
        default_doc_name=doc_name, 
        words_per_chunk=words_per_chunk, 
        overlap=overlap
    )
    
    if not chunks:
        return 0

    conn = psycopg2.connect(DB_URL)
    setup_database(conn, clear_existing=clear_existing)
    
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    inserted_count = 0
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        try:
            embeddings = fetch_embeddings(batch)
            query = "INSERT INTO book_chunks (chunk_text, embedding) VALUES (%s, %s)"
            data_to_insert = list(zip(batch, embeddings))
            
            with conn.cursor() as cur:
                execute_batch(cur, query, data_to_insert)
            conn.commit()
            
            inserted_count += len(batch)
            if progress_callback:
                progress_callback(batch_num, total_batches)
        except Exception as e:
            conn.rollback()
            conn.close()
            raise e
            
    conn.close()
    return inserted_count