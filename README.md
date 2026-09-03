# Document Intelligence & Context Engine

A completely local, privacy-first Retrieval-Augmented Generation (RAG) pipeline designed for precise document analysis. This engine is optimized to run locally via WSL 2, utilizing Docker for seamless container orchestration. It prevents mid-sentence cutoffs and out-of-memory errors through exact-token prompt budgeting and hardware-aware quantization.

## Architecture & Retrieval Pipeline

The system follows a strict two-stage retrieval process to guarantee factual accuracy and high informational density before assembling the prompt. 

*   **Hybrid Database Retrieval:** Combines dense vector search (cosine similarity) and sparse lexical search (BM25) entirely within PostgreSQL.
*   **Reciprocal Rank Fusion:** Merges the candidate lists natively in SQL using the standard 60 constant to reward consensus between retrievers:
    $$\text{Score}(d) = \frac{1}{60 + \text{rank}_{\text{vector}}} + \frac{1}{60 + \text{rank}_{\text{bm25}}}$$
*   **Cross-Encoder Reranking:** Applies full cross-attention to the top 20 database candidates to determine the final context priority. The raw logit is normalized for the UI using the sigmoid function:
    $$\text{Confidence} = \left( \frac{1}{1 + e^{-\text{logit}}} \right) \times 100$$
*   **Dynamic Prompt Budgeting:** Queries the local `llama.cpp` inference server's native `/tokenize` API to count Byte-Pair Encoding (BPE) tokens exactly, iteratively injecting chunks until a strict 6,500-token budget is reached.

## System Setup (Docker)

Ensure your `docker-compose.yml` is configured to map the GGUF models correctly and that your `scratchpad.md` has read/write permissions.

*   **Model Requirements:** Place `Qwen3-8B-Q5_K_M.gguf`, `bge-m3-FP16.gguf`, and `bge-reranker-v2-m3-Q4_K_M.gguf` in the `/models` directory.
*   **Deployment:** Boot the multi-container environment (Database, Embedding Server, Reranker Server, LLM Server, and Agent) by running:
    ```bash
    docker compose up -d --force-recreate
    ```
*   **Access:** Open the Streamlit frontend via `http://localhost:8501`. 

## Hardware & VRAM Optimization

This pipeline is engineered to stay strictly within an 8 GB VRAM limit (e.g., NVIDIA RTX 4060) while maintaining a massive context window.

*   **LLM Offloading:** The Qwen3 8B model is fully offloaded to the GPU (`-ngl 99`).
*   **Auxiliary CPU Execution:** The embedding model and cross-encoder operate in system RAM to preserve GPU memory for text generation.
*   **KV Cache Quantization:** Extends the LLM context window safely up to 12,288 tokens by applying 8-bit quantization to the key-value cache (`-ctk q8_0`, `-ctv q8_0`).
