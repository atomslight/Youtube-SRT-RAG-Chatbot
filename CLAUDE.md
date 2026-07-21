# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- **Web Server (FastAPI):** `python api.py` or `uvicorn api:app --host 0.0.0.0 --port 8000 --reload`
- **CLI Version:** `python main.py`

### Dependency Management
- **Web Dependencies:** `pip install -r requirements_web.txt`
- **Core AI Dependencies:** `pip install numpy pysrt yt-dlp langchain-core langchain-ollama rank-bm25 sentence-transformers torch`

### AI Model Setup (via Ollama)
- **Generation:** `ollama pull gemma4:31b-cloud`
- **Embedding:** `ollama pull dengcao/Qwen3-Embedding-8B:Q4_K_M`
- **Reranking:** `ollama pull Krakekai/qwen3-reranker-8b`

## Architecture Overview

The project is a local RAG (Retrieval-Augmented Generation) system for YouTube videos, prioritizing privacy by running all AI components via Ollama.

### Core Components
- **`api.py`**: FastAPI server managing the web interface and session-based API endpoints.
- **`main.py`**: CLI entry point for the application.
- **`rag_core.py`**: The heart of the system, implementing the following pipeline:
    - **Ingestion**: Fetches transcripts using `yt-dlp`.
    - **Processing**: Chunks transcripts and generates local embeddings.
    - **Retrieval**: Hybrid search combining BM25 (keyword) and Semantic (embedding) search.
    - **Reranking**: Uses a Cross-Encoder to refine the top retrieved chunks.
    - **Generation**: Local LLM generation with timestamped citations.
- **`static/index.html`**: Vanilla JS frontend using Server-Sent Events (SSE) for real-time streaming of processing and answer updates.

### Data Flow
`YouTube URL` $\rightarrow$ `Transcript Extraction` $\rightarrow$ `Chunking/Indexing` $\rightarrow$ `Hybrid Retrieval` $\rightarrow$ `Reranking` $\rightarrow$ `LLM Generation` $\rightarrow$ `SSE Stream to UI`

### Key Patterns
- **Privacy-First**: All AI tasks (embeddings, reranking, generation) are handled locally through Ollama.
- **Streaming UI**: Uses SSE for the `process` and `ask` endpoints to provide a responsive, streaming user experience.
- **Hybrid Search**: Combines BM25 and Vector search to balance precision and recall before a final reranking stage.
- **Session Management**: Server-side session state is stored in a dictionary indexed by UUIDs in `api.py`.
