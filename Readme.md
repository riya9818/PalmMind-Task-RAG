# RAG Backend API

FastAPI backend implementing document ingestion, custom Retrieval-Augmented Generation (RAG), Redis-based conversation memory, and LLM-powered interview booking.

## Features

* Upload `.pdf` and `.txt` documents
* Two chunking strategies: Fixed and Semantic
* Local embeddings using Sentence Transformers
* Vector storage with Qdrant
* Redis-backed multi-turn chat memory
* Custom RAG pipeline (no RetrievalQAChain)
* Interview booking using LLM extraction
* SQLite metadata and booking storage

## Tech Stack

* FastAPI
* Qdrant
* Redis
* SQLite + SQLAlchemy
* Sentence Transformers
* OpenRouter (Llama 3.3)

## Project Structure

```text
api/
services/
db/
schemas/
main.py
config.py
```

## Setup

1. Clone repository
2. Create `.env` from `.env.example`
3. Add OpenRouter API key
4. Run:

```bash
docker compose up --build
```

Swagger UI:

```text
http://localhost:8000/docs
```

## API Endpoints

### POST /documents/upload

* Accepts PDF/TXT files
* Extracts text
* Applies selected chunking strategy
* Generates embeddings
* Stores vectors in Qdrant
* Stores metadata in SQLite

### POST /chat

* Multi-turn conversation support
* Redis-backed chat memory
* Custom RAG retrieval pipeline
* Interview booking detection and storage

### GET /health

Checks connectivity to Qdrant, Redis, and OpenRouter.

## RAG Workflow

```text
User Query
    ↓
Embedding Generation
    ↓
Qdrant Similarity Search
    ↓
Context Retrieval
    ↓
Prompt Construction
    ↓
LLM Response
    ↓
Redis Memory Update
```

## Assignment Requirements Coverage

| Requirement             | Status   |
| ----------------------- | -------- |
| PDF/TXT Upload          | ✓        |
| Two Chunking Strategies | ✓        |
| Embedding Generation    | ✓        |
| Vector Database         | ✓ Qdrant |
| Metadata Storage        | ✓ SQLite |
| Custom RAG              | ✓        |
| No RetrievalQAChain     | ✓        |
| Redis Chat Memory       | ✓        |
| Multi-turn Conversation | ✓        |
| Interview Booking       | ✓        |
| Booking Persistence     | ✓        |
| Modular Architecture    | ✓        |
