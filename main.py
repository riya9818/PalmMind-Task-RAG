import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import chat, health, ingest
from db.session import init_db
from services.vector_store import ensure_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Backend API",
    description="Document ingestion and conversational RAG with interview booking support.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(health.router)


@app.on_event("startup")
async def startup() -> None:
    """Run setup tasks when the server starts."""
    logger.info("Initialising database tables...")
    init_db()

    logger.info("Ensuring Qdrant collection exists...")
    ensure_collection()

    logger.info("RAG Backend is ready.")
