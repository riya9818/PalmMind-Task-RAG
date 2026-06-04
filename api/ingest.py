import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from config import settings
from db.models import Chunk, Document
from db.session import get_db
from schemas.dto import IngestResponse
from services.chunking import ChunkingStrategy, get_chunks
from services.document_processor import extract_text
from services.embeddings import embed_batch
from services.vector_store import store_chunks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Ingestion"])

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@router.post("/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    strategy: Annotated[
        ChunkingStrategy,
        Query(description="Chunking strategy: 'fixed' (fixed-size with overlap) or 'semantic' (paragraph-based)"),
    ] = "fixed",
    db: Session = Depends(get_db),
) -> IngestResponse:
    """
    Upload a .pdf or .txt file.

    - Extracts text from the file
    - Applies the selected chunking strategy
    - Generates embeddings for each chunk
    - Stores vectors in Qdrant
    - Saves document metadata and chunks in SQLite

    **strategy** options:
    - `fixed` — splits into 500-char chunks with 50-char overlap
    - `semantic` — splits on paragraph breaks, merges short paragraphs
    """

    # Validate file extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Only .pdf and .txt are allowed.",
        )

    # Save uploaded file to disk
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    contents = await file.read()
    file_path.write_bytes(contents)
    logger.info("Saved uploaded file: %s (%d bytes)", file.filename, len(contents))

    try:
        # Extract raw text
        raw_text = extract_text(str(file_path))
        if not raw_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract any text from the uploaded file.",
            )

        # Chunk the text
        chunks = get_chunks(raw_text, strategy=strategy)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No chunks were produced from the document.",
            )
        logger.info("Produced %d chunks using strategy '%s'", len(chunks), strategy)

        # Save document record to DB first (we need its id for Qdrant metadata)
        doc = Document(
            filename=file.filename,
            file_type=suffix.lstrip("."),
            chunking_strategy=strategy,
        )
        db.add(doc)
        db.flush()  # assigns doc.id without full commit

        # Generate embeddings for all chunks in one batch call
        vectors = embed_batch(chunks)

        # Store vectors in Qdrant
        vector_ids = store_chunks(
            chunks=chunks,
            vectors=vectors,
            document_id=doc.id,
            filename=file.filename,
        )

        # Save chunk records to SQLite
        for idx, (chunk_text, vid) in enumerate(zip(chunks, vector_ids)):
            db.add(Chunk(
                document_id=doc.id,
                chunk_index=idx,
                content=chunk_text,
                vector_id=vid,
            ))

        db.commit()
        logger.info("Ingestion complete: document_id=%d, %d chunks", doc.id, len(chunks))

        return IngestResponse(
            document_id=doc.id,
            filename=file.filename,
            chunks_created=len(chunks),
            chunking_strategy=strategy,
            message="Document ingested successfully.",
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )
    finally:
        # Clean up the file from disk after processing
        if file_path.exists():
            os.remove(file_path)
