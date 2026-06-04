from datetime import datetime
from sqlalchemy import Column, Integer, ForeignKey, String, DateTime, Text
from sqlalchemy.orm import DeclarativeBase,relationship

class Base(DeclarativeBase):
    pass

class Document(Base):
    """Stores metadata about each uploaded file."""

    __tablename__ = "documents"

    id: int = Column(Integer, primary_key=True, index=True)
    filename: str = Column(String(255), nullable=False)
    file_type: str = Column(String(10), nullable=False)   # "pdf" or "txt"
    chunking_strategy: str = Column(String(20), nullable=False)  # "fixed" or "semantic"
    upload_date: datetime = Column(DateTime, default=datetime.utcnow)

    chunks: list["Chunk"] = relationship("Chunk", back_populates="document", cascade="all, delete")


class Chunk(Base):
    """Each chunk of text extracted from a document."""

    __tablename__ = "chunks"

    id: int = Column(Integer, primary_key=True, index=True)
    document_id: int = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index: int = Column(Integer, nullable=False)
    content: str = Column(Text, nullable=False)
    vector_id: str = Column(String(100), nullable=False)  # ID stored in Qdrant

    document: Document = relationship("Document", back_populates="chunks")


class InterviewBooking(Base):
    """Stores interview bookings collected via the chatbot."""

    __tablename__ = "interview_bookings"

    id: int = Column(Integer, primary_key=True, index=True)
    session_id: str = Column(String(100), nullable=False)
    name: str = Column(String(255), nullable=True)
    email: str = Column(String(255), nullable=True)
    date: str = Column(String(20), nullable=True)   # e.g. "2026-06-10"
    time: str = Column(String(10), nullable=True)   # e.g. "14:00"
    created_at: datetime = Column(DateTime, default=datetime.utcnow)