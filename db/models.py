from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)
    chunking_strategy = Column(String(20), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete")


class Chunk(Base):
    __tablename__ = "chunks"
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    vector_id = Column(String(100), nullable=False)

    document = relationship("Document", back_populates="chunks")


class InterviewBooking(Base):
    __tablename__ = "interview_bookings"
    __allow_unmapped__ = True

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False)
    name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    date = Column(String(20), nullable=True)
    time = Column(String(10), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)