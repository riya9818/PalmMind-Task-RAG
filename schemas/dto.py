from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, EmailStr


# ── Ingestion 

class IngestResponse(BaseModel):
    document_id: int
    filename: str
    chunks_created: int
    chunking_strategy: str
    message: str


# ── Chat 

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    booking_detected: bool = False


# ── Booking 
class BookingInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None


class BookingResponse(BaseModel):
    id: int
    session_id: str
    name: Optional[str]
    email: Optional[str]
    date: Optional[str]
    time: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ── Health

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    services: dict[str, str]
