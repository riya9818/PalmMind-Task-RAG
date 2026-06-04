import json
import logging
import re
from typing import Optional

import httpx
from sqlalchemy.orm import Session

from config import settings
from db.models import InterviewBooking
from schemas.dto import BookingInfo

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are an assistant that extracts interview booking information from user messages.
Analyze the message and return ONLY a JSON object with these fields:
- "intent": "booking" if the user wants to book an interview, "other" if not
- "name": the person's full name if mentioned, or null
- "email": their email address if mentioned, or null
- "date": the date in YYYY-MM-DD format if mentioned, or null (today is 2026-06-04)
- "time": the time in HH:MM 24-hour format if mentioned, or null
Return ONLY the JSON object, no explanation, no markdown."""


async def detect_and_extract_booking(
    message: str,
    history: str,
) -> Optional[BookingInfo]:
    """
    Use the LLM to detect if the user wants to book an interview
    and extract any booking details from the message.
    Returns BookingInfo if booking intent found, None otherwise.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if history.strip():
        messages.append({"role": "user", "content": f"Conversation so far:\n{history}"})

    messages.append({"role": "user", "content": message})

    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rag-backend",
    }
    payload = {
        "model": settings.OPENROUTER_MODEL,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            response.raise_for_status()
            raw_text = response.json()["choices"][0]["message"]["content"]

        parsed = _parse_json_response(raw_text)
        if not parsed or parsed.get("intent") != "booking":
            return None

        return BookingInfo(
            name=parsed.get("name"),
            email=parsed.get("email"),
            date=parsed.get("date"),
            time=parsed.get("time"),
        )

    except Exception as e:
        logger.error("Booking extraction error: %s", str(e))
        return None


def _parse_json_response(text: str) -> Optional[dict]:
    """Try to parse JSON from LLM response, handling markdown code blocks."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def save_booking(
    session_id: str,
    booking_info: BookingInfo,
    db: Session,
) -> InterviewBooking:
    """Persist booking info to SQLite."""
    booking = InterviewBooking(
        session_id=session_id,
        name=booking_info.name,
        email=booking_info.email,
        date=booking_info.date,
        time=booking_info.time,
    )
    db.add(booking)
    db.commit()
    db.refresh(booking)
    logger.info("Saved booking id=%d for session %s", booking.id, session_id)
    return booking


def build_booking_reply(booking_info: BookingInfo) -> str:
    """Generate a friendly reply asking for any missing booking fields."""
    missing: list[str] = []
    if not booking_info.name:
        missing.append("your full name")
    if not booking_info.email:
        missing.append("your email address")
    if not booking_info.date:
        missing.append("your preferred date (e.g. 2026-06-15)")
    if not booking_info.time:
        missing.append("your preferred time (e.g. 14:00)")

    if missing:
        fields = ", ".join(missing)
        return f"I'd love to help you book an interview! Could you please provide {fields}?"

    return (
        f"Great! I've booked your interview.\n\n"
        f"**Name:** {booking_info.name}\n"
        f"**Email:** {booking_info.email}\n"
        f"**Date:** {booking_info.date}\n"
        f"**Time:** {booking_info.time}\n\n"
        f"We'll be in touch soon!"
    )
