import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from db.session import get_db
from schemas.dto import ChatRequest, ChatResponse
from services.booking import build_booking_reply, detect_and_extract_booking, save_booking
from services.memory import append_message, format_history_for_prompt, get_history
from services.rag import generate_answer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Send a message and get an AI answer using RAG.

    - Loads conversation history from Redis (multi-turn support)
    - Checks if the message is an interview booking request
    - If booking: extracts details using LLM, asks for missing info, saves to DB
    - Otherwise: runs the custom RAG pipeline (embed → search Qdrant → prompt → ans)
    - Saves the exchange back to Redis for the next turn

    **session_id**: use any unique string per user/conversation (e.g. a UUID)
    """
    session_id = request.session_id
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty.",
        )

    # Load existing conversation history from Redis
    history = get_history(session_id)
    history_text = format_history_for_prompt(history)

    # Check if the user is trying to book an interview
    booking_info = await detect_and_extract_booking(
        message=user_message,
        history=history_text,
    )

    booking_detected = booking_info is not None

    if booking_detected:
        # Build a reply asking for any missing fields, or confirm the booking
        reply = build_booking_reply(booking_info)

        # If all required fields are present, save the booking
        if booking_info.name and booking_info.email and booking_info.date and booking_info.time:
            save_booking(session_id=session_id, booking_info=booking_info, db=db)
            logger.info("Interview booking saved for session %s", session_id)

    else:
        # Standard RAG flow
        reply = await generate_answer(
            question=user_message,
            chat_history=history_text,
        )

    # Persist this exchange to Redis for future turns
    append_message(session_id, role="user", content=user_message)
    append_message(session_id, role="assistant", content=reply)

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        booking_detected=booking_detected,
    )
