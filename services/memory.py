import json
import logging

import redis

from config import settings

logger = logging.getLogger(__name__)

# Each session's history expires after 2 hours of inactivity
SESSION_TTL_SECONDS = 7200


def _get_client() -> redis.Redis:
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        decode_responses=True,
    )


def _session_key(session_id: str) -> str:
    return f"chat:session:{session_id}"


def get_history(session_id: str) -> list[dict[str, str]]:
    """
    Retrieve the full conversation history for a session.
    Returns a list of {"role": "user"/"assistant", "content": "..."} dicts.
    """
    client = _get_client()
    raw = client.get(_session_key(session_id))
    if not raw:
        return []
    return json.loads(raw)


def append_message(session_id: str, role: str, content: str) -> None:
    """
    Add a single message to the session history and reset the TTL.
    role should be "user" or "assistant".
    """
    history = get_history(session_id)
    history.append({"role": role, "content": content})

    client = _get_client()
    client.setex(
        _session_key(session_id),
        SESSION_TTL_SECONDS,
        json.dumps(history),
    )


def clear_history(session_id: str) -> None:
    """Delete a session's chat history."""
    client = _get_client()
    client.delete(_session_key(session_id))
    logger.info("Cleared history for session %s", session_id)


def format_history_for_prompt(history: list[dict[str, str]]) -> str:
    """
    Convert history list into a plain-text block for the LLM prompt.
    """
    lines: list[str] = []
    for msg in history:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)
