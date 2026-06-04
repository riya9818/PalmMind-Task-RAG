import logging
from typing import Any

import httpx

from config import settings
from services.embeddings import embed_text
from services.vector_store import search_similar

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


async def generate_answer(
    question: str,
    chat_history: str,
    top_k: int = 5,
) -> str:
    """
    Custom RAG pipeline 

    1. Embed the user's question
    2. Search Qdrant for the most relevant chunks
    3. Build a prompt with context + history + question
    4. Send to OpenRouter and return the answer
    """

    # Step 1 — embed the query
    query_vector = embed_text(question)

    # Step 2 — retrieve relevant chunks from Qdrant
    retrieved: list[dict[str, Any]] = search_similar(query_vector, top_k=top_k)

    # Step 3 — build context string from retrieved chunks
    if retrieved:
        context_parts = [chunk.get("content", "") for chunk in retrieved]
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = "No relevant documents found."

    # Step 4 — build messages list for the LLM
    messages = _build_messages(context=context, history=chat_history, question=question)

    # Step 5 — call OpenRouter
    answer = await _call_openrouter(messages)
    return answer


def _build_messages(context: str, history: str, question: str) -> list[dict[str, str]]:
    """
    Build the messages array for the OpenRouter chat completions API.
    Structure: system message → optional history → user question
    """
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions based on provided document context. "
                "Always answer from the context when possible. If the context doesn't contain the answer, say so honestly.\n\n"
                f"Context from documents:\n{context}"
            ),
        }
    ]

    # Inject previous conversation turns from Redis
    if history.strip():
        for line in history.strip().split("\n"):
            if line.startswith("User: "):
                messages.append({"role": "user", "content": line[6:]})
            elif line.startswith("Assistant: "):
                messages.append({"role": "assistant", "content": line[11:]})

    # Add the current question
    messages.append({"role": "user", "content": question})
    return messages


async def _call_openrouter(messages: list[dict[str, str]]) -> str:
    """Send messages to OpenRouter and return the reply text."""
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rag-backend",  # required by OpenRouter
    }
    payload = {
        "model": settings.OPENROUTER_MODEL,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except httpx.ConnectError:
            logger.error("Cannot connect to OpenRouter")
            return "I'm having trouble connecting to the AI model. Please check your API key and internet connection."
        except httpx.TimeoutException:
            logger.error("OpenRouter request timed out")
            return "The AI model took too long to respond. Please try again."
        except KeyError:
            logger.error("Unexpected OpenRouter response format: %s", response.text)
            return "Received an unexpected response from the AI model."
        except Exception as e:
            logger.error("OpenRouter error: %s", str(e))
            return f"An error occurred while generating a response: {str(e)}"
