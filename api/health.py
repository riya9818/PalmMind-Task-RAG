import logging

import httpx
import redis
from fastapi import APIRouter
from qdrant_client import QdrantClient

from config import settings
from schemas.dto import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check connectivity to all dependent services."""
    services: dict[str, str] = {}

    # Qdrant
    try:
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=3)
        client.get_collections()
        services["qdrant"] = "ok"
    except Exception as e:
        services["qdrant"] = f"error: {str(e)}"

    # Redis
    try:
        r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, socket_timeout=3)
        r.ping()
        services["redis"] = "ok"
    except Exception as e:
        services["redis"] = f"error: {str(e)}"

    # OpenRouter (just check internet connectivity)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("https://openrouter.ai/api/v1/models",
                                    headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"})
            services["openrouter"] = "ok" if resp.status_code == 200 else f"http {resp.status_code}"
    except Exception as e:
        services["openrouter"] = f"error: {str(e)}"

    return HealthResponse(status="ok", services=services)
