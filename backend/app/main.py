"""FastAPI application entry point."""

from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.routes import router
from app.api.websocket import ws_router
from app.config import settings

app = FastAPI(
    title="LLM-Conductor",
    description="Agent-based Symbolic Music Orchestration System",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Check service health and Ollama connectivity."""
    ollama_ok = False
    openai_ok = bool(settings.openai_api_key)

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "ollama": ollama_ok,
        "openai_configured": openai_ok,
    }


# Mount static files for outputs (MIDI/MP3)
output_dir = Path(settings.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/outputs", StaticFiles(directory=str(output_dir)), name="outputs")

# Include routers
app.include_router(chat_router)  # New GPT-4o Conductor chat API
app.include_router(router)
app.include_router(ws_router)
