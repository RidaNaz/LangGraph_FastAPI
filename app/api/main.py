from fastapi import APIRouter

from .routes import text
from .routes import text_audio
from .routes import text_files
from .routes import agentic_rag
from .routes import x_ray

api_router = APIRouter()
api_router.include_router(text.router, prefix="/text", tags=["text"])
api_router.include_router(text_audio.router, prefix="/text_audio", tags=["text_audio"])
api_router.include_router(text_files.router, prefix="/text_files", tags=["text_files"])
api_router.include_router(agentic_rag.router, prefix="/agentic_rag", tags=["agentic_rag"])
api_router.include_router(x_ray.router, prefix="/x_ray", tags=["x_ray"])