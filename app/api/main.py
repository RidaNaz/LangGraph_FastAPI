from fastapi import APIRouter

from .routes import text
from .routes import files
from .routes import text_audio
from .routes import tool
from .routes import x_ray


api_router = APIRouter()
api_router.include_router(text.router, prefix="/text", tags=["text"])
api_router.include_router(text_audio.router, prefix="/text_audio", tags=["text_audio"])
api_router.include_router(files.router, prefix="/files", tags=["files"])
api_router.include_router(tool.router, prefix="/tool", tags=["tool"])
api_router.include_router(x_ray.router, prefix="/x_ray", tags=["x_ray"])