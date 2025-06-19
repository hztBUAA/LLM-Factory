"""
FastAPI application factory.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routes import router


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="LLM Factory API",
        description="Unified API for multiple LLM providers with load balancing",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.include_router(router, prefix="/api/v1")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "llm-factory"}
    
    @app.on_event("startup")
    async def startup_event():
        """Application startup event."""
        logger.info("LLM Factory API starting up...")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown event."""
        logger.info("LLM Factory API shutting down...")
    
    return app
