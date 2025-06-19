"""
Entry point for running the API server.
"""

import uvicorn
from .app import create_app

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )
