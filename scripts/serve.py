"""Production Uvicorn server runner for GeoVision REST API.

Usage:
    python scripts/serve.py --host 0.0.0.0 --port 8000
    python scripts/serve.py --reload
"""

from __future__ import annotations

import argparse

import uvicorn

from geovision.logger import get_logger

logger = get_logger("geovision.serve")


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoVision FastAPI Production Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port number to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", type=str, default="info", help="Logging level")
    args = parser.parse_args()

    logger.info(f"Starting GeoVision API Server on http://{args.host}:{args.port}")
    logger.info(f"Interactive Swagger Docs available at http://{args.host}:{args.port}/docs")
    logger.info(f"ReDoc Documentation available at http://{args.host}:{args.port}/redoc")

    uvicorn.run(
        "geovision.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
