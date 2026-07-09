# Document QA System -- FastAPI + Frontend

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.config import Configuration


def main():
    parser = argparse.ArgumentParser(description="Document QA System - FastAPI Backend")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    config = Configuration.from_env()
    if not config.llm_api_key or "placeholder" in config.llm_api_key:
        print("???????? .env ?????????? LLM_API_KEY")
        sys.exit(1)

    import uvicorn
    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
