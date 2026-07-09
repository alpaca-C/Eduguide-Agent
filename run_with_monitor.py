# run_with_monitor.py -- Zero-intrusion wrapper: enables token monitoring then starts the app.
#
# Usage (drop-in replacement for python src/main.py):
#   python run_with_monitor.py
#   python run_with_monitor.py --port 8080
#   python run_with_monitor.py --host 0.0.0.0 --reload
#
# Or with uvicorn directly:
#   python run_with_monitor.py --uvicorn src.api:app --port 7860

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Enable token monitoring BEFORE any LangChain imports
os.environ["MONITORING_ENABLED"] = "true"
import src.monitoring  # noqa: E402 -- activates the token tracker

# Now delegate to the original entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Research Agent with Token Monitoring")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--uvicorn", action="store_true",
                        help="Use uvicorn directly instead of src.main")
    args, unknown = parser.parse_known_args()

    if args.uvicorn:
        import uvicorn
        uvicorn.run(
            "src.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    else:
        from src.main import main
        # Override sys.argv for main's argparse
        sys.argv = [sys.argv[0]]
        if args.host != "127.0.0.1":
            sys.argv.extend(["--host", args.host])
        if args.port != 7860:
            sys.argv.extend(["--port", str(args.port)])
        if args.reload:
            sys.argv.append("--reload")
        main()
