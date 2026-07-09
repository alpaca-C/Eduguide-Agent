"""BDD test runner -- all LLM calls mocked, zero API cost.

Usage:
    python test.py              # Run all tests
    python test.py --verbose    # Verbose output
    python test.py --tags=qa    # Run only QA-related tests
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

args = [sys.executable, "-m", "behave", str(ROOT / "tests" / "features")]

if "--verbose" in sys.argv or "-v" in sys.argv:
    args.append("--no-capture")
else:
    args.append("--format=progress")

for arg in sys.argv[1:]:
    if arg.startswith("--tags="):
        args.extend(["--tags", arg.split("=", 1)[1]])

sys.exit(subprocess.run(args, cwd=str(ROOT)).returncode)
