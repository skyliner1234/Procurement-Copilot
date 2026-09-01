#!/usr/bin/env python3
"""Single entry point.

    python run.py

Builds anything that is missing (database, risk model, knowledge-base index),
then serves the API and the dashboard.  Uses FastAPI + uvicorn when they are
installed and falls back to the standard-library server otherwise, so the demo
starts on a machine with nothing but Python.

    python run.py --rebuild     force a full rebuild first
    python run.py --port 8080   serve on a different port
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.app import config                                    # noqa: E402


def _needs_build() -> bool:
    return not (config.DB_PATH.exists()
                and config.METRICS_ARTIFACT.exists()
                and config.RAG_INDEX.exists())


def main() -> int:
    argv = sys.argv[1:]
    if "--port" in argv:
        config.PORT = int(argv[argv.index("--port") + 1])

    if "--rebuild" in argv or _needs_build():
        print("Building application artefacts (first run or --rebuild)...\n")
        sys.path.insert(0, str(ROOT / "scripts"))
        import build_all
        if build_all.main() != 0:
            print("\nBuild reported problems; starting anyway so you can inspect them.")

    print()
    try:
        import uvicorn                                            # noqa: F401
        from backend import api_fastapi                           # noqa: F401
        print(f"  Procurement Copilot (FastAPI)  ->  http://{config.HOST}:{config.PORT}")
        print(f"  API docs                       ->  http://{config.HOST}:{config.PORT}/docs")
        print("  Ctrl-C to stop.\n")
        uvicorn.run("backend.api_fastapi:app", host=config.HOST, port=config.PORT,
                    log_level="warning")
    except ImportError:
        print("  FastAPI/uvicorn not installed - using the standard-library server.")
        print("  (pip install -r requirements.txt to enable FastAPI and /docs)\n")
        from backend import serve_stdlib
        serve_stdlib.serve(config.HOST, config.PORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
