"""
DSPy Medical AI System — Main Entry Point

Self-optimizing medical AI with declarative LLM pipelines,
automatic prompt optimization, multi-agent reasoning,
and evaluation-driven refinement.

Usage:
    python app.py                        # Start production server
    python app.py --optimize             # Run optimization pipeline
    python app.py --evaluate             # Run evaluation suite
    python app.py --index                # Index documents
"""

from __future__ import annotations

import argparse
import sys
import uvicorn

from config.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DSPy Medical AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  (default)     Start the production API server
  --optimize    Run DSPy prompt optimization pipeline
  --evaluate    Run evaluation suite
  --index       Index documents into vector store
        """,
    )
    parser.add_argument("--optimize", action="store_true", help="Run optimization pipeline")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation suite")
    parser.add_argument("--index", action="store_true", help="Index documents into vector store")
    parser.add_argument("--host", type=str, default=None, help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    if args.optimize:
        from scripts.optimize import main as optimize_main
        optimize_main()
        return

    if args.evaluate:
        from scripts.evaluate import main as evaluate_main
        evaluate_main()
        return

    if args.index:
        from scripts.index_documents import main as index_main
        index_main()
        return

    # ── Start production server ──
    settings = get_settings()
    host = args.host or settings.server_host
    port = args.port or settings.server_port
    workers = args.workers or settings.server_workers

    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║          DSPy Medical AI System v2.0.0                  ║
    ║                                                        ║
    ║  Self-Optimizing · Multi-Agent · Evidence-Based         ║
    ║                                                        ║
    ║  Server: http://{host}:{port}                     ║
    ║  API:    http://{host}:{port}/api/v1/chat         ║
    ║  Health: http://{host}:{port}/api/v1/health       ║
    ║  Docs:   http://{host}:{port}/docs                ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "api.server:create_app",
        factory=True,
        host=host,
        port=port,
        workers=1 if args.reload else workers,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
