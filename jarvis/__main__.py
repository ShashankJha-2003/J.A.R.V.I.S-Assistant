"""Unified CLI entrypoint for `python -m jarvis` and `python main.py`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
import webbrowser

from jarvis.cli.terminal import print_diagnostics, run_cli_repl, run_voice_only
from jarvis.config import JarvisConfig
from jarvis.core.engine import JarvisEngine


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _open_browser_later(url: str, delay: float = 1.2) -> None:
    def _worker() -> None:
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="J.A.R.V.I.S. — Advanced Python AI Voice Control Assistant",
    )
    parser.add_argument("--cli", action="store_true", help="Launch interactive Rich Terminal HUD instead of web server.")
    parser.add_argument("--voice", action="store_true", help="Run headless hands-free voice control mode.")
    parser.add_argument("--no-wake", action="store_true", help="In --voice mode, process all speech without wake word.")
    parser.add_argument("--doctor", action="store_true", help="Run subsystem diagnostics and exit.")
    parser.add_argument("--command", "-c", type=str, default="", help="Execute a single command and print the response.")
    parser.add_argument("--host", type=str, default="", help="Web HUD host override (default from config).")
    parser.add_argument("--port", type=int, default=0, help="Web HUD port override (default from config).")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the Web HUD in the browser.")
    parser.add_argument("--silent", action="store_true", help="Disable spoken TTS responses.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = JarvisConfig.load()
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.silent:
        config.speak_responses = False

    _configure_logging(config.log_level)

    if args.doctor:
        async def _diag() -> None:
            engine = JarvisEngine(config=config, enable_tts=False)
            await engine.start(start_background_tasks=False)
            print_diagnostics(engine)
            await engine.stop()

        asyncio.run(_diag())
        return

    if args.command:
        async def _oneshot() -> None:
            engine = JarvisEngine(config=config, enable_tts=not args.silent)
            await engine.start(start_background_tasks=False)
            res = await engine.handle_command(args.command, source="oneshot", speak=not args.silent)
            print(res.get("reply", ""))
            await engine.stop()

        asyncio.run(_oneshot())
        return

    if args.cli:
        engine = JarvisEngine(config=config, enable_tts=not args.silent)
        asyncio.run(run_cli_repl(engine, auto_voice=args.voice))
        return

    if args.voice:
        engine = JarvisEngine(config=config, enable_tts=not args.silent)
        asyncio.run(run_voice_only(engine, require_wake_word=not args.no_wake))
        return

    # Default mode: Launch Futuristic FastAPI + WebSocket Holographic HUD Server
    import uvicorn

    from jarvis.web.server import create_app

    engine = JarvisEngine(config=config, enable_tts=not args.silent)
    app = create_app(engine=engine, start_background_tasks=True)
    url = f"http://{config.host}:{config.port}"
    print(f"\n  ⚡ J.A.R.V.I.S. Holographic HUD running at: {url}\n")
    if config.auto_open_browser and not args.no_browser:
        _open_browser_later(url)
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level.lower())


if __name__ == "__main__":
    main()
