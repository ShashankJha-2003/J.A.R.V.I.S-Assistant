"""Rich Terminal Console, Headless Voice Loop, and Diagnostics (`--doctor`) for J.A.R.V.I.S."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from jarvis import __version__
from jarvis.core.engine import JarvisEngine
from jarvis.core.events import Event, EventType

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    Console = None  # type: ignore
    Panel = None  # type: ignore
    Table = None  # type: ignore


def _print_banner(engine: JarvisEngine) -> None:
    st = engine.get_status()
    if Console and Panel:
        console = Console()
        body = (
            f"[bold cyan]J.A.R.V.I.S. AI Voice Control Assistant v{__version__}[/bold cyan]\n"
            f"LLM: [green]{st['llm']['provider']}[/green] ({st['llm']['model']})  |  "
            f"STT: [yellow]{st['stt']['active']}[/yellow]  |  "
            f"TTS: [magenta]{st['tts']['active']}[/magenta]\n"
            f"Skills: [bold]{st['skills_count']}[/bold]  |  "
            f"Vector RAG: [bold]{st['memory']['total_vectors']}[/bold] entries\n"
            f"[dim]Commands: type naturally, or /listen (push-to-talk), /voice on|off, /status, /skills, /quit[/dim]"
        )
        console.print(Panel(body, border_style="cyan", title="SYSTEM ONLINE"))
    else:
        print(f"\n=== J.A.R.V.I.S. v{__version__} ===")
        print(f"LLM: {st['llm']['provider']} | STT: {st['stt']['active']} | TTS: {st['tts']['active']}")
        print("Type a command, or /listen, /voice on, /voice off, /status, /quit\n")


def print_diagnostics(engine: JarvisEngine) -> None:
    st = engine.get_status()
    if Console and Table:
        console = Console()
        table = Table(title=f"J.A.R.V.I.S. v{__version__} System Diagnostics", border_style="cyan")
        table.add_column("Subsystem", style="bold cyan")
        table.add_column("Active / Status", style="green")
        table.add_column("Details", style="white")
        table.add_row("AI Brain (LLM)", f"{st['llm']['provider']} ({st['llm']['model']})", f"Fast-Path: {st['config']['fast_path_routing']}")
        table.add_row("Speech-to-Text (STT)", st["stt"]["active"], f"Available: {', '.join(st['stt']['available']) or 'Browser Web Speech'}")
        table.add_row("Text-to-Speech (TTS)", st["tts"]["active"], f"Voice: {st['tts']['voice']} | Available: {', '.join(st['tts']['available'])}")
        mic = st["microphone"]
        table.add_row("Local Microphone", "READY" if mic["available"] else "BROWSER ONLY", f"sounddevice={mic['sounddevice_installed']}, inputs={mic['input_devices_found']}")
        table.add_row("Skill Registry", f"{st['enabled_skills_count']}/{st['skills_count']} enabled", "OpenAI + Gemini function schemas active")
        mem = st["memory"]
        table.add_row("Memory & RAG Store", f"{mem['total_vectors']} vectors", f"Facts={mem['facts']}, RAG={mem['rag_chunks']}, Notes={mem['notes']}, Tasks={mem['open_tasks']}")
        console.print(table)
    else:
        print("J.A.R.V.I.S. Diagnostics:", st)


async def run_cli_repl(engine: JarvisEngine, auto_voice: bool = False) -> None:
    """Interactive Terminal Console supporting typed + local microphone voice commands."""
    console = Console() if Console else None
    await engine.start(start_background_tasks=True)
    _print_banner(engine)

    async def _on_reminder(ev: Event) -> None:
        alert = ev.payload.get("alert", ev.payload.get("message", ""))
        if console:
            console.print(f"\n[bold yellow]⏰ {alert}[/bold yellow]")
        else:
            print(f"\n[REMINDER] {alert}")

    unsub = engine.event_bus.subscribe(EventType.REMINDER_TRIGGERED, _on_reminder)

    if auto_voice:
        res = await engine.start_local_voice_loop(require_wake_word=True)
        msg = res.get("message", "")
        if console:
            console.print(f"[cyan]{msg}[/cyan]")
        else:
            print(msg)

    try:
        while True:
            try:
                raw = await asyncio.to_thread(input, "\nYou > ")
            except (EOFError, KeyboardInterrupt):
                break
            cmd = raw.strip()
            if not cmd:
                continue
            low = cmd.lower()
            if low in {"/quit", "/exit", "exit", "quit"}:
                break
            if low == "/status":
                print_diagnostics(engine)
                continue
            if low == "/skills":
                for s in engine.registry.list_skills():
                    line = f"  • {s.name} [{s.category}] — {s.description}"
                    if console:
                        console.print(line)
                    else:
                        print(line)
                continue
            if low == "/listen":
                if console:
                    console.print("[bold cyan]🎙 Listening on local microphone...[/bold cyan]")
                res = await engine.listen_once(speak=True)
                if res.get("transcript"):
                    if console:
                        console.print(f"[dim]Heard: {res['transcript']}[/dim]")
                        console.print(f"[bold green]J.A.R.V.I.S. >[/bold green] {res.get('reply', '')}")
                    else:
                        print(f"Heard: {res['transcript']}\nJ.A.R.V.I.S. > {res.get('reply', '')}")
                else:
                    print(res.get("error", "No audio captured."))
                continue
            if low == "/voice on":
                res = await engine.start_local_voice_loop(require_wake_word=True)
                print(res["message"])
                continue
            if low == "/voice off":
                res = await engine.stop_local_voice_loop()
                print(res["message"])
                continue

            reply = await engine.handle_command(cmd, source="cli")
            tools = reply.get("tool_events") or []
            if tools and console:
                for t in tools:
                    console.print(f"  [dim cyan]⚙ {t['skill']}({t['arguments']})[/dim cyan]")
            if console:
                console.print(f"[bold cyan]J.A.R.V.I.S.[/bold cyan] [dim]({reply.get('provider')})[/dim] > {reply.get('reply', '')}")
            else:
                print(f"J.A.R.V.I.S. > {reply.get('reply', '')}")
    finally:
        unsub()
        await engine.stop()


async def run_voice_only(engine: JarvisEngine, require_wake_word: bool = True) -> None:
    """Headless continuous local microphone voice assistant."""
    await engine.start(start_background_tasks=True)
    _print_banner(engine)
    res = await engine.start_local_voice_loop(require_wake_word=require_wake_word)
    print(res["message"])
    if not res["ok"]:
        await engine.stop()
        sys.exit(1)

    try:
        while True:
            await asyncio.sleep(1.0)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await engine.stop()
