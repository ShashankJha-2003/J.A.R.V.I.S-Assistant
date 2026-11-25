"""Event-Driven Proactive Background Monitor (Battery, System Health, File Watcher, Upcoming Reminders).

Runs asynchronously inside JarvisEngine so J.A.R.V.I.S. can speak up proactively on
important events (low battery, newly dropped knowledge docs, upcoming blocks/reminders)
without waiting to be prompted.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import psutil

logger = logging.getLogger(__name__)

AlertCallback = Callable[[dict[str, Any]], Awaitable[None]]


class ProactiveMonitor:
    """Watches battery, system pressure, knowledge folder changes, and upcoming reminders."""

    def __init__(
        self,
        knowledge_dir: Path,
        on_alert: AlertCallback,
        on_knowledge_changed: Callable[[Path], Awaitable[dict[str, Any]]] | None = None,
        battery_low_percent: int = 15,
        ram_high_percent: int = 93,
        poll_interval_seconds: float = 15.0,
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.on_alert = on_alert
        self.on_knowledge_changed = on_knowledge_changed
        self.battery_low_percent = battery_low_percent
        self.ram_high_percent = ram_high_percent
        self.poll_interval_seconds = poll_interval_seconds

        self._file_mtimes: dict[str, float] = {}
        self._last_battery_alert_at: float = 0.0
        self._last_ram_alert_at: float = 0.0
        self._running = False
        self._snapshot_knowledge_files()

    def _snapshot_knowledge_files(self) -> dict[str, float]:
        snap: dict[str, float] = {}
        if not self.knowledge_dir.exists():
            return snap
        supported = {".md", ".txt", ".rst", ".json", ".pdf"}
        for p in self.knowledge_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in supported:
                try:
                    snap[str(p)] = p.stat().st_mtime
                except OSError:
                    pass
        self._file_mtimes = snap
        return snap

    async def check_once(self, now: float | None = None) -> list[dict[str, Any]]:
        """Run one full proactive check cycle (also directly unit-testable)."""
        ts = now if now is not None else time.time()
        fired: list[dict[str, Any]] = []

        # 1. Battery low trigger (cooldown: 10 minutes)
        try:
            batt = psutil.sensors_battery()
            if batt is not None and not batt.power_plugged and batt.percent <= self.battery_low_percent:
                if ts - self._last_battery_alert_at >= 600:
                    self._last_battery_alert_at = ts
                    alert = {
                        "kind": "battery_low",
                        "icon": "🔋",
                        "title": f"Battery at {int(round(batt.percent))}%",
                        "message": f"Heads up — your battery is at {int(round(batt.percent))}% and unplugged. You may want to plug in soon.",
                        "expression": "serious",
                        "speak": True,
                    }
                    fired.append(alert)
                    await self.on_alert(alert)
        except Exception:
            pass

        # 2. High RAM pressure alert (cooldown: 10 minutes)
        try:
            ram = psutil.virtual_memory().percent
            if ram >= self.ram_high_percent and (ts - self._last_ram_alert_at >= 600):
                self._last_ram_alert_at = ts
                alert = {
                    "kind": "memory_pressure",
                    "icon": "⚠",
                    "title": f"High RAM Usage ({ram:.0f}%)",
                    "message": f"System memory usage just hit {ram:.0f}%. You might want to close a few heavy tabs or apps.",
                    "expression": "serious",
                    "speak": True,
                }
                fired.append(alert)
                await self.on_alert(alert)
        except Exception:
            pass

        # 3. Knowledge directory file-change watcher (auto-indexes new/updated notes & PDFs!)
        if self.knowledge_dir.exists():
            prev = dict(self._file_mtimes)
            current = self._snapshot_knowledge_files()
            changed_paths = [
                Path(p) for p, mtime in current.items() if p not in prev or mtime > prev[p]
            ]
            for path in changed_paths:
                chunks = 0
                if self.on_knowledge_changed is not None:
                    try:
                        res = await self.on_knowledge_changed(path)
                        chunks = int(res.get("chunks_indexed", 0))
                    except Exception as exc:
                        logger.debug("Auto-index failed for %s: %s", path, exc)
                alert = {
                    "kind": "knowledge_updated",
                    "icon": "📚",
                    "title": f"Indexed {path.name}",
                    "message": f"I noticed '{path.name}' was updated in your knowledge folder and indexed {chunks} chunk(s) into memory.",
                    "expression": "smiling",
                    "speak": False,
                }
                fired.append(alert)
                await self.on_alert(alert)

        return fired

    async def run_loop(self) -> None:
        self._running = True
        while self._running:
            try:
                await self.check_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Proactive monitor tick error: %s", exc)
            await asyncio.sleep(self.poll_interval_seconds)

    def stop(self) -> None:
        self._running = False
