"""Desktop application launching, window management, typing, and media playback skills."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from typing import Any

from jarvis.core.safety import RiskLevel, is_app_allowed, resolve_safe_shortcut
from jarvis.skills.registry import SkillContext, SkillRegistry, SkillResult

try:
    import pygetwindow as gw  # type: ignore
except ImportError:
    gw = None

try:
    import pyautogui  # type: ignore
except ImportError:
    pyautogui = None


APP_CATALOG: dict[str, list[str]] = {
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"],
    "calc": ["calc.exe"],
    "paint": ["mspaint.exe"],
    "explorer": ["explorer.exe"],
    "file explorer": ["explorer.exe"],
    "files": ["explorer.exe"],
    "cmd": ["cmd.exe"],
    "command prompt": ["cmd.exe"],
    "powershell": ["powershell.exe"],
    "terminal": ["wt.exe", "powershell.exe"],
    "task manager": ["taskmgr.exe"],
    "settings": ["ms-settings:"],
    "chrome": ["chrome.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe"],
    "google chrome": ["chrome.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe"],
    "edge": ["msedge.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"],
    "microsoft edge": ["msedge.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"],
    "vs code": ["code.cmd", "code"],
    "vscode": ["code.cmd", "code"],
    "code": ["code.cmd", "code"],
    "visual studio code": ["code.cmd", "code"],
    "spotify": ["spotify.exe"],
    "discord": ["discord.exe"],
    "snipping tool": ["snippingtool.exe"],
}


def _send_media_vk(vk: int) -> None:
    if os.name != "nt":
        return
    user32 = ctypes.windll.user32
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, 0x0002, 0)


def register_app_skills(registry: SkillRegistry) -> None:
    @registry.register(
        name="open_application",
        description="Launch a desktop application by name (e.g., Notepad, Calculator, Chrome, Edge, VS Code, Terminal, Explorer, Settings, Spotify).",
        category="automation",
        risk=RiskLevel.MODERATE,
        parameters={
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Name of the application to open.",
                }
            },
            "required": ["app_name"],
        },
        examples=["open notepad", "launch calculator", "open vs code", "start chrome"],
    )
    def open_application(ctx: SkillContext, app_name: str) -> SkillResult:
        allowed, reason = is_app_allowed(app_name, strict=ctx.config.strict_allowlist)
        if not allowed:
            return SkillResult(ok=False, message=reason, data={"blocked": True, "app": app_name})

        key = app_name.strip().lower()
        candidates = APP_CATALOG.get(key, [app_name.strip()])

        def _win_show_startupinfo() -> Any:
            if os.name != "nt":
                return None
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 1  # SW_SHOWNORMAL
            return si

        for candidate in candidates:
            if candidate.endswith(":") and os.name == "nt":
                ctypes.windll.shell32.ShellExecuteW(None, "open", candidate, None, None, 1)
                return SkillResult(ok=True, message=f"Opening {app_name}.", data={"app": app_name, "target": candidate})
            resolved = shutil.which(candidate) or (candidate if os.path.exists(candidate) else None)
            if resolved:
                subprocess.Popen(
                    [resolved],
                    startupinfo=_win_show_startupinfo(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return SkillResult(ok=True, message=f"Launched {app_name}.", data={"app": app_name, "target": resolved})

        # Fallback: Windows `start` shell command for registered App Paths
        if os.name == "nt":
            try:
                subprocess.Popen(
                    ["cmd", "/c", "start", "", app_name.strip()],
                    startupinfo=_win_show_startupinfo(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return SkillResult(ok=True, message=f"Attempting to launch {app_name}.", data={"app": app_name})
            except Exception as exc:
                return SkillResult(ok=False, message=f"Could not open {app_name}: {exc}")

        return SkillResult(ok=False, message=f"Could not locate application '{app_name}'.")

    @registry.register(
        name="manage_window",
        description="List open desktop windows or focus, minimize, maximize, or close a window matching a title.",
        category="automation",
        risk=RiskLevel.MODERATE,
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "focus", "minimize", "maximize", "close"],
                },
                "title": {
                    "type": "string",
                    "description": "Partial window title to match (not required for 'list').",
                },
            },
            "required": ["action"],
        },
        examples=["list open windows", "focus chrome window", "minimize notepad"],
    )
    def manage_window(ctx: SkillContext, action: str, title: str = "") -> SkillResult:
        if gw is None:
            return SkillResult(ok=False, message="PyGetWindow is not installed.")
        act = action.lower().strip()
        titles = [t for t in gw.getAllTitles() if t and t.strip()]
        if act == "list":
            top = titles[:15]
            summary = ", ".join(top[:6]) + (f" (and {len(top) - 6} more)" if len(top) > 6 else "")
            return SkillResult(ok=True, message=f"Open windows: {summary or 'None detected.'}", data={"windows": top})

        if not title.strip():
            return SkillResult(ok=False, message="Please specify a window title.")

        needle = title.strip().lower()
        matches = [w for w in gw.getAllWindows() if needle in (w.title or "").lower()]
        if not matches:
            return SkillResult(ok=False, message=f"No open window matched '{title}'.")

        target = matches[0]
        if act == "focus":
            if target.isMinimized:
                target.restore()
            target.activate()
            return SkillResult(ok=True, message=f"Focused window '{target.title}'.", data={"window": target.title})
        if act == "minimize":
            target.minimize()
            return SkillResult(ok=True, message=f"Minimized '{target.title}'.", data={"window": target.title})
        if act == "maximize":
            target.maximize()
            return SkillResult(ok=True, message=f"Maximized '{target.title}'.", data={"window": target.title})
        if act == "close":
            target.close()
            return SkillResult(ok=True, message=f"Closed '{target.title}'.", data={"window": target.title})
        return SkillResult(ok=False, message=f"Unsupported window action '{action}'.")

    @registry.register(
        name="media_control",
        description="Control system media playback: play/pause, next track, previous track, or stop.",
        category="automation",
        risk=RiskLevel.SAFE,
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["play_pause", "next", "previous", "stop"],
                }
            },
            "required": ["command"],
        },
        examples=["pause music", "play next song", "previous track"],
    )
    def media_control(ctx: SkillContext, command: str) -> SkillResult:
        vk_map = {
            "play_pause": (0xB3, "Toggled media play/pause."),
            "next": (0xB0, "Skipped to next media track."),
            "previous": (0xB1, "Returned to previous media track."),
            "stop": (0xB2, "Stopped media playback."),
        }
        entry = vk_map.get(command.lower().strip())
        if not entry:
            return SkillResult(ok=False, message=f"Unknown media command '{command}'.")
        _send_media_vk(entry[0])
        return SkillResult(ok=True, message=entry[1], data={"command": command})

    @registry.register(
        name="type_text",
        description="Type dictated text into the currently active desktop window.",
        category="automation",
        risk=RiskLevel.MODERATE,
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to type."},
                "press_enter": {"type": "boolean", "description": "Whether to press Enter after typing."},
            },
            "required": ["text"],
        },
        examples=["type hello from jarvis", "dictate meeting starts at 4 pm"],
    )
    def type_text(ctx: SkillContext, text: str, press_enter: bool = False) -> SkillResult:
        if pyautogui is None:
            return SkillResult(ok=False, message="PyAutoGUI is not installed.")
        pyautogui.write(text, interval=0.015)
        if press_enter:
            pyautogui.press("enter")
        return SkillResult(ok=True, message=f"Typed {len(text)} characters into the active window.", data={"length": len(text)})

    @registry.register(
        name="press_shortcut",
        description="Execute an explicit allowlisted keyboard shortcut (e.g., Ctrl+C, Ctrl+V, Alt+Tab, Win+D, Ctrl+T, Ctrl+W, Ctrl+Z, Ctrl+S).",
        category="automation",
        risk=RiskLevel.MODERATE,
        parameters={
            "type": "object",
            "properties": {
                "shortcut": {
                    "type": "string",
                    "description": "Allowlisted shortcut name or combo (e.g., 'alt+tab', 'ctrl+c', 'show desktop', 'new tab').",
                }
            },
            "required": ["shortcut"],
        },
        examples=["press alt+tab", "switch window", "show desktop", "press ctrl+s", "new tab"],
    )
    def press_shortcut(ctx: SkillContext, shortcut: str) -> SkillResult:
        combo = resolve_safe_shortcut(shortcut)
        if not combo:
            return SkillResult(
                ok=False,
                message=f"Shortcut '{shortcut}' is not in the safe shortcut allowlist.",
                data={"blocked": True, "shortcut": shortcut},
            )
        if pyautogui is None:
            return SkillResult(ok=False, message="PyAutoGUI is not installed.")
        pyautogui.hotkey(*combo)
        label = "+".join(k.upper() for k in combo)
        return SkillResult(
            ok=True,
            message=f"Executed shortcut {label}.",
            data={"shortcut": label, "keys": list(combo)},
        )
