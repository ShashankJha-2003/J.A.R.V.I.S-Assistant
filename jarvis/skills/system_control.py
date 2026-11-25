"""System control & hardware telemetry skills for Windows/cross-platform desktops."""

from __future__ import annotations

import ctypes
import datetime as dt
import os
import platform
import shutil
import subprocess
import time
from typing import Any

from jarvis.core.safety import RiskLevel
from jarvis.skills.registry import SkillContext, SkillRegistry, SkillResult

# Optional dependencies resolved dynamically for graceful degradation
try:
    import psutil  # type: ignore
except ImportError:
    psutil = None

try:
    import pyperclip  # type: ignore
except ImportError:
    pyperclip = None


def collect_system_telemetry() -> dict[str, Any]:
    """Collect real-time CPU, RAM, Disk, Battery, and OS telemetry."""
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "hostname": platform.node(),
        "python": platform.python_version(),
        "timestamp": time.time(),
    }

    if psutil is not None:
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage(os.path.abspath(os.sep))
        cpu_pct = psutil.cpu_percent(interval=0.05)
        boot_ts = psutil.boot_time()
        uptime_sec = max(0, int(time.time() - boot_ts))
        info.update(
            {
                "cpu_percent": round(cpu_pct, 1),
                "cpu_cores": psutil.cpu_count(logical=True) or 1,
                "ram_percent": round(vm.percent, 1),
                "ram_used_gb": round(vm.used / (1024**3), 2),
                "ram_total_gb": round(vm.total / (1024**3), 2),
                "disk_percent": round(disk.percent, 1),
                "disk_free_gb": round(disk.free / (1024**3), 1),
                "disk_total_gb": round(disk.total / (1024**3), 1),
                "uptime_hours": round(uptime_sec / 3600, 1),
            }
        )
        batt = psutil.sensors_battery()
        if batt is not None:
            info["battery_percent"] = round(batt.percent, 1)
            info["power_plugged"] = bool(batt.power_plugged)
        else:
            info["battery_percent"] = None
            info["power_plugged"] = True
    else:
        # Lightweight native Windows / stdlib fallback when psutil is not installed
        disk = shutil.disk_usage(os.path.abspath(os.sep))
        disk_pct = round(((disk.total - disk.free) / max(1, disk.total)) * 100.0, 1)
        info.update(
            {
                "cpu_percent": 0.0,
                "cpu_cores": os.cpu_count() or 1,
                "disk_percent": disk_pct,
                "disk_free_gb": round(disk.free / (1024**3), 1),
                "disk_total_gb": round(disk.total / (1024**3), 1),
                "ram_percent": 0.0,
                "ram_used_gb": 0.0,
                "ram_total_gb": 0.0,
                "battery_percent": None,
                "power_plugged": True,
                "uptime_hours": 0.0,
            }
        )
        if os.name == "nt":
            try:
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                mem = MEMORYSTATUSEX()
                mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                    total_gb = mem.ullTotalPhys / (1024**3)
                    used_gb = (mem.ullTotalPhys - mem.ullAvailPhys) / (1024**3)
                    info["ram_percent"] = float(mem.dwMemoryLoad)
                    info["ram_used_gb"] = round(used_gb, 2)
                    info["ram_total_gb"] = round(total_gb, 2)

                class SYSTEM_POWER_STATUS(ctypes.Structure):
                    _fields_ = [
                        ("ACLineStatus", ctypes.c_byte),
                        ("BatteryFlag", ctypes.c_byte),
                        ("BatteryLifePercent", ctypes.c_byte),
                        ("SystemStatusFlag", ctypes.c_byte),
                        ("BatteryLifeTime", ctypes.c_ulong),
                        ("BatteryFullLifeTime", ctypes.c_ulong),
                    ]

                pwr = SYSTEM_POWER_STATUS()
                if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(pwr)):
                    if 0 <= pwr.BatteryLifePercent <= 100:
                        info["battery_percent"] = float(pwr.BatteryLifePercent)
                    info["power_plugged"] = pwr.ACLineStatus == 1
            except Exception:
                pass
    return info


def _send_vk(vk_code: int, repeat: int = 1) -> None:
    if os.name != "nt":
        return
    KEYEVENTF_KEYUP = 0x0002
    user32 = ctypes.windll.user32
    for _ in range(max(1, repeat)):
        user32.keybd_event(vk_code, 0, 0, 0)
        user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)


def _set_exact_volume_windows(level: int) -> bool:
    """Try pycaw CoreAudio first; fall back to VK_VOLUME stepping."""
    level = max(0, min(100, int(level)))
    try:
        from ctypes import POINTER, cast
        from comtypes import CLSCTX_ALL  # type: ignore
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMasterVolumeLevelScalar(level / 100.0, None)
        return True
    except Exception:
        pass

    if os.name == "nt":
        # Windows volume keys move in 2% increments: step to 0 then step up
        _send_vk(0xAE, repeat=50)  # VK_VOLUME_DOWN
        if level > 0:
            _send_vk(0xAF, repeat=round(level / 2))  # VK_VOLUME_UP
        return True
    return False


def register_system_skills(registry: SkillRegistry) -> None:
    @registry.register(
        name="get_system_status",
        description="Get current system telemetry including CPU usage, RAM usage, disk space, battery percentage, and uptime.",
        category="system",
        risk=RiskLevel.SAFE,
        examples=["what is my system status", "check cpu and battery", "how much ram am i using"],
    )
    def get_system_status(ctx: SkillContext) -> SkillResult:
        stats = collect_system_telemetry()
        batt_text = (
            f", battery at {stats['battery_percent']:.0f}% ({'charging' if stats.get('power_plugged') else 'on battery'})"
            if stats.get("battery_percent") is not None
            else ""
        )
        msg = (
            f"System telemetry nominal: CPU is at {stats['cpu_percent']}%, "
            f"memory at {stats['ram_percent']}% ({stats['ram_used_gb']} of {stats['ram_total_gb']} GB), "
            f"and primary storage has {stats['disk_free_gb']} GB free{batt_text}."
        )
        return SkillResult(ok=True, message=msg, data=stats)

    @registry.register(
        name="control_volume",
        description="Control system audio volume: mute, unmute, turn up, turn down, or set to a specific percentage (0-100).",
        category="system",
        risk=RiskLevel.MODERATE,
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["set", "up", "down", "mute", "unmute"],
                    "description": "The volume action to perform.",
                },
                "level": {
                    "type": "integer",
                    "description": "Target percentage (0-100) when action is 'set', or step count when up/down.",
                },
            },
            "required": ["action"],
        },
        examples=["set volume to 50 percent", "mute audio", "turn volume up"],
    )
    def control_volume(ctx: SkillContext, action: str, level: int | None = None) -> SkillResult:
        act = action.lower().strip()
        if act in ("mute", "unmute"):
            _send_vk(0xAD, repeat=1)  # VK_VOLUME_MUTE toggles mute
            return SkillResult(ok=True, message=f"Audio {'muted' if act == 'mute' else 'toggled unmute'}.", data={"action": act})
        if act == "up":
            steps = max(1, min(25, (level // 2) if level else 5))
            _send_vk(0xAF, repeat=steps)
            return SkillResult(ok=True, message=f"Volume increased by approximately {steps * 2} percent.", data={"action": act})
        if act == "down":
            steps = max(1, min(25, (level // 2) if level else 5))
            _send_vk(0xAE, repeat=steps)
            return SkillResult(ok=True, message=f"Volume decreased by approximately {steps * 2} percent.", data={"action": act})
        if act == "set":
            target = max(0, min(100, int(level if level is not None else 50)))
            ok = _set_exact_volume_windows(target)
            return SkillResult(
                ok=ok,
                message=f"System volume set to {target} percent." if ok else "Volume control is only supported on Windows.",
                data={"action": "set", "level": target},
            )
        return SkillResult(ok=False, message=f"Unsupported volume action '{action}'.")

    @registry.register(
        name="take_screenshot",
        description="Capture a screenshot of the current desktop screen and save it into the captures folder.",
        category="system",
        risk=RiskLevel.MODERATE,
        parameters={
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Optional filename without directory path (defaults to timestamp).",
                }
            },
        },
        examples=["take a screenshot", "capture my screen"],
    )
    def take_screenshot(ctx: SkillContext, filename: str = "") -> SkillResult:
        ctx.config.ensure_directories()
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = (filename.strip() or f"screenshot_{stamp}.png").replace("/", "_").replace("\\", "_")
        if not safe_name.lower().endswith(".png"):
            safe_name += ".png"
        out_path = ctx.config.captures_dir / safe_name

        try:
            from PIL import ImageGrab  # type: ignore

            img = ImageGrab.grab()
            img.save(out_path)
            return SkillResult(
                ok=True,
                message=f"Screenshot captured and saved as {out_path.name}.",
                data={"path": str(out_path), "width": img.width, "height": img.height},
            )
        except Exception as exc:
            return SkillResult(ok=False, message=f"Could not capture screenshot: {exc}")

    @registry.register(
        name="clipboard_action",
        description="Read the current text in the system clipboard or copy new text to the clipboard.",
        category="system",
        risk=RiskLevel.SAFE,
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["read", "copy"]},
                "text": {"type": "string", "description": "Text to copy when action is 'copy'."},
            },
            "required": ["action"],
        },
        examples=["read my clipboard", "copy hello world to clipboard"],
    )
    def clipboard_action(ctx: SkillContext, action: str, text: str = "") -> SkillResult:
        if pyperclip is None:
            return SkillResult(ok=False, message="pyperclip is not installed.")
        if action == "read":
            clip = pyperclip.paste() or ""
            if not clip.strip():
                return SkillResult(ok=True, message="Your clipboard is currently empty.", data={"text": ""})
            preview = clip[:240] + ("..." if len(clip) > 240 else "")
            return SkillResult(ok=True, message=f"Clipboard contains: {preview}", data={"text": clip})
        if action == "copy":
            pyperclip.copy(text)
            return SkillResult(ok=True, message="Text copied to your clipboard.", data={"length": len(text)})
        return SkillResult(ok=False, message=f"Unknown clipboard action: {action}")

    @registry.register(
        name="control_brightness",
        description="Adjust laptop display brightness on Windows (0-100%, up, down) via WMI.",
        category="system",
        risk=RiskLevel.MODERATE,
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["set", "up", "down"],
                    "description": "Set exact percentage, brighten (+15%), or dim (-15%).",
                },
                "level": {
                    "type": "integer",
                    "description": "Target brightness percentage (10-100) when action is 'set'.",
                },
            },
            "required": ["action"],
        },
        examples=["set brightness to 70 percent", "dim screen", "brighten screen"],
    )
    def control_brightness(ctx: SkillContext, action: str, level: int | None = None) -> SkillResult:
        act = action.lower().strip()
        if os.name == "nt":
            try:
                cur_proc = subprocess.run(
                    [
                        "powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness).CurrentBrightness",
                    ],
                    capture_output=True, text=True, timeout=5,
                )
                cur_val = int(cur_proc.stdout.strip().splitlines()[0]) if cur_proc.stdout.strip() else 60
                if act == "set":
                    target = max(10, min(100, int(level if level is not None else 70)))
                elif act == "up":
                    target = min(100, cur_val + (int(level) if level else 15))
                elif act == "down":
                    target = max(10, cur_val - (int(level) if level else 15))
                else:
                    return SkillResult(ok=False, message=f"Unsupported brightness action '{action}'.")

                subprocess.run(
                    [
                        "powershell", "-NoProfile", "-Command",
                        f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{target})",
                    ],
                    capture_output=True, text=True, timeout=5, check=True,
                )
                return SkillResult(
                    ok=True,
                    message=f"Display brightness set to {target} percent.",
                    data={"action": act, "level": target},
                )
            except Exception as exc:
                return SkillResult(ok=False, message=f"Brightness control is unavailable on this monitor: {exc}")
        return SkillResult(ok=False, message="Brightness control is configured for Windows WMI displays.")

    @registry.register(
        name="lock_screen",
        description="Lock the Windows workstation screen immediately.",
        category="system",
        risk=RiskLevel.MODERATE,
        examples=["lock my computer", "lock the screen"],
    )
    def lock_screen(ctx: SkillContext) -> SkillResult:
        if os.name == "nt":
            ctypes.windll.user32.LockWorkStation()
            return SkillResult(ok=True, message="Workstation locked.")
        return SkillResult(ok=False, message="Lock screen is configured for Windows.")

    @registry.register(
        name="power_control",
        description="Shutdown, restart, or put the computer to sleep (requires user confirmation).",
        category="system",
        risk=RiskLevel.DANGEROUS,
        parameters={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["shutdown", "restart", "sleep"],
                    "description": "Power state action to execute.",
                }
            },
            "required": ["mode"],
        },
        examples=["shutdown my pc", "restart computer", "put pc to sleep"],
    )
    def power_control(ctx: SkillContext, mode: str) -> SkillResult:
        m = mode.lower().strip()
        if os.name != "nt":
            return SkillResult(ok=False, message="Power control is configured for Windows.")
        if m == "shutdown":
            subprocess.Popen(["shutdown", "/s", "/t", "5"])
            return SkillResult(ok=True, message="Initiating system shutdown in 5 seconds.")
        if m == "restart":
            subprocess.Popen(["shutdown", "/r", "/t", "5"])
            return SkillResult(ok=True, message="Initiating system restart in 5 seconds.")
        if m == "sleep":
            subprocess.Popen(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
            return SkillResult(ok=True, message="Putting system into sleep mode.")
        return SkillResult(ok=False, message=f"Unknown power mode: {mode}")
