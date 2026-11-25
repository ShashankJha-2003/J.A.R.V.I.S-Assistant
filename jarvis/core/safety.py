"""Safety policy engine and confirmation manager for high-impact voice commands."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


SAFE_APP_ALLOWLIST: set[str] = {
    "notepad", "calculator", "calc", "paint", "explorer", "file explorer", "files",
    "terminal", "powershell", "cmd", "command prompt", "task manager", "settings",
    "chrome", "google chrome", "edge", "microsoft edge", "vs code", "vscode",
    "code", "visual studio code", "spotify", "discord", "snipping tool",
    "word", "excel", "powerpoint", "onenote", "outlook", "vlc", "obs",
}

SAFE_SHORTCUT_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "copy": ("ctrl", "c"),
    "ctrl+c": ("ctrl", "c"),
    "paste": ("ctrl", "v"),
    "ctrl+v": ("ctrl", "v"),
    "cut": ("ctrl", "x"),
    "ctrl+x": ("ctrl", "x"),
    "undo": ("ctrl", "z"),
    "ctrl+z": ("ctrl", "z"),
    "redo": ("ctrl", "y"),
    "ctrl+y": ("ctrl", "y"),
    "save": ("ctrl", "s"),
    "ctrl+s": ("ctrl", "s"),
    "select all": ("ctrl", "a"),
    "ctrl+a": ("ctrl", "a"),
    "switch window": ("alt", "tab"),
    "alt+tab": ("alt", "tab"),
    "show desktop": ("win", "d"),
    "win+d": ("win", "d"),
    "new tab": ("ctrl", "t"),
    "ctrl+t": ("ctrl", "t"),
    "close tab": ("ctrl", "w"),
    "ctrl+w": ("ctrl", "w"),
    "reopen tab": ("ctrl", "shift", "t"),
    "ctrl+shift+t": ("ctrl", "shift", "t"),
    "next tab": ("ctrl", "tab"),
    "ctrl+tab": ("ctrl", "tab"),
    "find": ("ctrl", "f"),
    "ctrl+f": ("ctrl", "f"),
    "screenshot": ("win", "shift", "s"),
}

_DANGEROUS_SHELL_PATTERN = re.compile(
    r"(?:[;&|`$><]|\brm\s+-rf\b|\bdel\s+/[fqsa]|\bformat\b|\bdiskpart\b|\breg\s+delete\b|\bshutdown\b|\bpowershell\s+-e|\bmkfs\b)",
    re.IGNORECASE,
)


def is_app_allowed(app_name: str, strict: bool = True) -> tuple[bool, str]:
    """Validate that an application launch target is safe and on the allowlist."""
    cleaned = app_name.strip().lower()
    if not cleaned:
        return False, "Empty application name."
    if _DANGEROUS_SHELL_PATTERN.search(cleaned):
        return False, f"Blocked potentially destructive shell pattern in '{app_name}'."
    if strict and cleaned not in SAFE_APP_ALLOWLIST:
        return (
            False,
            f"'{app_name}' is not in the safe application allowlist. Allowed apps include: "
            "VS Code, Chrome, Edge, Spotify, Notepad, Calculator, Explorer, Terminal, Settings.",
        )
    return True, cleaned


def resolve_safe_shortcut(shortcut: str) -> tuple[str, ...] | None:
    """Resolve a human shortcut phrase or key combo against the strict safe shortcut allowlist."""
    key = re.sub(r"\s+", " ", shortcut.strip().lower()).replace(" + ", "+")
    return SAFE_SHORTCUT_ALLOWLIST.get(key)


class RiskLevel(str, Enum):
    SAFE = "safe"           # Read-only queries, notes, timers, search
    MODERATE = "moderate"   # App launches, volume, screenshots, media keys
    DANGEROUS = "dangerous" # Shutdown, restart, sleep, destructive deletions


@dataclass
class PendingConfirmation:
    token: str
    skill_name: str
    arguments: dict[str, Any]
    description: str
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 45.0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "skill_name": self.skill_name,
            "arguments": self.arguments,
            "description": self.description,
            "expires_in": max(0.0, round(self.ttl_seconds - (time.time() - self.created_at), 1)),
        }


AFFIRMATIVE_PHRASES = {
    "yes", "yeah", "yep", "confirm", "confirmed", "proceed", "do it",
    "execute", "affirmative", "go ahead", "approve", "approved", "sure", "ok", "okay",
}
NEGATIVE_PHRASES = {
    "no", "nope", "cancel", "abort", "stop", "deny", "negative", "never mind",
    "nevermind", "don't", "do not", "reject",
}


class SafetyGuard:
    """Gates dangerous voice-triggered system actions behind explicit confirmation."""

    def __init__(self, require_confirmation: bool = True) -> None:
        self.require_confirmation = require_confirmation
        self._pending: PendingConfirmation | None = None

    def needs_confirmation(self, risk: RiskLevel) -> bool:
        return self.require_confirmation and risk == RiskLevel.DANGEROUS

    def request_confirmation(
        self,
        skill_name: str,
        arguments: dict[str, Any],
        description: str,
    ) -> PendingConfirmation:
        pending = PendingConfirmation(
            token=uuid.uuid4().hex[:10],
            skill_name=skill_name,
            arguments=arguments,
            description=description,
        )
        self._pending = pending
        return pending

    @property
    def pending(self) -> PendingConfirmation | None:
        if self._pending and self._pending.is_expired:
            self._pending = None
        return self._pending

    def clear_pending(self) -> None:
        self._pending = None

    def classify_reply(self, utterance: str) -> str | None:
        """Return 'approve', 'deny', or None if utterance isn't a confirmation reply."""
        cleaned = utterance.strip().lower().rstrip(".!?")
        if cleaned in AFFIRMATIVE_PHRASES:
            return "approve"
        if cleaned in NEGATIVE_PHRASES:
            return "deny"
        return None

    def pop_if_approved(self, token: str | None = None) -> PendingConfirmation | None:
        pending = self.pending
        if not pending:
            return None
        if token is not None and pending.token != token:
            return None
        self._pending = None
        return pending
