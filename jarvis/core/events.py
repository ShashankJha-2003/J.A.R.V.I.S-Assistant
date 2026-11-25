"""Typed asynchronous Pub/Sub Event Bus and Assistant State definitions."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class AssistantState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    EXECUTING = "executing"
    SPEAKING = "speaking"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    MUTED = "muted"
    ERROR = "error"


class EventType(str, Enum):
    STATE_CHANGED = "state_changed"
    WAKE_DETECTED = "wake_detected"
    AUDIO_LEVEL = "audio_level"
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    RESPONSE_FINAL = "response_final"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    EXPRESSION_CHANGED = "expression_changed"
    ACTION_CARD = "action_card"
    CONFIRMATION_REQUESTED = "confirmation_requested"
    REMINDER_TRIGGERED = "reminder_triggered"
    PROACTIVE_ALERT = "proactive_alert"
    TELEMETRY = "telemetry"
    SETTINGS_UPDATED = "settings_updated"
    SYSTEM_LOG = "system_log"
    ERROR = "error"


@dataclass(slots=True)
class Event:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = "core"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
        }


EventHandler = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """Lightweight, fault-isolated async publish/subscribe event bus."""

    def __init__(self) -> None:
        self._subscribers: dict[EventType | None, list[EventHandler]] = {}
        self._history: list[Event] = []
        self._max_history = 100

    def subscribe(
        self,
        event_type: EventType | None,
        handler: EventHandler,
    ) -> Callable[[], None]:
        """Subscribe to a specific EventType, or `None` for all events.

        Returns an unsubscribe callable.
        """
        bucket = self._subscribers.setdefault(event_type, [])
        bucket.append(handler)

        def unsubscribe() -> None:
            if handler in bucket:
                bucket.remove(handler)

        return unsubscribe

    async def publish(
        self,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
        source: str = "core",
    ) -> Event:
        """Publish an event to matching subscribers concurrently."""
        event = Event(type=event_type, payload=payload or {}, source=source)
        if event_type not in (EventType.AUDIO_LEVEL, EventType.TELEMETRY):
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]

        handlers = list(self._subscribers.get(event_type, ())) + list(
            self._subscribers.get(None, ())
        )
        if not handlers:
            return event

        async def _safe_invoke(fn: EventHandler) -> None:
            try:
                result = fn(event)
                if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                    await result
            except Exception as exc:
                logger.exception("EventBus subscriber failed for %s: %s", event_type.value, exc)

        await asyncio.gather(*(_safe_invoke(h) for h in handlers))
        return event

    def recent_events(self, limit: int = 30) -> list[dict[str, Any]]:
        return [ev.to_dict() for ev in self._history[-limit:]]
