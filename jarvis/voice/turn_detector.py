"""Semantic Turn Detection fused with VAD silence timing.

Distinguishes a natural mid-thought pause ("open...", "remind me to...", "um...")
from a genuinely finished user utterance so J.A.R.V.I.S. responds quickly without
cutting the user off mid-sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Trailing tokens that strongly signal the speaker paused to think and is NOT finished
_TRAILING_INCOMPLETE_WORDS = {
    "a", "an", "the", "and", "or", "but", "so", "because", "if", "when", "while",
    "although", "though", "unless", "until", "since", "that", "which", "who",
    "to", "for", "of", "in", "on", "at", "with", "about", "from", "into", "by",
    "um", "uh", "erm", "hmm", "like", "well",
    "open", "launch", "search", "remind", "remember", "explain", "define", "tell",
    "play", "set", "change", "adjust", "turn", "switch", "run", "delete", "kill",
}

_COMPLETE_SHORT_COMMANDS = {
    "stop", "cancel", "yes", "no", "confirm", "abort", "help", "mute", "unmute",
    "hi", "hello", "hey", "jarvis", "hey jarvis", "ok jarvis", "bye", "goodbye",
    "pause", "resume", "next", "previous", "smile", "laugh",
}


@dataclass
class TurnDecision:
    is_complete: bool
    reason: str
    recommended_wait_ms: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_complete": self.is_complete,
            "reason": self.reason,
            "recommended_wait_ms": self.recommended_wait_ms,
            "confidence": round(self.confidence, 3),
        }


class SemanticTurnDetector:
    """Combines acoustic VAD silence duration with lightweight semantic endpointing."""

    def __init__(
        self,
        fast_endpoint_ms: int = 480,
        normal_endpoint_ms: int = 900,
        thought_pause_hold_ms: int = 2100,
    ) -> None:
        self.fast_endpoint_ms = fast_endpoint_ms
        self.normal_endpoint_ms = normal_endpoint_ms
        self.thought_pause_hold_ms = thought_pause_hold_ms

    def evaluate(self, transcript: str, silence_ms: int, vad_active: bool = False) -> TurnDecision:
        """Decide whether the user has finished speaking or merely paused to think."""
        if vad_active:
            return TurnDecision(False, "speech_in_progress", self.normal_endpoint_ms, 0.98)

        raw = (transcript or "").strip()
        if not raw:
            return TurnDecision(False, "empty_transcript", self.normal_endpoint_ms, 0.99)

        low = raw.lower()
        clean = re.sub(r"[^\w\s']", "", low).strip()
        words = clean.split()

        # Explicit terminal punctuation (. ? !) signals high completion confidence
        if raw[-1] in ".?!" and len(words) >= 2:
            done = silence_ms >= self.fast_endpoint_ms
            return TurnDecision(done, "terminal_punctuation", self.fast_endpoint_ms, 0.94)

        # Known complete short commands / wake phrases finish immediately on brief silence
        if clean in _COMPLETE_SHORT_COMMANDS:
            done = silence_ms >= self.fast_endpoint_ms
            return TurnDecision(done, "complete_short_command", self.fast_endpoint_ms, 0.95)

        # Trailing ellipsis or comma signals unfinished thought
        if raw.endswith((",", "...", "-", ":")):
            done = silence_ms >= self.thought_pause_hold_ms
            return TurnDecision(done, "trailing_pause_punctuation", self.thought_pause_hold_ms, 0.88)

        # Trailing hesitation, preposition, conjunction, or verb awaiting object -> wait longer!
        last_word = words[-1] if words else ""
        if last_word in _TRAILING_INCOMPLETE_WORDS:
            done = silence_ms >= self.thought_pause_hold_ms
            return TurnDecision(done, f"mid_thought_pause({last_word})", self.thought_pause_hold_ms, 0.86)

        # Multi-word phrase ending cleanly in a content word -> standard crisp endpoint
        if len(words) >= 3:
            done = silence_ms >= self.normal_endpoint_ms
            return TurnDecision(done, "semantic_clause_complete", self.normal_endpoint_ms, 0.85)

        done = silence_ms >= int(self.normal_endpoint_ms * 1.15)
        return TurnDecision(done, "short_phrase_timeout", int(self.normal_endpoint_ms * 1.15), 0.72)
