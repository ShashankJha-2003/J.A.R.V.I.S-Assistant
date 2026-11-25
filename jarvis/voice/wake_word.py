"""Wake-word detection: Optional Picovoice Porcupine (`pvporcupine`) + Fuzzy Transcript Detector."""

from __future__ import annotations

import difflib
import importlib.util
import logging
import re

logger = logging.getLogger(__name__)

try:
    import pvporcupine  # type: ignore
except Exception:
    pvporcupine = None


class WakeWordDetector:
    """Detects wake phrases ('Jarvis', 'Hey Jarvis', 'Hey Nova') via Porcupine or fuzzy STT."""

    def __init__(
        self,
        wake_words: list[str],
        fuzzy_cutoff: float = 0.8,
        picovoice_access_key: str = "",
    ) -> None:
        self.wake_words = [w.strip().lower() for w in wake_words if w.strip()]
        self.fuzzy_cutoff = fuzzy_cutoff
        self.picovoice_access_key = picovoice_access_key.strip()
        self._porcupine = None
        self._init_porcupine()

    def _init_porcupine(self) -> None:
        if pvporcupine is None or not self.picovoice_access_key:
            return
        try:
            supported = [k for k in ("jarvis", "porcupine", "computer", "alexa") if any(k in w for w in self.wake_words)]
            keywords = supported or ["jarvis"]
            self._porcupine = pvporcupine.create(
                access_key=self.picovoice_access_key,
                keywords=keywords,
            )
            logger.info("Picovoice Porcupine wake-word engine active for: %s", keywords)
        except Exception as exc:
            logger.debug("Porcupine init skipped (%s); using fuzzy transcript detector.", exc)
            self._porcupine = None

    @property
    def active_engine(self) -> str:
        return "pvporcupine" if self._porcupine is not None else "fuzzy_transcript"

    def available_engines(self) -> list[str]:
        engines = ["fuzzy_transcript"]
        if importlib.util.find_spec("pvporcupine") is not None:
            engines.insert(0, "pvporcupine")
        return engines

    def process_pcm_frame(self, pcm_int16: list[int]) -> bool:
        """Process a 512-sample 16kHz int16 frame via Porcupine if active."""
        if self._porcupine is None:
            return False
        try:
            return int(self._porcupine.process(pcm_int16)) >= 0
        except Exception:
            return False

    def close(self) -> None:
        if self._porcupine is not None:
            try:
                self._porcupine.delete()
            except Exception:
                pass
            self._porcupine = None

    def update_wake_words(self, wake_words: list[str]) -> None:
        self.wake_words = [w.strip().lower() for w in wake_words if w.strip()]

    def detect_and_extract(self, transcript: str) -> tuple[bool, str]:
        """Return `(detected, remaining_command)`."""
        raw = transcript.strip()
        if not raw:
            return False, ""
        low = raw.lower()

        for phrase in sorted(self.wake_words, key=len, reverse=True):
            pattern = re.compile(rf"\b{re.escape(phrase)}\b[,.:!?\s]*", re.IGNORECASE)
            match = pattern.search(raw)
            if match:
                remaining = raw[match.end() :].lstrip(" ,.:!?").strip()
                return True, remaining

        tokens = re.findall(r"[a-zA-Z0-9']+", low)
        single_targets = [w for w in self.wake_words if " " not in w]
        for tok in tokens[:3]:
            if difflib.get_close_matches(tok, single_targets, n=1, cutoff=self.fuzzy_cutoff):
                m = re.search(rf"\b{re.escape(tok)}\b[,.:!?\s]*", raw, re.IGNORECASE)
                if m:
                    return True, raw[m.end() :].lstrip(" ,.:!?").strip()

        return False, raw
