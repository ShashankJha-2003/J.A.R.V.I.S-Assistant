"""Voice Activity Detection (VAD) — Silero VAD (optional neural) + Hybrid Energy/ZCR fallback."""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch  # type: ignore
    from silero_vad import load_silero_vad  # type: ignore
except Exception:
    torch = None
    load_silero_vad = None


class VoiceActivityDetector:
    """Detects speech start/end using Silero VAD (when installed) or adaptive RMS + ZCR."""

    def __init__(self, engine: str = "auto", energy_threshold: float = 0.015) -> None:
        self.requested_engine = engine.lower().strip()
        self.energy_threshold = energy_threshold
        self._silero_model = None
        if self.requested_engine in ("auto", "silero") and load_silero_vad is not None:
            try:
                self._silero_model = load_silero_vad()
            except Exception as exc:
                logger.debug("Silero VAD unavailable (%s); using adaptive energy VAD.", exc)

    @property
    def active_engine(self) -> str:
        return "silero_vad" if self._silero_model is not None else "adaptive_energy_zcr"

    def available_engines(self) -> list[str]:
        avail = ["adaptive_energy_zcr"]
        if load_silero_vad is not None:
            avail.insert(0, "silero_vad")
        return avail

    def is_speech(self, mono_float32: np.ndarray, sample_rate: int = 16000) -> tuple[bool, float]:
        """Return `(is_speech, confidence_or_rms)` for a mono float32 PCM frame."""
        if mono_float32.size == 0:
            return False, 0.0

        rms = float(np.sqrt(np.mean(np.square(mono_float32))))
        if self._silero_model is not None and torch is not None and sample_rate in (8000, 16000):
            try:
                # Silero expects 512-sample chunks at 16kHz
                chunk_len = 512 if sample_rate == 16000 else 256
                probs: list[float] = []
                for start in range(0, len(mono_float32) - chunk_len + 1, chunk_len):
                    tensor = torch.from_numpy(mono_float32[start : start + chunk_len].astype(np.float32))
                    probs.append(float(self._silero_model(tensor, sample_rate).item()))
                if probs:
                    peak_prob = max(probs)
                    return peak_prob >= 0.45, round(peak_prob, 3)
            except Exception:
                pass

        # Hybrid energy + zero-crossing rate filter (rejects low-frequency hum/desk bumps)
        signs = np.signbit(mono_float32)
        zcr = float(np.mean(signs[1:] != signs[:-1])) if mono_float32.size > 1 else 0.0
        speech_like = rms >= self.energy_threshold and (0.01 <= zcr <= 0.48)
        return speech_like, round(rms, 4)
