"""Real-time Microphone Audio Capture with Energy VAD and RMS Level Metering.

Primary backend: `sounddevice` (PortAudio + NumPy 16kHz PCM, zero C-build pain on Windows).
Fallback backend: `SpeechRecognition.Microphone` (if PyAudio is installed).
"""

from __future__ import annotations

import io
import logging
import threading
import time
import wave
from dataclasses import dataclass
from typing import Callable

import numpy as np

from jarvis.voice.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd  # type: ignore
except Exception:
    sd = None

try:
    import speech_recognition as sr  # type: ignore
except Exception:
    sr = None


SAMPLE_RATE = 16000
CHANNELS = 1


@dataclass
class RecordedAudio:
    pcm16_bytes: bytes
    wav_bytes: bytes
    sample_rate: int = SAMPLE_RATE
    duration_seconds: float = 0.0
    peak_rms: float = 0.0


def pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit signed PCM
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class AudioRecorder:
    """Captures microphone utterances using adaptive energy VAD."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        energy_threshold: float = 0.015,
        silence_duration: float = 1.3,
        max_duration: float = 12.0,
        vad_engine: str = "auto",
    ) -> None:
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.max_duration = max_duration
        self.vad = VoiceActivityDetector(engine=vad_engine, energy_threshold=energy_threshold)

    def status(self) -> dict[str, object]:
        sd_devices = 0
        if sd is not None:
            try:
                devices = sd.query_devices()
                sd_devices = sum(1 for d in devices if d.get("max_input_channels", 0) > 0)
            except Exception:
                sd_devices = 0
        return {
            "sounddevice_installed": sd is not None,
            "input_devices_found": sd_devices,
            "speech_recognition_installed": sr is not None,
            "vad_engine": self.vad.active_engine,
            "available_vad": self.vad.available_engines(),
            "available": (sd is not None and sd_devices > 0),
        }

    def record_utterance(
        self,
        stop_event: threading.Event | None = None,
        on_level: Callable[[float], None] | None = None,
        start_timeout: float = 6.0,
    ) -> RecordedAudio | None:
        """Record a single spoken utterance and auto-stop after trailing silence."""
        if sd is not None:
            return self._record_sounddevice(stop_event, on_level, start_timeout)
        if sr is not None:
            return self._record_speech_recognition(start_timeout)
        raise RuntimeError(
            "No local Python microphone driver is available. Install `sounddevice` or use the Web HUD Browser Microphone."
        )

    def _record_sounddevice(
        self,
        stop_event: threading.Event | None,
        on_level: Callable[[float], None] | None,
        start_timeout: float,
    ) -> RecordedAudio | None:
        chunk_samples = int(self.sample_rate * 0.1)  # 100ms chunks
        frames: list[np.ndarray] = []
        pre_buffer: list[np.ndarray] = []
        speech_started = False
        started_at = time.monotonic()
        speech_start_time = 0.0
        last_voice_time = 0.0
        peak_rms = 0.0

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype="float32",
            blocksize=chunk_samples,
        ) as stream:
            while True:
                if stop_event and stop_event.is_set():
                    return None

                data, _overflow = stream.read(chunk_samples)
                mono = data[:, 0] if data.ndim > 1 else data
                self.vad.energy_threshold = self.energy_threshold
                is_voice, metric = self.vad.is_speech(mono, self.sample_rate)
                rms = float(np.sqrt(np.mean(np.square(mono))))
                peak_rms = max(peak_rms, rms)
                if on_level:
                    try:
                        on_level(min(1.0, rms * 10.0))
                    except Exception:
                        pass

                now = time.monotonic()
                if not speech_started:
                    pre_buffer.append(mono.copy())
                    if len(pre_buffer) > 4:
                        pre_buffer.pop(0)
                    if is_voice:
                        speech_started = True
                        speech_start_time = now
                        last_voice_time = now
                        frames.extend(pre_buffer)
                    elif (now - started_at) >= start_timeout:
                        return None
                else:
                    frames.append(mono.copy())
                    if is_voice or rms >= self.energy_threshold * 0.75:
                        last_voice_time = now
                    effective_silence = max(0.75, float(self.silence_duration))
                    if (now - last_voice_time) >= effective_silence:
                        break
                    if (now - speech_start_time) >= self.max_duration:
                        break

        if not frames:
            return None

        audio_f32 = np.concatenate(frames)
        audio_i16 = np.clip(audio_f32 * 32767.0, -32768, 32767).astype(np.int16)
        pcm_bytes = audio_i16.tobytes()
        wav_bytes = pcm16_to_wav(pcm_bytes, self.sample_rate, CHANNELS)
        duration = len(audio_i16) / float(self.sample_rate)
        return RecordedAudio(
            pcm16_bytes=pcm_bytes,
            wav_bytes=wav_bytes,
            sample_rate=self.sample_rate,
            duration_seconds=round(duration, 2),
            peak_rms=round(peak_rms, 4),
        )

    def _record_speech_recognition(self, start_timeout: float) -> RecordedAudio | None:
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = max(0.85, min(1.2, float(self.silence_duration)))
        recognizer.non_speaking_duration = 0.5
        try:
            with sr.Microphone(sample_rate=self.sample_rate) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.4)
                audio = recognizer.listen(
                    source,
                    timeout=start_timeout,
                    phrase_time_limit=self.max_duration,
                )
            wav_bytes = audio.get_wav_data(convert_rate=self.sample_rate, convert_width=2)
            pcm_bytes = audio.get_raw_data(convert_rate=self.sample_rate, convert_width=2)
            duration = len(pcm_bytes) / float(self.sample_rate * 2)
            return RecordedAudio(
                pcm16_bytes=pcm_bytes,
                wav_bytes=wav_bytes,
                sample_rate=self.sample_rate,
                duration_seconds=round(duration, 2),
                peak_rms=0.2,
            )
        except sr.WaitTimeoutError:
            return None
