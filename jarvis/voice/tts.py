"""Multi-Engine Text-to-Speech (TTS) with Neural Edge-TTS, Offline pyttsx3, and Barge-In Interrupt.

- Edge-TTS (`en-GB-RyanNeural`): Lifelike neural voice played natively via Windows WinMM MCI
  (no external ffmpeg/pygame required, instant stop/interrupt support).
- pyttsx3 (Windows SAPI5): 100% offline fallback on a dedicated COM-safe worker thread.
"""

from __future__ import annotations

import asyncio
import base64
import ctypes
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import httpx

from jarvis.ai.supervisor import detect_language_style
from jarvis.config import JarvisConfig
from jarvis.voice.voice_cloner import NeuralVoiceCloner, get_emotion_voice_profile

logger = logging.getLogger(__name__)

try:
    import edge_tts  # type: ignore
except ImportError:
    edge_tts = None

try:
    import pyttsx3  # type: ignore
except ImportError:
    pyttsx3 = None


def clean_text_for_speech(text: str) -> str:
    """Strip markdown formatting, code fences, and raw URLs for crisp spoken output."""
    cleaned = re.sub(r"```[\s\S]*?```", " code snippet omitted ", text)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+", "the link", cleaned)
    cleaned = re.sub(r"[*_#~>|]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# Ordered digraphs + single letters -> (viseme, relative_weight)
_VISEME_RULES: list[tuple[str, str, float]] = [
    ("oo", "OO", 1.8), ("ou", "OO", 1.7), ("ow", "OO", 1.7), ("qu", "OO", 1.2),
    ("ee", "EE", 1.8), ("ea", "EE", 1.6), ("ie", "EE", 1.6), ("ei", "EE", 1.5),
    ("ai", "AA", 1.7), ("ay", "AA", 1.6), ("ar", "AA", 1.7), ("au", "AA", 1.6), ("aw", "AA", 1.6),
    ("ph", "FV", 0.8), ("th", "MID", 0.75), ("sh", "OO", 0.85), ("ch", "EE", 0.8),
    ("wh", "OO", 1.0), ("ck", "MID", 0.6), ("ng", "MID", 0.65),
    ("a", "AA", 1.65), ("e", "EE", 1.35), ("i", "EE", 1.4), ("y", "EE", 1.2),
    ("o", "OO", 1.65), ("u", "OO", 1.45), ("w", "OO", 1.05),
    ("m", "MBP", 0.7), ("b", "MBP", 0.65), ("p", "MBP", 0.65),
    ("f", "FV", 0.8), ("v", "FV", 0.8),
    ("s", "EE", 0.75), ("z", "EE", 0.75), ("c", "EE", 0.65),
    ("r", "OO", 0.75), ("t", "MID", 0.6), ("d", "MID", 0.6),
    ("n", "MID", 0.65), ("l", "MID", 0.7), ("k", "MID", 0.6),
    ("g", "MID", 0.65), ("h", "MID", 0.55), ("j", "EE", 0.7), ("x", "EE", 0.7),
]


_DEVANAGARI_VISEMES: dict[str, tuple[str, float]] = {
    "अ": ("AA", 1.5), "आ": ("AA", 1.7), "ा": ("AA", 1.6),
    "इ": ("EE", 1.4), "ई": ("EE", 1.6), "ि": ("EE", 1.3), "ी": ("EE", 1.5),
    "ए": ("EE", 1.4), "ऐ": ("AA", 1.4), "े": ("EE", 1.3), "ै": ("AA", 1.4),
    "उ": ("OO", 1.5), "ऊ": ("OO", 1.7), "ु": ("OO", 1.4), "ू": ("OO", 1.6),
    "ओ": ("OO", 1.6), "औ": ("OO", 1.6), "ो": ("OO", 1.5), "ौ": ("OO", 1.5),
    "प": ("MBP", 0.7), "फ": ("FV", 0.8), "ब": ("MBP", 0.7), "भ": ("MBP", 0.7), "म": ("MBP", 0.75),
    "व": ("FV", 0.8),
}


def word_to_visemes(word: str) -> list[tuple[str, float]]:
    """Convert an English or Devanagari Hindi word into weighted articulatory visemes."""
    if re.search(r"[\u0900-\u097F]", word):
        seq_hi: list[tuple[str, float]] = []
        for ch in word:
            if "\u0900" <= ch <= "\u097F":
                vis, wt = _DEVANAGARI_VISEMES.get(ch, ("MID", 0.65))
                if seq_hi and seq_hi[-1][0] == vis:
                    seq_hi[-1] = (vis, min(2.2, seq_hi[-1][1] + wt * 0.35))
                else:
                    seq_hi.append((vis, wt))
        return seq_hi or [("MID", 1.0)]

    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return [("REST", 1.0)]
    # Silent trailing 'e' on longer words (e.g., "time", "make")
    if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiou":
        w = w[:-1]

    seq: list[tuple[str, float]] = []
    i = 0
    while i < len(w):
        matched = False
        for token, vis, weight in _VISEME_RULES:
            if w.startswith(token, i):
                if seq and seq[-1][0] == vis:
                    seq[-1] = (vis, min(2.4, seq[-1][1] + weight * 0.45))
                else:
                    seq.append((vis, weight))
                i += len(token)
                matched = True
                break
        if not matched:
            i += 1
    return seq or [("MID", 1.0)]


def build_viseme_timeline(word_boundaries: list[tuple[int, int, str]]) -> list[dict[str, object]]:
    """Build millisecond-accurate [{t, d, v}] viseme timeline from Edge-TTS WordBoundary tuples."""
    timeline: list[dict[str, object]] = []
    prev_end = 0
    for offset_ms, dur_ms, word in word_boundaries:
        if offset_ms - prev_end >= 45:
            timeline.append({"t": prev_end, "d": offset_ms - prev_end, "v": "REST"})
        visemes = word_to_visemes(word)
        total_w = sum(wt for _, wt in visemes) or 1.0
        cur_t = float(offset_ms)
        safe_dur = max(40, dur_ms)
        for vis, wt in visemes:
            seg_d = (wt / total_w) * safe_dur
            timeline.append({"t": round(cur_t), "d": max(25, round(seg_d)), "v": vis})
            cur_t += seg_d
        prev_end = offset_ms + safe_dur
    if prev_end > 0:
        timeline.append({"t": prev_end, "d": 200, "v": "REST"})
    return timeline


def _estimate_visemes(spoken: str) -> list[dict[str, object]]:
    """Fallback timed viseme estimation (~62ms/char) for engines without native word boundaries."""
    words: list[tuple[int, int, str]] = []
    cur = 60
    for word in spoken.split():
        dur = max(110, min(520, len(word) * 62))
        words.append((cur, dur, word))
        cur += dur + (160 if word.endswith((".", "!", "?")) else (90 if word.endswith(",") else 45))
    return build_viseme_timeline(words)


def _mci_send(command: str) -> tuple[int, str]:
    buf = ctypes.create_unicode_buffer(256)
    err = ctypes.windll.winmm.mciSendStringW(command, buf, 255, 0)
    return int(err), buf.value


class TTSEngine:
    """Interruptible Neural + Offline Speech Synthesizer."""

    def __init__(self, config: JarvisConfig) -> None:
        self.config = config
        self._stop_flag = threading.Event()
        self._speaking = False
        self._active_mci_alias: str | None = None
        self._pyttsx_engine = None
        self._lock = asyncio.Lock()
        self.cloner = NeuralVoiceCloner(
            reference_audio_path=config.cloned_voice_path,
            models_dir=config.data_dir / "models" / "openvoice_v2",
        )

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def available_engines(self) -> list[str]:
        engines: list[str] = []
        if self.config.cloned_voice_enabled and edge_tts is not None and self.cloner.is_available:
            engines.append("cloned")
        if self.config.elevenlabs_api_key:
            engines.append("elevenlabs")
        if self.config.cartesia_api_key:
            engines.append("cartesia")
        if edge_tts is not None:
            engines.append("edge")
        if self.config.piper_model_path and Path(self.config.piper_model_path).exists() and shutil.which("piper"):
            engines.append("piper")
        if pyttsx3 is not None:
            engines.append("pyttsx3")
        engines.append("none")
        return engines

    def active_engine_name(self) -> str:
        if not self.config.speak_responses:
            return "none"
        choice = self.config.tts_engine.lower().strip()
        avail = self.available_engines()
        if choice == "none":
            return "none"
        if choice != "auto" and choice in avail:
            return choice
        for pref in ("cloned", "elevenlabs", "cartesia", "edge", "piper", "pyttsx3"):
            if pref in avail:
                return pref
        return "none"

    async def warmup_cloner(self) -> None:
        """Warm up OpenVoice v2 ONNX sessions & cache neutral base speaker timbre in background."""
        if not (self.config.cloned_voice_enabled and self.cloner.is_available):
            return
        try:
            neutral_mp3: bytes | None = None
            if edge_tts is not None and not self.cloner.source_se_cache.exists():
                comm = edge_tts.Communicate(
                    "Hello there! Everything is working smoothly and clearly on my side today. Tell me what you would like to do next.",
                    voice="en-IN-PrabhatNeural",
                )
                buf = bytearray()
                async for ch in comm.stream():
                    if ch.get("type") == "audio" and ch.get("data"):
                        buf.extend(ch["data"])
                if buf:
                    neutral_mp3 = bytes(buf)
            await asyncio.to_thread(self.cloner.warmup, neutral_mp3)
        except Exception as exc:
            logger.debug("Voice cloner warmup skipped: %s", exc)

    def stop(self) -> None:
        """Immediately interrupt any ongoing speech (barge-in)."""
        self._stop_flag.set()
        if self._active_mci_alias and os.name == "nt":
            try:
                _mci_send(f"stop {self._active_mci_alias}")
                _mci_send(f"close {self._active_mci_alias}")
            except Exception:
                pass
            self._active_mci_alias = None
        if self._pyttsx_engine is not None:
            try:
                self._pyttsx_engine.stop()
            except Exception:
                pass

    async def _synth_edge_raw(
        self,
        spoken: str,
        *,
        voice: str,
        expression: str = "neutral",
    ) -> tuple[bytes, list[tuple[int, int, str]]]:
        """Generate expressive Edge-TTS MP3 bytes + WordBoundary timestamps for a given emotion."""
        prof = get_emotion_voice_profile(expression)
        communicate = edge_tts.Communicate(
            spoken,
            voice=voice,
            rate=prof.rate,
            pitch=prof.pitch,
            volume=prof.volume,
            boundary="WordBoundary",
        )
        chunks: list[bytes] = []
        boundaries: list[tuple[int, int, str]] = []
        async for chunk in communicate.stream():
            ctype = chunk.get("type")
            if ctype == "audio" and chunk.get("data"):
                chunks.append(chunk["data"])
            elif ctype == "WordBoundary":
                off_ms = int(chunk.get("offset", 0)) // 10000
                dur_ms = int(chunk.get("duration", 0)) // 10000
                word = str(chunk.get("text", ""))
                if word:
                    boundaries.append((off_ms, dur_ms, word))
        return b"".join(chunks), boundaries

    async def synthesize_speech_with_visemes(
        self,
        text: str,
        *,
        expression: str = "neutral",
    ) -> dict[str, object]:
        """Generate multi-emotional speech audio + millisecond viseme timeline.

        Default engine: Local OpenVoice v2 neural voice clone of `My Voice.mpeg`
        layered over 10-emotion expressive prosody (`en-IN-PrabhatNeural` base).
        """
        spoken = clean_text_for_speech(text)
        if not spoken:
            return {"audio_base64": None, "visemes": [], "mime": "audio/mpeg"}

        engine = self.active_engine_name()
        if engine == "elevenlabs" and self.config.elevenlabs_api_key:
            res = await self._synth_elevenlabs(spoken)
            if res.get("audio_base64"):
                return res
        if engine == "cartesia" and self.config.cartesia_api_key:
            res = await self._synth_cartesia(spoken)
            if res.get("audio_base64"):
                return res
        if engine == "piper":
            res = await asyncio.to_thread(self._synth_piper, spoken)
            if res.get("audio_base64"):
                return res

        if edge_tts is None:
            return {"audio_base64": None, "visemes": [], "mime": "audio/mpeg"}

        use_cloner = engine == "cloned" and self.config.cloned_voice_enabled and self.cloner.is_available
        lang_style = detect_language_style(spoken)
        is_hi = lang_style in ("hindi", "hinglish")
        hi_voice = self.config.hindi_tts_voice or "hi-IN-MadhurNeural"
        en_voice = "en-IN-PrabhatNeural" if use_cloner else (self.config.edge_tts_voice or "en-IN-PrabhatNeural")
        base_voice = hi_voice if is_hi else en_voice
        source_key = "hi" if is_hi else "en"
        try:
            mp3_bytes, boundaries = await self._synth_edge_raw(
                spoken,
                voice=base_voice,
                expression=expression,
            )
            if not mp3_bytes:
                return {"audio_base64": None, "visemes": [], "mime": "audio/mpeg"}

            if use_cloner:
                try:
                    mp3_bytes = await asyncio.to_thread(
                        self.cloner.convert_mp3_bytes,
                        mp3_bytes,
                        expression=expression,
                        source_key=source_key,
                    )
                except Exception as exc:
                    logger.warning("Neural voice clone conversion failed (%s); using expressive base voice.", exc)

            return {
                "audio_base64": base64.b64encode(mp3_bytes).decode("ascii"),
                "visemes": build_viseme_timeline(boundaries),
                "mime": "audio/mpeg",
                "voice": f"My Voice (Cloned/{source_key})" if use_cloner else base_voice,
                "base_voice": base_voice,
                "language": lang_style,
                "expression": expression,
            }
        except Exception as exc:
            logger.debug("TTS synthesis skipped: %s", exc)
            return {"audio_base64": None, "visemes": [], "mime": "audio/mpeg"}

    async def _synth_elevenlabs(self, spoken: str) -> dict[str, object]:
        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/"
            f"{self.config.elevenlabs_voice_id}/with-timestamps"
        )
        headers = {
            "xi-api-key": self.config.elevenlabs_api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": spoken,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {"stability": 0.45, "similarity_boost": 0.78},
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
            audio_b64 = data.get("audio_base64")
            align = data.get("alignment") or {}
            chars = align.get("characters") or []
            starts = align.get("character_start_times_seconds") or []
            ends = align.get("character_end_times_seconds") or []
            # Convert character-level timestamps into word boundaries for 6-viseme lip-sync
            words: list[tuple[int, int, str]] = []
            cur_word, w_start, w_end = "", None, 0.0
            for ch, st, en in zip(chars, starts, ends):
                if ch.isalpha() or ch == "'":
                    if w_start is None:
                        w_start = float(st)
                    cur_word += ch
                    w_end = float(en)
                elif cur_word and w_start is not None:
                    words.append((int(w_start * 1000), max(40, int((w_end - w_start) * 1000)), cur_word))
                    cur_word, w_start = "", None
            if cur_word and w_start is not None:
                words.append((int(w_start * 1000), max(40, int((w_end - w_start) * 1000)), cur_word))
            return {"audio_base64": audio_b64, "visemes": build_viseme_timeline(words), "mime": "audio/mpeg"}
        except Exception as exc:
            logger.warning("ElevenLabs TTS failed (%s); falling back.", exc)
            return {"audio_base64": None, "visemes": [], "mime": "audio/mpeg"}

    async def _synth_cartesia(self, spoken: str) -> dict[str, object]:
        headers = {
            "X-API-Key": self.config.cartesia_api_key,
            "Cartesia-Version": "2024-06-10",
            "Content-Type": "application/json",
        }
        payload = {
            "model_id": "sonic-english",
            "transcript": spoken,
            "voice": {"mode": "id", "id": self.config.cartesia_voice_id},
            "output_format": {"container": "mp3", "bit_rate": 128000, "sample_rate": 44100},
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.post("https://api.cartesia.ai/tts/bytes", headers=headers, json=payload)
                r.raise_for_status()
                return {
                    "audio_base64": base64.b64encode(r.content).decode("ascii"),
                    "visemes": _estimate_visemes(spoken),
                    "mime": "audio/mpeg",
                }
        except Exception as exc:
            logger.warning("Cartesia TTS failed (%s); falling back.", exc)
            return {"audio_base64": None, "visemes": [], "mime": "audio/mpeg"}

    def _synth_piper(self, spoken: str) -> dict[str, object]:
        piper_bin = shutil.which("piper")
        if not piper_bin or not self.config.piper_model_path:
            return {"audio_base64": None, "visemes": [], "mime": "audio/wav"}
        tmp_wav = Path(tempfile.gettempdir()) / f"jarvis_piper_{uuid.uuid4().hex[:8]}.wav"
        try:
            subprocess.run(
                [piper_bin, "--model", self.config.piper_model_path, "--output_file", str(tmp_wav)],
                input=spoken.encode("utf-8"),
                check=True,
                capture_output=True,
                timeout=15,
            )
            wav_bytes = tmp_wav.read_bytes()
            return {
                "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
                "visemes": _estimate_visemes(spoken),
                "mime": "audio/wav",
            }
        except Exception as exc:
            logger.warning("Piper TTS failed (%s); falling back.", exc)
            return {"audio_base64": None, "visemes": [], "mime": "audio/wav"}
        finally:
            tmp_wav.unlink(missing_ok=True)

    async def synthesize_mp3_base64(self, text: str) -> str | None:
        """Generate Edge-TTS neural MP3 bytes encoded as base64 (for Web HUD audio)."""
        result = await self.synthesize_speech_with_visemes(text)
        return result.get("audio_base64")  # type: ignore[return-value]

    async def speak(self, text: str, expression: str = "neutral") -> None:
        """Speak text aloud on local speakers using the active TTS engine."""
        engine = self.active_engine_name()
        if engine == "none":
            return
        spoken = clean_text_for_speech(text)
        if not spoken:
            return

        async with self._lock:
            self._stop_flag.clear()
            self._speaking = True
            try:
                if engine in ("cloned", "edge"):
                    ok = await self._speak_edge(spoken, expression=expression, clone=(engine == "cloned"))
                    if not ok and pyttsx3 is not None and not self._stop_flag.is_set():
                        await asyncio.to_thread(self._speak_pyttsx3, spoken)
                elif engine == "pyttsx3":
                    await asyncio.to_thread(self._speak_pyttsx3, spoken)
            finally:
                self._speaking = False

    async def _speak_edge(self, text: str, *, expression: str = "neutral", clone: bool = False) -> bool:
        if edge_tts is None:
            return False
        tmp_path = Path(tempfile.gettempdir()) / f"jarvis_tts_{uuid.uuid4().hex[:10]}.mp3"
        try:
            lang_style = detect_language_style(text)
            is_hi = lang_style in ("hindi", "hinglish")
            base_voice = (self.config.hindi_tts_voice or "hi-IN-MadhurNeural") if is_hi else (
                "en-IN-PrabhatNeural" if clone else (self.config.edge_tts_voice or "en-IN-PrabhatNeural")
            )
            source_key = "hi" if is_hi else "en"
            mp3_bytes, _ = await self._synth_edge_raw(text, voice=base_voice, expression=expression)
            if not mp3_bytes:
                return False
            if clone and self.cloner.is_available:
                try:
                    mp3_bytes = await asyncio.to_thread(
                        self.cloner.convert_mp3_bytes,
                        mp3_bytes,
                        expression=expression,
                        source_key=source_key,
                    )
                except Exception as exc:
                    logger.warning("Cloned playback fallback (%s).", exc)
            tmp_path.write_bytes(mp3_bytes)
            if self._stop_flag.is_set():
                return True
            if os.name == "nt":
                await asyncio.to_thread(self._play_mp3_winmm, tmp_path)
                return True
            return False
        except Exception as exc:
            logger.warning("Edge-TTS playback failed (%s); falling back to offline TTS.", exc)
            return False
        finally:
            tmp_path.unlink(missing_ok=True)

    def _play_mp3_winmm(self, mp3_path: Path) -> None:
        alias = f"jarvis_{uuid.uuid4().hex[:8]}"
        self._active_mci_alias = alias
        err, _ = _mci_send(f'open "{mp3_path}" type mpegvideo alias {alias}')
        if err != 0:
            self._active_mci_alias = None
            raise RuntimeError(f"WinMM MCI failed to open audio (code {err}).")
        try:
            _mci_send(f"play {alias}")
            while not self._stop_flag.is_set():
                code, mode = _mci_send(f"status {alias} mode")
                if code != 0 or mode.lower() != "playing":
                    break
                time.sleep(0.05)
        finally:
            _mci_send(f"stop {alias}")
            _mci_send(f"close {alias}")
            if self._active_mci_alias == alias:
                self._active_mci_alias = None

    def _speak_pyttsx3(self, text: str) -> None:
        if pyttsx3 is None or self._stop_flag.is_set():
            return
        com_initialized = False
        if os.name == "nt":
            try:
                import pythoncom  # type: ignore

                pythoncom.CoInitialize()
                com_initialized = True
            except Exception:
                pass
        try:
            engine = pyttsx3.init()
            self._pyttsx_engine = engine
            engine.setProperty("rate", int(self.config.tts_rate))
            engine.setProperty("volume", max(0.0, min(1.0, float(self.config.tts_volume))))
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            logger.warning("pyttsx3 playback error: %s", exc)
        finally:
            self._pyttsx_engine = None
            if com_initialized:
                try:
                    import pythoncom  # type: ignore

                    pythoncom.CoUninitialize()
                except Exception:
                    pass
