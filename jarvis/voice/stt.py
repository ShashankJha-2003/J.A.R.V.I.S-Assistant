"""Multi-Engine Speech-to-Text (STT):
- Offline Local: faster-whisper, vosk
- High-Accuracy Cloud: Deepgram (Nova-2), AssemblyAI, Groq Whisper, OpenAI Whisper, Google SR
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path

import httpx

from jarvis.config import JarvisConfig
from jarvis.voice.audio import RecordedAudio

logger = logging.getLogger(__name__)

try:
    import speech_recognition as sr  # type: ignore
except ImportError:
    sr = None

try:
    from faster_whisper import WhisperModel  # type: ignore
except ImportError:
    WhisperModel = None

try:
    import vosk  # type: ignore
except ImportError:
    vosk = None


class STTEngine:
    """Unified Speech-to-Text engine supporting local offline & cloud neural providers."""

    def __init__(self, config: JarvisConfig) -> None:
        self.config = config
        self._whisper_model = None
        self._vosk_model = None

    def available_engines(self) -> list[str]:
        engines: list[str] = []
        if self.config.deepgram_api_key:
            engines.append("deepgram")
        if self.config.assemblyai_api_key:
            engines.append("assemblyai")
        if WhisperModel is not None:
            engines.append("faster_whisper")
        if vosk is not None and self.config.vosk_model_path and Path(self.config.vosk_model_path).exists():
            engines.append("vosk")
        if self.config.groq_api_key:
            engines.append("groq_whisper")
        if self.config.openai_api_key:
            engines.append("openai_whisper")
        if sr is not None:
            engines.append("google")
        return engines

    def active_engine_name(self) -> str:
        choice = self.config.stt_engine.lower().strip()
        avail = self.available_engines()
        if choice != "auto" and choice in avail:
            return choice
        for pref in ("deepgram", "faster_whisper", "groq_whisper", "openai_whisper", "assemblyai", "vosk", "google"):
            if pref in avail:
                return pref
        return "browser_web_speech"

    async def transcribe(self, audio: RecordedAudio, *, language: str | None = None) -> str:
        engine = self.active_engine_name()
        lang = (language or self.config.language or "en-IN").strip()
        transcript = ""
        if engine == "deepgram":
            transcript = await self._transcribe_deepgram(audio, language=lang)
        elif engine == "assemblyai":
            transcript = await self._transcribe_assemblyai(audio)
        elif engine == "faster_whisper":
            transcript = await asyncio.to_thread(self._transcribe_faster_whisper, audio, lang)
        elif engine == "vosk":
            transcript = await asyncio.to_thread(self._transcribe_vosk, audio)
        elif engine == "groq_whisper":
            transcript = await self._transcribe_whisper_api(
                audio,
                url="https://api.groq.com/openai/v1/audio/transcriptions",
                api_key=self.config.groq_api_key,
                model="whisper-large-v3-turbo",
                language=lang,
            )
        elif engine == "openai_whisper":
            transcript = await self._transcribe_whisper_api(
                audio,
                url=f"{self.config.openai_base_url}/audio/transcriptions",
                api_key=self.config.openai_api_key,
                model="whisper-1",
                language=lang,
            )
        elif engine == "google":
            transcript = await asyncio.to_thread(self._transcribe_google_sr, audio, lang)
        if transcript:
            logger.info("RAW_STT (%s/%s): %s", engine, lang, transcript)
        return transcript

    async def _transcribe_deepgram(self, audio: RecordedAudio, *, language: str = "en-IN") -> str:
        headers = {
            "Authorization": f"Token {self.config.deepgram_api_key}",
            "Content-Type": "audio/wav",
        }
        params: dict[str, str] = {
            "model": "nova-2",
            "smart_format": "true",
        }
        if language.lower() in ("auto", "multi"):
            params["detect_language"] = "true"
        else:
            params["language"] = language.split("-")[0]
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.deepgram.com/v1/listen",
                headers=headers,
                params=params,
                content=audio.wav_bytes,
            )
            resp.raise_for_status()
            data = resp.json()
        alts = (
            data.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])
        )
        return str(alts[0].get("transcript", "")).strip() if alts else ""

    async def _transcribe_assemblyai(self, audio: RecordedAudio) -> str:
        headers = {"authorization": self.config.assemblyai_api_key}
        async with httpx.AsyncClient(timeout=30.0) as client:
            up = await client.post(
                "https://api.assemblyai.com/v2/upload",
                headers=headers,
                content=audio.wav_bytes,
            )
            up.raise_for_status()
            upload_url = up.json()["upload_url"]

            tr = await client.post(
                "https://api.assemblyai.com/v2/transcript",
                headers=headers,
                json={"audio_url": upload_url, "speech_model": "best"},
            )
            tr.raise_for_status()
            tid = tr.json()["id"]

            for _ in range(25):
                poll = await client.get(f"https://api.assemblyai.com/v2/transcript/{tid}", headers=headers)
                poll.raise_for_status()
                body = poll.json()
                if body.get("status") == "completed":
                    return str(body.get("text", "")).strip()
                if body.get("status") == "error":
                    logger.warning("AssemblyAI error: %s", body.get("error"))
                    return ""
                await asyncio.sleep(0.4)
        return ""

    def _transcribe_faster_whisper(self, audio: RecordedAudio, language: str = "en-IN") -> str:
        if WhisperModel is None:
            return ""
        if self._whisper_model is None:
            logger.info("Loading faster-whisper model '%s'...", self.config.whisper_model)
            self._whisper_model = WhisperModel(
                self.config.whisper_model,
                device="auto",
                compute_type="int8",
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio.wav_bytes)
            tmp_path = Path(tmp.name)
        try:
            # Allow auto language detection for multilingual Hindi/Hinglish/English speech
            lang_arg = None if language.lower() in ("auto", "en-in", "hi-in") else language.split("-")[0]
            segments, _info = self._whisper_model.transcribe(str(tmp_path), beam_size=3, language=lang_arg)
            return " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            tmp_path.unlink(missing_ok=True)

    def _transcribe_vosk(self, audio: RecordedAudio) -> str:
        if vosk is None or not self.config.vosk_model_path:
            return ""
        if self._vosk_model is None:
            self._vosk_model = vosk.Model(self.config.vosk_model_path)
        rec = vosk.KaldiRecognizer(self._vosk_model, audio.sample_rate)
        rec.AcceptWaveform(audio.pcm16_bytes)
        res = json.loads(rec.FinalResult() or "{}")
        return str(res.get("text", "")).strip()

    async def _transcribe_whisper_api(
        self,
        audio: RecordedAudio,
        url: str,
        api_key: str,
        model: str,
        language: str = "en-IN",
    ) -> str:
        files = {"file": ("speech.wav", audio.wav_bytes, "audio/wav")}
        data: dict[str, str] = {"model": model}
        if language.lower() not in ("auto", "en-in", "hi-in"):
            data["language"] = language.split("-")[0]
        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(url, headers=headers, files=files, data=data)
            resp.raise_for_status()
            payload = resp.json()
        return str(payload.get("text", "")).strip()

    def _transcribe_google_sr(self, audio: RecordedAudio, language: str = "en-IN") -> str:
        if sr is None:
            return ""
        recognizer = sr.Recognizer()
        sr_audio = sr.AudioData(audio.pcm16_bytes, audio.sample_rate, 2)
        # Try configured Indian-English/Hindi locale first, then fallback to companion Hindi/Indian-English locale
        langs = [language]
        if language.lower() == "en-in":
            langs.append("hi-IN")
        elif language.lower() == "hi-in":
            langs.append("en-IN")
        for loc in langs:
            try:
                text = str(recognizer.recognize_google(sr_audio, language=loc)).strip()
                if text:
                    return text
            except sr.UnknownValueError:
                continue
            except Exception as exc:
                logger.warning("Google STT (%s) failed: %s", loc, exc)
                break
        return ""
