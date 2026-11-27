"""Centralized configuration management for J.A.R.V.I.S.

Loads settings from environment variables and `.env`, supports runtime updates
from the Web HUD or CLI, and creates required local directories automatically.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_list(key: str, default: list[str]) -> list[str]:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return list(default)
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


@dataclass
class JarvisConfig:
    """Runtime configuration for all J.A.R.V.I.S. subsystems."""

    # Identity & Persona
    assistant_name: str = "J.A.R.V.I.S."
    user_name: str = "Sir"
    persona: str = "calm"  # calm | crisp | witty | analytical
    wake_words: list[str] = field(
        default_factory=lambda: ["jarvis", "hey jarvis", "ok jarvis"]
    )

    # AI Brain / LLM
    llm_provider: str = "auto"  # auto | gemini | openai | groq | ollama | offline
    free_ai_fallback: bool = True
    fast_path_routing: bool = True
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    max_tool_steps: int = 5
    llm_temperature: float = 0.4

    # Orchestration
    orchestrator: str = "custom_asyncio"  # custom_asyncio | pipecat | livekit

    # Voice Engine (STT + VAD + Wake + TTS)
    stt_engine: str = "auto"  # auto | faster_whisper | vosk | deepgram | assemblyai | groq_whisper | openai_whisper | google
    whisper_model: str = "base"
    vosk_model_path: str = ""
    deepgram_api_key: str = ""
    assemblyai_api_key: str = ""
    picovoice_access_key: str = ""
    vad_engine: str = "auto"  # auto | silero | energy
    language: str = "en-IN"  # en-IN (Indian English/Hinglish) | hi-IN (Hindi Devanagari) | en-US

    tts_engine: str = "auto"  # auto | cloned | edge | elevenlabs | cartesia | piper | kokoro | pyttsx3 | none
    cloned_voice_enabled: bool = False
    cloned_voice_path: Path = field(default_factory=lambda: PROJECT_ROOT / "My Voice.mpeg")
    edge_tts_voice: str = "en-IN-PrabhatNeural"
    hindi_tts_voice: str = "hi-IN-MadhurNeural"
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "pNInz6obpgDQGcFmaJgB"
    cartesia_api_key: str = ""
    cartesia_voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091"
    piper_model_path: str = ""
    kokoro_voice: str = "bm_george"
    tts_rate: int = 185
    tts_volume: float = 1.0
    speak_responses: bool = True
    mic_energy_threshold: float = 0.015
    silence_duration: float = 0.9
    max_listen_seconds: float = 12.0

    # Web HUD Server
    host: str = "127.0.0.1"
    port: int = 8765
    auto_open_browser: bool = True

    # Safety, Proactive Triggers, Smart Home & Paths
    require_confirmation: bool = True
    strict_allowlist: bool = True
    proactive_alerts_enabled: bool = True
    battery_low_threshold: int = 15
    home_assistant_url: str = ""
    home_assistant_token: str = ""
    mqtt_broker_host: str = ""
    mqtt_broker_port: int = 1883
    default_city: str = "London"
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    knowledge_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "knowledge")
    captures_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "captures")
    log_level: str = "INFO"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.captures_dir.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jarvis.db"

    @property
    def runtime_overrides_path(self) -> Path:
        return self.data_dir / "runtime_settings.json"

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize config safely for HUD / API responses (redacts API keys)."""
        data = asdict(self)
        data["data_dir"] = str(self.data_dir)
        data["knowledge_dir"] = str(self.knowledge_dir)
        data["captures_dir"] = str(self.captures_dir)
        data["cloned_voice_path"] = str(self.cloned_voice_path)
        data["cloned_voice_ready"] = bool(self.cloned_voice_enabled and self.cloned_voice_path.exists())
        data["gemini_api_key_configured"] = bool(self.gemini_api_key)
        data["openai_api_key_configured"] = bool(self.openai_api_key)
        data["groq_api_key_configured"] = bool(self.groq_api_key)
        data["deepgram_api_key_configured"] = bool(self.deepgram_api_key)
        data["assemblyai_api_key_configured"] = bool(self.assemblyai_api_key)
        data["elevenlabs_api_key_configured"] = bool(self.elevenlabs_api_key)
        data["cartesia_api_key_configured"] = bool(self.cartesia_api_key)
        data["picovoice_configured"] = bool(self.picovoice_access_key)
        data["home_assistant_configured"] = bool(self.home_assistant_url and self.home_assistant_token)
        data["mqtt_configured"] = bool(self.mqtt_broker_host)
        for secret_key in (
            "gemini_api_key",
            "openai_api_key",
            "groq_api_key",
            "deepgram_api_key",
            "assemblyai_api_key",
            "elevenlabs_api_key",
            "cartesia_api_key",
            "picovoice_access_key",
            "home_assistant_token",
        ):
            data.pop(secret_key, None)
        return data

    def apply_runtime_updates(self, updates: dict[str, Any], persist: bool = True) -> dict[str, Any]:
        """Apply safe live setting updates from the Web HUD or CLI."""
        allowed = {
            "assistant_name",
            "user_name",
            "persona",
            "wake_words",
            "llm_provider",
            "fast_path_routing",
            "proactive_alerts_enabled",
            "battery_low_threshold",
            "gemini_model",
            "openai_model",
            "groq_model",
            "ollama_url",
            "ollama_model",
            "stt_engine",
            "language",
            "tts_engine",
            "cloned_voice_enabled",
            "edge_tts_voice",
            "hindi_tts_voice",
            "tts_rate",
            "tts_volume",
            "speak_responses",
            "require_confirmation",
            "default_city",
            "mic_energy_threshold",
            "silence_duration",
        }
        applied: dict[str, Any] = {}
        for key, value in updates.items():
            if key not in allowed:
                continue
            if key == "wake_words" and isinstance(value, str):
                value = [w.strip().lower() for w in value.split(",") if w.strip()]
            setattr(self, key, value)
            applied[key] = value

        if persist and applied:
            self.ensure_directories()
            existing: dict[str, Any] = {}
            if self.runtime_overrides_path.exists():
                try:
                    existing = json.loads(self.runtime_overrides_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    existing = {}
            existing.update(applied)
            self.runtime_overrides_path.write_text(
                json.dumps(existing, indent=2), encoding="utf-8"
            )
        return applied

    def save_llm_config(
        self,
        *,
        llm_provider: str | None = None,
        gemini_api_key: str | None = None,
        gemini_model: str | None = None,
        openai_api_key: str | None = None,
        openai_model: str | None = None,
        openai_base_url: str | None = None,
        groq_api_key: str | None = None,
        groq_model: str | None = None,
        ollama_url: str | None = None,
        ollama_model: str | None = None,
        env_file: Path | None = None,
    ) -> dict[str, Any]:
        """Update LLM provider/keys live in memory and persist safely into local `.env`."""
        env_updates: dict[str, str] = {}
        runtime_updates: dict[str, Any] = {}

        if llm_provider is not None and llm_provider.strip():
            val = llm_provider.strip().lower()
            if val in {"auto", "gemini", "openai", "groq", "ollama", "offline"}:
                self.llm_provider = val
                env_updates["JARVIS_LLM_PROVIDER"] = val
                runtime_updates["llm_provider"] = val

        key_map = [
            ("gemini_api_key", gemini_api_key, "GEMINI_API_KEY"),
            ("openai_api_key", openai_api_key, "OPENAI_API_KEY"),
            ("groq_api_key", groq_api_key, "GROQ_API_KEY"),
        ]
        for attr, val, env_name in key_map:
            if val is not None and val.strip():
                cleaned = val.strip()
                setattr(self, attr, cleaned)
                os.environ[env_name] = cleaned
                env_updates[env_name] = cleaned

        opt_map = [
            ("gemini_model", gemini_model, "JARVIS_GEMINI_MODEL"),
            ("openai_model", openai_model, "JARVIS_OPENAI_MODEL"),
            ("openai_base_url", openai_base_url, "JARVIS_OPENAI_BASE_URL"),
            ("groq_model", groq_model, "JARVIS_GROQ_MODEL"),
            ("ollama_url", ollama_url, "JARVIS_OLLAMA_URL"),
            ("ollama_model", ollama_model, "JARVIS_OLLAMA_MODEL"),
        ]
        for attr, val, env_name in opt_map:
            if val is not None and val.strip():
                cleaned = val.strip().rstrip("/") if "url" in attr else val.strip()
                setattr(self, attr, cleaned)
                os.environ[env_name] = cleaned
                env_updates[env_name] = cleaned
                if attr != "openai_base_url":
                    runtime_updates[attr] = cleaned

        if runtime_updates:
            self.apply_runtime_updates(runtime_updates, persist=True)

        target_env = env_file or (PROJECT_ROOT / ".env")
        if env_updates:
            lines: list[str] = []
            if target_env.exists():
                lines = target_env.read_text(encoding="utf-8").splitlines()
            seen: set[str] = set()
            new_lines: list[str] = []
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    k = stripped.split("=", 1)[0].strip()
                    if k in env_updates:
                        new_lines.append(f"{k}={env_updates[k]}")
                        seen.add(k)
                        continue
                new_lines.append(line)
            for k, v in env_updates.items():
                if k not in seen:
                    new_lines.append(f"{k}={v}")
            target_env.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")

        return {
            "llm_provider": self.llm_provider,
            "gemini_configured": bool(self.gemini_api_key),
            "openai_configured": bool(self.openai_api_key),
            "groq_configured": bool(self.groq_api_key),
            "gemini_model": self.gemini_model,
            "openai_model": self.openai_model,
            "groq_model": self.groq_model,
            "ollama_url": self.ollama_url,
            "ollama_model": self.ollama_model,
        }

    @classmethod
    def load(cls, env_file: Path | None = None, load_overrides: bool = True) -> "JarvisConfig":
        """Load configuration from `.env`, environment variables, and runtime overrides."""
        target_env = env_file or (PROJECT_ROOT / ".env")
        if target_env.exists():
            load_dotenv(target_env, override=False)

        def _resolve_path(env_key: str, default_rel: str) -> Path:
            raw = os.getenv(env_key, default_rel).strip()
            p = Path(raw)
            return p if p.is_absolute() else (PROJECT_ROOT / p)

        cfg = cls(
            assistant_name=os.getenv("JARVIS_ASSISTANT_NAME", "J.A.R.V.I.S."),
            user_name=os.getenv("JARVIS_USER_NAME", "Sir"),
            persona=os.getenv("JARVIS_PERSONA", "calm").strip().lower(),
            wake_words=_env_list("JARVIS_WAKE_WORDS", ["jarvis", "hey jarvis", "ok jarvis"]),
            llm_provider=os.getenv("JARVIS_LLM_PROVIDER", "auto").strip().lower(),
            free_ai_fallback=_env_bool("JARVIS_FREE_AI_FALLBACK", True),
            fast_path_routing=_env_bool("JARVIS_FAST_PATH_ROUTING", True),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("JARVIS_GEMINI_MODEL", "gemini-2.5-flash").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_model=os.getenv("JARVIS_OPENAI_MODEL", "gpt-4o-mini").strip(),
            openai_base_url=os.getenv("JARVIS_OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/"),
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            groq_model=os.getenv("JARVIS_GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
            ollama_url=os.getenv("JARVIS_OLLAMA_URL", "http://localhost:11434").strip().rstrip("/"),
            ollama_model=os.getenv("JARVIS_OLLAMA_MODEL", "llama3.2").strip(),
            orchestrator=os.getenv("JARVIS_ORCHESTRATOR", "custom_asyncio").strip().lower(),
            stt_engine=os.getenv("JARVIS_STT_ENGINE", "auto").strip().lower(),
            whisper_model=os.getenv("JARVIS_WHISPER_MODEL", "base").strip(),
            vosk_model_path=os.getenv("JARVIS_VOSK_MODEL_PATH", "").strip(),
            deepgram_api_key=os.getenv("DEEPGRAM_API_KEY", "").strip(),
            assemblyai_api_key=os.getenv("ASSEMBLYAI_API_KEY", "").strip(),
            picovoice_access_key=os.getenv("PICOVOICE_ACCESS_KEY", "").strip(),
            vad_engine=os.getenv("JARVIS_VAD_ENGINE", "auto").strip().lower(),
            language=os.getenv("JARVIS_LANGUAGE", "en-IN").strip(),
            tts_engine=os.getenv("JARVIS_TTS_ENGINE", "auto").strip().lower(),
            cloned_voice_enabled=_env_bool("JARVIS_CLONED_VOICE_ENABLED", False),
            cloned_voice_path=_resolve_path("JARVIS_CLONED_VOICE_PATH", "My Voice.mpeg"),
            edge_tts_voice=os.getenv("JARVIS_EDGE_TTS_VOICE", "en-IN-PrabhatNeural").strip(),
            hindi_tts_voice=os.getenv("JARVIS_HINDI_TTS_VOICE", "hi-IN-MadhurNeural").strip(),
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", "").strip(),
            elevenlabs_voice_id=os.getenv("JARVIS_ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB").strip(),
            cartesia_api_key=os.getenv("CARTESIA_API_KEY", "").strip(),
            cartesia_voice_id=os.getenv("JARVIS_CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091").strip(),
            piper_model_path=os.getenv("JARVIS_PIPER_MODEL_PATH", "").strip(),
            kokoro_voice=os.getenv("JARVIS_KOKORO_VOICE", "bm_george").strip(),
            tts_rate=_env_int("JARVIS_TTS_RATE", 185),
            tts_volume=_env_float("JARVIS_TTS_VOLUME", 1.0),
            speak_responses=_env_bool("JARVIS_SPEAK_RESPONSES", True),
            mic_energy_threshold=_env_float("JARVIS_MIC_ENERGY_THRESHOLD", 0.015),
            silence_duration=_env_float("JARVIS_SILENCE_DURATION", 0.9),
            max_listen_seconds=_env_float("JARVIS_MAX_LISTEN_SECONDS", 12.0),
            host=os.getenv("JARVIS_HOST", "127.0.0.1").strip(),
            port=_env_int("JARVIS_PORT", 8765),
            auto_open_browser=_env_bool("JARVIS_AUTO_OPEN_BROWSER", True),
            require_confirmation=_env_bool("JARVIS_REQUIRE_CONFIRMATION", True),
            strict_allowlist=_env_bool("JARVIS_STRICT_ALLOWLIST", True),
            proactive_alerts_enabled=_env_bool("JARVIS_PROACTIVE_ALERTS", True),
            battery_low_threshold=_env_int("JARVIS_BATTERY_LOW_THRESHOLD", 15),
            home_assistant_url=os.getenv("HOME_ASSISTANT_URL", "").strip().rstrip("/"),
            home_assistant_token=os.getenv("HOME_ASSISTANT_TOKEN", "").strip(),
            mqtt_broker_host=os.getenv("MQTT_BROKER_HOST", "").strip(),
            mqtt_broker_port=_env_int("MQTT_BROKER_PORT", 1883),
            default_city=os.getenv("JARVIS_DEFAULT_CITY", "London").strip(),
            data_dir=_resolve_path("JARVIS_DATA_DIR", "data"),
            knowledge_dir=_resolve_path("JARVIS_KNOWLEDGE_DIR", "knowledge"),
            captures_dir=_resolve_path("JARVIS_CAPTURES_DIR", "captures"),
            log_level=os.getenv("JARVIS_LOG_LEVEL", "INFO").strip().upper(),
        )
        cfg.ensure_directories()

        if load_overrides and cfg.runtime_overrides_path.exists():
            try:
                saved = json.loads(cfg.runtime_overrides_path.read_text(encoding="utf-8"))
                if isinstance(saved, dict):
                    cfg.apply_runtime_updates(saved, persist=False)
            except Exception:
                pass
        return cfg
