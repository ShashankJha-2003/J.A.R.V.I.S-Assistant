"""Main J.A.R.V.I.S. Runtime Engine orchestrating voice, AI brain, skills, reminders, and telemetry."""

from __future__ import annotations

import asyncio
import logging
import random
import re
import threading
import time
from typing import Any

from jarvis.ai.brain import JarvisBrain
from jarvis.ai.intent_router import strip_wake_prefix
from jarvis.ai.providers import BaseLLMProvider
from jarvis.ai.supervisor import PERSONA_PROFILES, get_persona
from jarvis.config import JarvisConfig
from jarvis.core.events import AssistantState, EventBus, EventType
from jarvis.core.mcp_client import MCPManager
from jarvis.core.proactive import ProactiveMonitor
from jarvis.core.realtime_pipeline import inspect_orchestrators
from jarvis.core.safety import SafetyGuard
from jarvis.memory.database import JarvisDatabase
from jarvis.memory.vector_store import SemanticVectorStore, vector_backend_info
from jarvis.skills.loader import create_default_registry
from jarvis.skills.registry import SkillContext, SkillRegistry
from jarvis.skills.system_control import collect_system_telemetry
from jarvis.voice.audio import AudioRecorder, RecordedAudio, pcm16_to_wav
from jarvis.voice.stt import STTEngine
from jarvis.voice.tts import TTSEngine
from jarvis.voice.turn_detector import SemanticTurnDetector
from jarvis.voice.wake_word import WakeWordDetector

logger = logging.getLogger(__name__)


_VOICE_SOURCES = {"ptt_voice", "hands_free_voice", "local_mic", "avatar_voice"}
_NOISE_FILLERS = {"uh", "um", "erm", "hm", "hmm", "ah", "oh", "eh", "mm", "mhm"}


def _is_stt_noise(text: str, source: str) -> bool:
    if source not in _VOICE_SOURCES:
        return False
    norm = re.sub(r"[^\w\s]", "", (text or "").lower()).strip()
    if not norm:
        return True
    if len(norm) < 2 and not norm.isdigit():
        return True
    if norm in _NOISE_FILLERS:
        return True
    return False


ENTRANCE_GREETINGS = [
    "Hey there! What's on your mind?",
    "Hey there! Good to hear from you — how can I help?",
    "Hey! I'm right here, what can I do for you?",
    "Hey there! Ready whenever you are.",
    "Hi! I'm listening — what are we working on today?",
]


def build_action_card(tool_events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Build a structured Liquid-Glass Bento confirmation card payload from executed tools."""
    if not tool_events:
        return None
    last = tool_events[-1]
    skill = str(last.get("skill", ""))
    args = last.get("arguments") or {}
    res = last.get("result") or {}
    data = res.get("data") or {}
    ok = bool(res.get("ok", True))
    lat = res.get("latency_ms", 0)

    if skill == "open_application":
        app = str(args.get("app_name") or data.get("app") or "App").title()
        return {
            "kind": "app",
            "icon": "🚀" if ok else "🛡",
            "title": f"Opened {app}" if ok else f"Blocked {app}",
            "subtitle": "Allowlist Verified • Laptop Control" if ok else "Safe Allowlist Policy",
            "status": "ok" if ok else "blocked",
            "metrics": [{"label": "LATENCY", "value": f"{lat:.0f}ms"}],
        }
    if skill == "get_system_status":
        metrics = [
            {"label": "CPU", "value": f"{data.get('cpu_percent', 0)}%"},
            {"label": "RAM", "value": f"{data.get('ram_percent', 0)}%"},
            {"label": "DISK FREE", "value": f"{data.get('disk_free_gb', 0)}G"},
        ]
        if data.get("battery_percent") is not None:
            metrics.append({"label": "BATTERY", "value": f"{data.get('battery_percent', 0):.0f}%"})
        return {
            "kind": "system",
            "icon": "⚡",
            "title": "System Telemetry",
            "subtitle": str(data.get("hostname") or data.get("os") or "Workstation"),
            "status": "ok",
            "metrics": metrics,
        }
    if skill == "get_weather" and ok:
        return {
            "kind": "weather",
            "icon": "🌤",
            "title": str(data.get("location") or args.get("city") or "Weather"),
            "subtitle": str(data.get("condition") or "").title(),
            "status": "ok",
            "metrics": [
                {"label": "TEMP", "value": f"{data.get('temperature_c', '--')}°C"},
                {"label": "HUMIDITY", "value": f"{data.get('humidity', '--')}%"},
                {"label": "WIND", "value": f"{data.get('wind_kmh', '--')} km/h"},
            ],
        }
    if skill in ("control_volume", "control_brightness"):
        label = "Audio Volume" if skill == "control_volume" else "Display Brightness"
        lvl = data.get("level")
        return {
            "kind": "control",
            "icon": "🔊" if skill == "control_volume" else "☀",
            "title": label,
            "subtitle": str(res.get("message", "")),
            "status": "ok" if ok else "error",
            "progress": int(lvl) if isinstance(lvl, (int, float)) else None,
            "metrics": [{"label": "ACTION", "value": str(args.get("action", "")).upper()}],
        }
    if skill == "press_shortcut":
        return {
            "kind": "shortcut",
            "icon": "⌨",
            "title": f"Shortcut {data.get('shortcut', args.get('shortcut', ''))}",
            "subtitle": "Safe Allowlist Executed" if ok else "Blocked Shortcut",
            "status": "ok" if ok else "blocked",
            "metrics": [{"label": "LATENCY", "value": f"{lat:.0f}ms"}],
        }
    if skill == "search_wikipedia" and ok:
        return {
            "kind": "knowledge",
            "icon": "📚",
            "title": str(data.get("topic") or args.get("topic") or "Knowledge"),
            "subtitle": str(data.get("source") or "Wikipedia"),
            "status": "ok",
            "metrics": [{"label": "SOURCE", "value": str(data.get("source") or "Wiki").upper()}],
        }
    if skill in ("analyze_screen", "analyze_camera"):
        return {
            "kind": "vision",
            "icon": "👁" if skill == "analyze_screen" else "📷",
            "title": "Screen Vision" if skill == "analyze_screen" else "Camera Vision",
            "subtitle": str(data.get("active_window") or data.get("vision_provider") or res.get("message", ""))[:110],
            "status": "ok" if ok else "error",
            "metrics": [{"label": "VIEW", "value": str(data.get("resolution") or "LIVE")}],
        }
    if skill == "run_project_tests":
        return {
            "kind": "tests",
            "icon": "🧪" if ok else "⚠",
            "title": "Pytest Suite" + (" Passed" if ok else " Failures"),
            "subtitle": str(res.get("message", ""))[:110],
            "status": "ok" if ok else "error",
            "metrics": [
                {"label": "PASSED", "value": str(data.get("passed", 0))},
                {"label": "FAILED", "value": str(data.get("failed", 0))},
            ],
        }
    if skill == "control_smart_device":
        lvl = data.get("brightness")
        return {
            "kind": "iot",
            "icon": "💡",
            "title": str(data.get("device") or "Smart Device").title(),
            "subtitle": f"State: {str(data.get('state', '')).upper()} • {data.get('transport', 'iot')}",
            "status": "ok" if ok else "error",
            "progress": int(lvl) if isinstance(lvl, (int, float)) and data.get("state") == "on" else None,
            "metrics": [{"label": "STATE", "value": str(data.get("state", "OFF")).upper()}],
        }
    if skill == "switch_persona":
        return {
            "kind": "persona",
            "icon": "🎭",
            "title": f"Persona: {data.get('label', 'Updated')}",
            "subtitle": str(data.get("description") or res.get("message", ""))[:110],
            "status": "ok",
            "metrics": [{"label": "VOICE", "value": str(data.get("edge_voice", ""))[:18]}],
        }
    if skill in ("set_reminder", "manage_tasks", "manage_notes", "take_screenshot", "web_search_and_open", "media_control", "calculate", "git_workflow", "index_codebase_rag"):
        icons = {
            "set_reminder": "⏱",
            "manage_tasks": "✅",
            "manage_notes": "📝",
            "take_screenshot": "📸",
            "web_search_and_open": "🌐",
            "media_control": "🎵",
            "calculate": "🧮",
            "git_workflow": "🌿",
            "index_codebase_rag": "🧠",
        }
        return {
            "kind": skill,
            "icon": icons.get(skill, "⚙"),
            "title": str(data.get("label") or skill.replace("_", " ").title()),
            "subtitle": str(res.get("message", ""))[:110],
            "status": "ok" if ok else "error",
            "url": str(data.get("url") or "") if skill == "web_search_and_open" and ok else None,
            "open_url": str(data.get("url") or "") if data.get("open_in_browser") and ok else None,
            "metrics": [{"label": "LATENCY", "value": f"{lat:.0f}ms"}],
        }
    return None


def _extract_open_url(tool_events: list[dict[str, Any]]) -> str | None:
    """Return a URL requested to open directly in the user's active browser tab."""
    for ev in reversed(tool_events or []):
        res = ev.get("result") or {}
        data = res.get("data") or {}
        if res.get("ok") and data.get("open_in_browser") and data.get("url"):
            url = str(data["url"]).strip()
            if url.startswith(("http://", "https://")):
                return url
    return None


class JarvisEngine:
    """Central coordinator for the J.A.R.V.I.S. AI Voice Control Assistant."""

    def __init__(
        self,
        config: JarvisConfig | None = None,
        provider_override: BaseLLMProvider | None = None,
        registry: SkillRegistry | None = None,
        enable_tts: bool = True,
    ) -> None:
        self.config = config or JarvisConfig.load()
        self.config.ensure_directories()

        self.event_bus = EventBus()
        self.db = JarvisDatabase(self.config.db_path)
        self.vector_store = SemanticVectorStore(self.db)
        self.safety = SafetyGuard(require_confirmation=self.config.require_confirmation)
        self.registry = registry or create_default_registry()
        self.brain = JarvisBrain(self.config, self.registry, provider_override=provider_override)

        self.audio = AudioRecorder(
            energy_threshold=self.config.mic_energy_threshold,
            silence_duration=self.config.silence_duration,
            max_duration=self.config.max_listen_seconds,
            vad_engine=self.config.vad_engine,
        )
        self.stt = STTEngine(self.config)
        self.tts = TTSEngine(self.config)
        self.wake_detector = WakeWordDetector(
            self.config.wake_words,
            picovoice_access_key=self.config.picovoice_access_key,
        )
        self.turn_detector = SemanticTurnDetector()
        self.mcp = MCPManager(self.config.data_dir / "mcp_servers.json")
        self.proactive = ProactiveMonitor(
            knowledge_dir=self.config.knowledge_dir,
            on_alert=self._handle_proactive_alert,
            on_knowledge_changed=self._auto_index_knowledge_file,
            battery_low_percent=self.config.battery_low_threshold,
        )
        self.orchestrator_info = inspect_orchestrators(
            self.config.orchestrator,
            gemini_configured=bool(self.config.gemini_api_key),
            openai_configured=bool(self.config.openai_api_key),
        )

        if not enable_tts:
            self.config.speak_responses = False

        self.state: AssistantState = AssistantState.IDLE
        self.default_session_id = "main"
        self._running = False
        self._tasks: list[asyncio.Task[Any]] = []
        self._voice_loop_task: asyncio.Task[Any] | None = None
        self._voice_stop_event = threading.Event()
        self._voice_active = False
        self._voice_require_wake_word = True
        self._started_at = time.time()

    def _make_context(self, session_id: str | None = None, defer_browser_open: bool = False) -> SkillContext:
        return SkillContext(
            config=self.config,
            db=self.db,
            vector_store=self.vector_store,
            event_bus=self.event_bus,
            safety=self.safety,
            session_id=session_id or self.default_session_id,
            defer_browser_open=defer_browser_open,
        )

    async def set_state(self, new_state: AssistantState, detail: str = "") -> None:
        if self.state != new_state or detail:
            self.state = new_state
            await self.event_bus.publish(
                EventType.STATE_CHANGED,
                {"state": new_state.value, "detail": detail},
                source="engine",
            )

    async def start(self, start_background_tasks: bool = True) -> None:
        """Initialize knowledge RAG index and start background schedulers."""
        if self._running:
            return
        self._running = True
        self._started_at = time.time()

        # Index local knowledge docs into vector store if present
        if self.config.knowledge_dir.exists():
            try:
                summary = await asyncio.to_thread(
                    self.vector_store.ingest_directory, self.config.knowledge_dir
                )
                logger.info(
                    "Indexed %s knowledge doc(s) (%s chunks).",
                    summary["files_indexed"],
                    summary["total_chunks"],
                )
            except Exception as exc:
                logger.warning("Knowledge auto-indexing skipped: %s", exc)

        if start_background_tasks:
            # Discover optional external MCP server tools asynchronously
            try:
                mcp_res = await self.mcp.discover_and_register(self.registry)
                if mcp_res["registered"]:
                    logger.info("Registered %d external MCP tools.", len(mcp_res["registered"]))
            except Exception as exc:
                logger.debug("MCP startup discovery skipped: %s", exc)
            # Warm up local OpenVoice v2 neural voice cloner in background so first spoken reply is snappy
            self._tasks.append(asyncio.create_task(self.tts.warmup_cloner(), name="jarvis-voice-warmup"))
            self._tasks.append(asyncio.create_task(self._reminder_loop(), name="jarvis-reminders"))
            self._tasks.append(asyncio.create_task(self._telemetry_loop(), name="jarvis-telemetry"))
            if self.config.proactive_alerts_enabled:
                self._tasks.append(asyncio.create_task(self.proactive.run_loop(), name="jarvis-proactive"))

        await self.set_state(AssistantState.IDLE, "Systems online")

    async def _auto_index_knowledge_file(self, path: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self.vector_store.ingest_file, path)

    async def _handle_proactive_alert(self, alert: dict[str, Any]) -> None:
        if not self.config.proactive_alerts_enabled:
            return
        msg = str(alert.get("message") or "").strip()
        if not msg:
            return
        self.db.add_message(self.default_session_id, "assistant", msg, metadata={"type": "proactive", "kind": alert.get("kind")})
        await self.event_bus.publish(EventType.PROACTIVE_ALERT, alert, source="proactive")

    async def stop(self) -> None:
        """Gracefully shut down background loops, mic capture, speech, and database."""
        self._running = False
        self.proactive.stop()
        await self.stop_local_voice_loop()
        self.tts.stop()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.db.close()

    # ------------------------------------------------------------------
    # Command Processing Pipeline (Voice, WebSocket HUD, REST, CLI)
    # ------------------------------------------------------------------
    async def handle_command(
        self,
        text: str,
        source: str = "text",
        session_id: str | None = None,
        speak: bool | None = None,
        include_browser_audio: bool = False,
    ) -> dict[str, Any]:
        """Process a voice or text command end-to-end."""
        sid = session_id or self.default_session_id
        raw = (text or "").strip()
        if not raw:
            return {"ok": False, "reply": "", "state": self.state.value}

        # Instant barge-in / stop command
        if raw.lower().strip(" .!?") in {"stop", "stop talking", "be quiet", "silence", "shut up", "cancel speech"}:
            self.tts.stop()
            await self.set_state(AssistantState.IDLE, "Speech interrupted")
            return {"ok": True, "reply": "Standing by.", "provider": "system", "model": "barge-in", "tool_events": []}

        # Strip wake phrase if user said "Hey Jarvis!" / "Jarvis"
        cleaned = strip_wake_prefix(raw, self.config.wake_words)
        if not cleaned:
            reply = random.choice(ENTRANCE_GREETINGS)
            self.db.add_message(sid, "user", raw, metadata={"source": source})
            await self._deliver_reply(
                reply, sid, "system", "wake", [], speak, include_browser_audio, expression="smiling"
            )
            return {
                "ok": True,
                "reply": reply,
                "provider": "system",
                "model": "wake",
                "expression": "smiling",
                "tool_events": [],
            }

        # Print & log the exact user transcript reaching Python before hitting the brain/LLM
        print(f"TRANSCRIPT: {cleaned} (source={source})", flush=True)
        logger.info("TRANSCRIPT: %s (source=%s)", cleaned, source)

        # Ignore tiny background noise / lone hesitation fillers from speech-to-text
        if _is_stt_noise(cleaned, source):
            logger.debug("Ignored short STT noise artifact: %r", cleaned)
            await self.set_state(AssistantState.IDLE)
            return {
                "ok": False,
                "ignored": True,
                "reply": "",
                "provider": "stt-filter",
                "model": "noise-gate",
                "tool_events": [],
            }

        # Interrupt any previous speech before processing new command
        self.tts.stop()

        await self.event_bus.publish(
            EventType.TRANSCRIPT_FINAL,
            {"text": cleaned, "raw": raw, "source": source, "session_id": sid},
            source=source,
        )
        self.db.add_message(sid, "user", cleaned, metadata={"source": source})

        ctx = self._make_context(sid, defer_browser_open=bool(include_browser_audio))

        # Check pending safety confirmation gate first
        if self.safety.pending is not None:
            decision = self.safety.classify_reply(cleaned)
            if decision == "approve":
                pending = self.safety.pop_if_approved()
                if pending:
                    await self.set_state(AssistantState.EXECUTING, f"Executing confirmed {pending.skill_name}")
                    res = await self.registry.execute(
                        pending.skill_name, pending.arguments, ctx, bypass_confirmation=True
                    )
                    tool_events = [{"skill": pending.skill_name, "arguments": pending.arguments, "result": res.to_dict()}]
                    audio_b64 = await self._deliver_reply(
                        res.message, sid, "safety", "confirmed", tool_events, speak, include_browser_audio
                    )
                    return {
                        "ok": res.ok,
                        "reply": res.message,
                        "provider": "safety",
                        "model": "confirmed",
                        "tool_events": tool_events,
                        "audio_base64": audio_b64,
                    }
            elif decision == "deny":
                pending = self.safety.pending
                self.safety.clear_pending()
                cancel_msg = f"Cancelled {pending.skill_name}." if pending else "Action cancelled."
                audio_b64 = await self._deliver_reply(
                    cancel_msg, sid, "safety", "cancelled", [], speak, include_browser_audio
                )
                return {
                    "ok": True,
                    "reply": cancel_msg,
                    "provider": "safety",
                    "model": "cancelled",
                    "tool_events": [],
                    "audio_base64": audio_b64,
                }

        # Standard AI Brain Processing (keep rolling last 8-10 conversation turns)
        await self.set_state(AssistantState.PROCESSING, "Analyzing command")
        history = self.db.get_recent_messages(sid, limit=20)
        # Exclude the just-added current user message from history list passed to brain
        prior_history = history[:-1] if history else []

        try:
            brain_reply = await self.brain.process(cleaned, ctx, history=prior_history)
        except Exception as exc:
            logger.exception("Brain processing failed: %s", exc)
            await self.set_state(AssistantState.ERROR, str(exc))
            err = f"I encountered an internal error while processing that request: {exc}"
            await self.event_bus.publish(EventType.ERROR, {"message": err}, source="engine")
            await self.set_state(AssistantState.IDLE)
            return {"ok": False, "reply": err, "provider": "error", "model": "error", "tool_events": []}

        if brain_reply.requires_confirmation:
            await self.set_state(AssistantState.AWAITING_CONFIRMATION, "Awaiting user confirmation")

        audio_b64 = await self._deliver_reply(
            text=brain_reply.text,
            session_id=sid,
            provider=brain_reply.provider,
            model=brain_reply.model,
            tool_events=brain_reply.tool_events,
            speak=speak,
            include_browser_audio=include_browser_audio,
            keep_state=brain_reply.requires_confirmation,
            expression=brain_reply.expression,
        )

        return {
            "ok": True,
            "reply": brain_reply.text,
            "provider": brain_reply.provider,
            "model": brain_reply.model,
            "mode": brain_reply.mode,
            "expression": brain_reply.expression,
            "tool_events": brain_reply.tool_events,
            "open_url": _extract_open_url(brain_reply.tool_events),
            "requires_confirmation": brain_reply.requires_confirmation,
            "confirmation": brain_reply.confirmation_data,
            "audio_base64": audio_b64,
        }

    async def resolve_confirmation(self, token: str, approved: bool) -> dict[str, Any]:
        """Approve or deny a pending dangerous skill from the Web HUD button or REST API."""
        pending = self.safety.pending
        if not pending or pending.token != token:
            return {"ok": False, "reply": "No matching pending confirmation found."}
        if not approved:
            self.safety.clear_pending()
            msg = f"Cancelled {pending.skill_name}."
            await self._deliver_reply(msg, self.default_session_id, "safety", "cancelled", [], None, False)
            return {"ok": True, "reply": msg}

        self.safety.pop_if_approved(token)
        ctx = self._make_context(self.default_session_id)
        await self.set_state(AssistantState.EXECUTING, f"Executing {pending.skill_name}")
        res = await self.registry.execute(pending.skill_name, pending.arguments, ctx, bypass_confirmation=True)
        tool_events = [{"skill": pending.skill_name, "arguments": pending.arguments, "result": res.to_dict()}]
        await self._deliver_reply(res.message, self.default_session_id, "safety", "confirmed", tool_events, None, False)
        return {"ok": res.ok, "reply": res.message, "tool_events": tool_events}

    async def _deliver_reply(
        self,
        text: str,
        session_id: str,
        provider: str,
        model: str,
        tool_events: list[dict[str, Any]],
        speak: bool | None,
        include_browser_audio: bool,
        keep_state: bool = False,
        expression: str = "smiling",
    ) -> str | None:
        self.db.add_message(
            session_id,
            "assistant",
            text,
            tool_calls=tool_events or None,
            metadata={"provider": provider, "model": model, "expression": expression},
        )

        # Emit action card & open_url IMMEDIATELY (before ~600ms TTS synthesis) so browser tabs pop open instantly!
        open_url = _extract_open_url(tool_events)
        card = build_action_card(tool_events)
        if card is not None:
            await self.event_bus.publish(EventType.ACTION_CARD, card, source="skills")

        audio_b64: str | None = None
        audio_mime = "audio/mpeg"
        visemes: list[dict[str, Any]] = []
        should_speak = self.config.speak_responses if speak is None else speak
        if include_browser_audio and should_speak:
            synth = await self.tts.synthesize_speech_with_visemes(text, expression=expression)
            audio_b64 = synth.get("audio_base64")  # type: ignore[assignment]
            visemes = synth.get("visemes") or []  # type: ignore[assignment]
            audio_mime = str(synth.get("mime") or "audio/mpeg")

        await self.event_bus.publish(
            EventType.EXPRESSION_CHANGED,
            {"expression": expression},
            source="brain",
        )
        await self.event_bus.publish(
            EventType.RESPONSE_FINAL,
            {
                "text": text,
                "provider": provider,
                "model": model,
                "expression": expression,
                "tool_events": tool_events,
                "card": card,
                "open_url": open_url,
                "session_id": session_id,
                "audio_base64": audio_b64,
                "audio_mime": audio_mime,
                "visemes": visemes,
            },
            source="brain",
        )

        # Speak on host speakers unless browser audio playback was requested exclusively
        if should_speak and not include_browser_audio:
            await self.set_state(AssistantState.SPEAKING, "Speaking response")
            await self.event_bus.publish(EventType.SPEECH_START, {"text": text}, source="tts")
            await self.tts.speak(text, expression=expression)
            await self.event_bus.publish(EventType.SPEECH_END, {}, source="tts")

        if not keep_state:
            await self.set_state(AssistantState.IDLE)
        return audio_b64

    # ------------------------------------------------------------------
    # Local Python Microphone Voice Control Loop
    # ------------------------------------------------------------------
    async def listen_once(self, speak: bool = True) -> dict[str, Any]:
        """Capture a single voice command from the local Python microphone and execute it."""
        self._voice_stop_event.clear()
        await self.set_state(AssistantState.LISTENING, "Listening on local microphone")

        loop = asyncio.get_running_loop()

        def _push_level(level: float) -> None:
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    self.event_bus.publish(EventType.AUDIO_LEVEL, {"level": round(level, 3)}, source="mic")
                )
            )

        try:
            recorded = await asyncio.to_thread(
                self.audio.record_utterance,
                self._voice_stop_event,
                _push_level,
                6.0,
            )
        except Exception as exc:
            await self.set_state(AssistantState.IDLE)
            return {"ok": False, "error": str(exc), "transcript": ""}

        if recorded is None:
            await self.set_state(AssistantState.IDLE)
            return {"ok": False, "error": "No speech detected.", "transcript": ""}

        await self.set_state(AssistantState.PROCESSING, "Transcribing speech")
        transcript = await self.stt.transcribe(recorded)
        if not transcript:
            await self.set_state(AssistantState.IDLE)
            return {"ok": False, "error": "Could not transcribe audio.", "transcript": ""}

        result = await self.handle_command(transcript, source="local_mic", speak=speak)
        result["transcript"] = transcript
        return result

    async def start_local_voice_loop(self, require_wake_word: bool = True) -> dict[str, Any]:
        """Start continuous hands-free local microphone listening in the background."""
        mic_status = self.audio.status()
        if not mic_status["available"]:
            return {
                "ok": False,
                "message": "No local Python audio input device detected. Use the HUD Browser Microphone or install `sounddevice`.",
            }
        if self._voice_active:
            return {"ok": True, "message": "Local voice loop is already running."}

        self._voice_require_wake_word = require_wake_word
        self._voice_stop_event.clear()
        self._voice_active = True
        self._voice_loop_task = asyncio.create_task(self._continuous_voice_worker(), name="jarvis-voice-loop")
        return {"ok": True, "message": "Hands-free local voice control activated."}

    async def stop_local_voice_loop(self) -> dict[str, Any]:
        self._voice_active = False
        self._voice_stop_event.set()
        if self._voice_loop_task:
            self._voice_loop_task.cancel()
            await asyncio.gather(self._voice_loop_task, return_exceptions=True)
            self._voice_loop_task = None
        await self.set_state(AssistantState.IDLE)
        return {"ok": True, "message": "Local voice control paused."}

    async def _continuous_voice_worker(self) -> None:
        loop = asyncio.get_running_loop()

        def _push_level(level: float) -> None:
            if not self._voice_active:
                return
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    self.event_bus.publish(EventType.AUDIO_LEVEL, {"level": round(level, 3)}, source="mic")
                )
            )

        while self._voice_active and self._running:
            try:
                await self.set_state(AssistantState.LISTENING, "Awaiting voice command")
                recorded = await asyncio.to_thread(
                    self.audio.record_utterance,
                    self._voice_stop_event,
                    _push_level,
                    5.0,
                )
                if recorded is None or not self._voice_active:
                    continue

                transcript = await self.stt.transcribe(recorded)
                if not transcript:
                    continue

                if self._voice_require_wake_word:
                    detected, remainder = self.wake_detector.detect_and_extract(transcript)
                    if not detected:
                        continue
                    await self.event_bus.publish(
                        EventType.WAKE_DETECTED, {"transcript": transcript, "command": remainder}, source="wake"
                    )
                    if not remainder:
                        await self.tts.speak(f"Yes, {self.config.user_name}?")
                        follow_up = await asyncio.to_thread(
                            self.audio.record_utterance, self._voice_stop_event, _push_level, 6.0
                        )
                        if follow_up:
                            remainder = await self.stt.transcribe(follow_up)
                    if remainder:
                        await self.handle_command(remainder, source="local_mic", speak=True)
                else:
                    await self.handle_command(transcript, source="local_mic", speak=True)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Local voice loop error: %s", exc)
                await asyncio.sleep(1.0)

    # ------------------------------------------------------------------
    # Background Loops: Proactive Reminders & System Telemetry
    # ------------------------------------------------------------------
    async def _reminder_loop(self) -> None:
        while self._running:
            try:
                due_list = self.db.pop_due_reminders()
                for rem in due_list:
                    alert = f"Reminder alert, {self.config.user_name}: {rem['message']}."
                    await self.event_bus.publish(
                        EventType.REMINDER_TRIGGERED,
                        {"id": rem["id"], "message": rem["message"], "alert": alert},
                        source="scheduler",
                    )
                    self.db.add_message(self.default_session_id, "assistant", alert, metadata={"type": "reminder"})
                    if self.config.speak_responses:
                        await self.tts.speak(alert)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Reminder loop error: %s", exc)
            await asyncio.sleep(1.0)

    async def _telemetry_loop(self) -> None:
        while self._running:
            try:
                stats = await asyncio.to_thread(collect_system_telemetry)
                stats["memory"] = self.db.memory_stats()
                stats["state"] = self.state.value
                stats["local_voice_active"] = self._voice_active
                await self.event_bus.publish(EventType.TELEMETRY, stats, source="telemetry")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Telemetry loop error: %s", exc)
            await asyncio.sleep(2.5)

    # ------------------------------------------------------------------
    # Diagnostics & Live Settings
    # ------------------------------------------------------------------
    def update_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        applied = self.config.apply_runtime_updates(updates)
        self.safety.require_confirmation = self.config.require_confirmation
        self.wake_detector.update_wake_words(self.config.wake_words)
        self.audio.energy_threshold = self.config.mic_energy_threshold
        self.audio.silence_duration = self.config.silence_duration
        self.brain.reset_provider_cache()
        return applied

    async def llm_diagnostics(self) -> dict[str, Any]:
        """Inspect configured LLM providers and probe local Ollama models if running."""
        import httpx

        ollama_running = False
        ollama_models: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                r = await client.get(f"{self.config.ollama_url}/api/tags")
                if r.status_code == 200:
                    ollama_running = True
                    ollama_models = [
                        str(m.get("name", ""))
                        for m in (r.json().get("models") or [])
                        if m.get("name")
                    ]
        except Exception:
            pass

        self.brain._ollama_checked = True
        self.brain._ollama_ok = ollama_running

        return {
            "active": self.brain.active_provider_info(),
            "llm_provider": self.config.llm_provider,
            "configured": {
                "gemini": bool(self.config.gemini_api_key),
                "openai": bool(self.config.openai_api_key),
                "groq": bool(self.config.groq_api_key),
                "ollama": ollama_running,
            },
            "models": {
                "gemini": self.config.gemini_model,
                "openai": self.config.openai_model,
                "groq": self.config.groq_model,
                "ollama": self.config.ollama_model,
            },
            "ollama_url": self.config.ollama_url,
            "ollama_models": ollama_models,
        }

    async def configure_llm(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist LLM provider/key configuration to `.env` and activate immediately."""
        saved = self.config.save_llm_config(
            llm_provider=payload.get("llm_provider"),
            gemini_api_key=payload.get("gemini_api_key"),
            gemini_model=payload.get("gemini_model"),
            openai_api_key=payload.get("openai_api_key"),
            openai_model=payload.get("openai_model"),
            openai_base_url=payload.get("openai_base_url"),
            groq_api_key=payload.get("groq_api_key"),
            groq_model=payload.get("groq_model"),
            ollama_url=payload.get("ollama_url"),
            ollama_model=payload.get("ollama_model"),
        )
        self.brain.reset_provider_cache()
        diag = await self.llm_diagnostics()

        test_note = ""
        if payload.get("test_connection"):
            prov = await self.brain._get_usable_provider()
            if prov is not None:
                try:
                    resp = await prov.chat(
                        messages=[{"role": "user", "content": "Say 'Connected' in two words."}],
                        system_prompt="Reply briefly.",
                        openai_tools=[],
                        gemini_declarations=[],
                        temperature=0.0,
                    )
                    test_note = f"Live test OK ({prov.name}/{prov.model}): {resp.text[:80]}"
                except Exception as exc:
                    return {
                        "ok": False,
                        "message": f"Saved config, but {prov.name} test failed: {exc}",
                        "saved": saved,
                        "diagnostics": diag,
                    }
            else:
                test_note = "Saved — built-in Web Knowledge + Offline Router active."

        await self.event_bus.publish(
            EventType.SETTINGS_UPDATED,
            {"llm": diag["active"], "status": self.get_status()},
            source="llm_config",
        )
        return {
            "ok": True,
            "message": test_note or f"Active AI Brain: {diag['active']['provider']} ({diag['active']['model']})",
            "saved": saved,
            "diagnostics": diag,
        }

    async def handle_audio_pcm(
        self,
        pcm16_bytes: bytes,
        sample_rate: int = 16000,
        speak: bool = True,
        include_browser_audio: bool = True,
    ) -> dict[str, Any]:
        """Transcribe raw browser-streamed PCM16 audio via Python STT and execute the command."""
        if not pcm16_bytes:
            return {"ok": False, "reply": "", "transcript": ""}
        await self.set_state(AssistantState.PROCESSING, "Transcribing streamed audio")
        wav_bytes = pcm16_to_wav(pcm16_bytes, sample_rate=sample_rate, channels=1)
        recorded = RecordedAudio(
            pcm16_bytes=pcm16_bytes,
            wav_bytes=wav_bytes,
            sample_rate=sample_rate,
            duration_seconds=round(len(pcm16_bytes) / float(sample_rate * 2), 2),
        )
        transcript = await self.stt.transcribe(recorded)
        if not transcript:
            await self.set_state(AssistantState.IDLE)
            return {"ok": False, "reply": "I didn't catch that — could you repeat?", "transcript": ""}
        res = await self.handle_command(
            transcript,
            source="ws_audio_stream",
            speak=speak,
            include_browser_audio=include_browser_audio,
        )
        res["transcript"] = transcript
        return res

    def get_status(self) -> dict[str, Any]:
        return {
            "assistant_name": self.config.assistant_name,
            "state": self.state.value,
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "orchestrator": self.orchestrator_info.to_dict(),
            "llm": self.brain.active_provider_info(),
            "persona": get_persona(self.config.persona).to_dict(),
            "personas_available": [p.to_dict() for p in PERSONA_PROFILES.values()],
            "supervisor_agents": self.brain.supervisor.list_agents(),
            "vector_backend": vector_backend_info(),
            "mcp": self.mcp.status(),
            "proactive_alerts_enabled": self.config.proactive_alerts_enabled,
            "stt": {
                "active": self.stt.active_engine_name(),
                "available": self.stt.available_engines(),
            },
            "tts": {
                "active": self.tts.active_engine_name(),
                "available": self.tts.available_engines(),
                "voice": self.config.edge_tts_voice,
            },
            "wake_word": {
                "active": self.wake_detector.active_engine,
                "available": self.wake_detector.available_engines(),
                "phrases": self.config.wake_words,
            },
            "microphone": self.audio.status(),
            "local_voice_active": self._voice_active,
            "skills_count": len(self.registry.list_skills()),
            "enabled_skills_count": len(self.registry.list_skills(enabled_only=True)),
            "memory": self.db.memory_stats(),
            "pending_confirmation": self.safety.pending.to_dict() if self.safety.pending else None,
            "config": self.config.to_public_dict(),
        }
