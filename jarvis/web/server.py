"""FastAPI + WebSocket server powering the J.A.R.V.I.S. Holographic Web HUD & REST API."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from jarvis import __version__
from jarvis.core.engine import JarvisEngine
from jarvis.core.events import Event, EventType
from jarvis.core.mcp_client import MCPServerConfig
from jarvis.skills.web_intelligence import open_url_on_os

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class CommandRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    source: str = "web_hud"
    session_id: str | None = None
    speak: bool | None = None
    browser_audio: bool = False


class ConfirmRequest(BaseModel):
    token: str
    approved: bool


class LocalVoiceRequest(BaseModel):
    action: str = Field(..., pattern="^(start|stop|once)$")
    require_wake_word: bool = True


class ToggleSkillRequest(BaseModel):
    enabled: bool


class NoteCreateRequest(BaseModel):
    title: str = ""
    content: str = Field(..., min_length=1)
    tags: str = ""


class TaskCreateRequest(BaseModel):
    title: str = Field(..., min_length=1)
    priority: str = "normal"


class MemoryCreateRequest(BaseModel):
    content: str = Field(..., min_length=1)
    namespace: str = "fact"


class LLMConfigureRequest(BaseModel):
    llm_provider: str = "auto"
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None
    openai_base_url: str | None = None
    groq_api_key: str | None = None
    groq_model: str | None = None
    ollama_url: str | None = None
    ollama_model: str | None = None
    test_connection: bool = False


class MCPServerCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    transport: str = Field(default="http", pattern="^(http|stdio)$")
    url: str = ""
    command: str = ""
    args: list[str] = Field(default_factory=list)


class TurnCheckRequest(BaseModel):
    transcript: str = ""
    silence_ms: int = 0
    vad_active: bool = False


class VisionFrameRequest(BaseModel):
    image_base64: str = Field(..., min_length=20)
    mime_type: str = "image/jpeg"
    question: str = "Describe what is currently shown on my screen and highlight any errors or key info."
    speak: bool = True


class ConnectionManager:
    """Tracks active WebSocket HUD clients and broadcasts EventBus updates."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self._connections:
            return
        # Send raw UTF-8 JSON (ensure_ascii=False) so Hindi Devanagari subtitles never arrive as \uXXXX escapes
        encoded = json.dumps(message, ensure_ascii=False, default=str)
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_text(encoded)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.discard(ws)


def create_app(engine: JarvisEngine | None = None, start_background_tasks: bool = True) -> FastAPI:
    """Create and wire the FastAPI application around a JarvisEngine instance."""
    jarvis_engine = engine or JarvisEngine()
    ws_manager = ConnectionManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def _forward_event(ev: Event) -> None:
            await ws_manager.broadcast(ev.to_dict())

        unsub = jarvis_engine.event_bus.subscribe(None, _forward_event)
        await jarvis_engine.start(start_background_tasks=start_background_tasks)
        try:
            yield
        finally:
            unsub()
            await jarvis_engine.stop()

    app = FastAPI(
        title="J.A.R.V.I.S. AI Voice Control Assistant",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.engine = jarvis_engine
    app.state.ws_manager = ws_manager

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def no_cache_static(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_hud() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="HUD frontend not found.")
        return FileResponse(index_path)

    # ------------------------------------------------------------------
    # Core Engine & Voice Control REST Endpoints
    # ------------------------------------------------------------------
    @app.get("/api/status")
    async def get_status() -> dict[str, Any]:
        return jarvis_engine.get_status()

    @app.post("/api/command")
    async def run_command_api(req: CommandRequest) -> dict[str, Any]:
        return await jarvis_engine.handle_command(
            text=req.text,
            source=req.source,
            session_id=req.session_id,
            speak=req.speak,
            include_browser_audio=req.browser_audio,
        )

    @app.post("/api/confirm")
    async def confirm_action(req: ConfirmRequest) -> dict[str, Any]:
        return await jarvis_engine.resolve_confirmation(req.token, req.approved)

    @app.post("/api/interrupt")
    async def interrupt_speech() -> dict[str, Any]:
        jarvis_engine.tts.stop()
        return {"ok": True, "message": "Speech interrupted."}

    @app.post("/api/voice/local")
    async def control_local_voice(req: LocalVoiceRequest) -> dict[str, Any]:
        if req.action == "start":
            return await jarvis_engine.start_local_voice_loop(require_wake_word=req.require_wake_word)
        if req.action == "stop":
            return await jarvis_engine.stop_local_voice_loop()
        return await jarvis_engine.listen_once()

    # ------------------------------------------------------------------
    # Skills, History, Notes, Tasks, Reminders, Memory/RAG, Audit, Settings
    # ------------------------------------------------------------------
    @app.get("/api/skills")
    async def list_skills() -> dict[str, Any]:
        return {"skills": [s.to_public_dict() for s in jarvis_engine.registry.list_skills()]}

    @app.post("/api/skills/{name}/toggle")
    async def toggle_skill(name: str, req: ToggleSkillRequest) -> dict[str, Any]:
        if not jarvis_engine.registry.set_enabled(name, req.enabled):
            raise HTTPException(status_code=404, detail=f"Skill '{name}' not found.")
        return {"ok": True, "name": name, "enabled": req.enabled}

    @app.get("/api/history")
    async def get_history(session_id: str = "main", limit: int = 40) -> dict[str, Any]:
        return {"messages": jarvis_engine.db.get_recent_messages(session_id, limit=limit)}

    @app.delete("/api/history")
    async def clear_history(session_id: str = "main") -> dict[str, Any]:
        deleted = jarvis_engine.db.clear_session(session_id)
        return {"ok": True, "deleted": deleted}

    @app.get("/api/notes")
    async def list_notes(q: str = "") -> dict[str, Any]:
        return {"notes": jarvis_engine.db.list_notes(query=q)}

    @app.post("/api/notes")
    async def create_note(req: NoteCreateRequest) -> dict[str, Any]:
        title = req.title.strip() or req.content.strip()[:45]
        return {"note": jarvis_engine.db.add_note(title=title, content=req.content, tags=req.tags)}

    @app.delete("/api/notes/{note_id}")
    async def delete_note(note_id: int) -> dict[str, Any]:
        return {"ok": jarvis_engine.db.delete_note(note_id)}

    @app.get("/api/tasks")
    async def list_tasks() -> dict[str, Any]:
        return {"tasks": jarvis_engine.db.list_tasks(include_completed=True)}

    @app.post("/api/tasks")
    async def create_task(req: TaskCreateRequest) -> dict[str, Any]:
        return {"task": jarvis_engine.db.add_task(title=req.title, priority=req.priority)}

    @app.post("/api/tasks/{task_id}/complete")
    async def complete_task(task_id: int) -> dict[str, Any]:
        return {"ok": jarvis_engine.db.complete_task(task_id)}

    @app.delete("/api/tasks/{task_id}")
    async def delete_task(task_id: int) -> dict[str, Any]:
        return {"ok": jarvis_engine.db.delete_task(task_id)}

    @app.get("/api/reminders")
    async def list_reminders() -> dict[str, Any]:
        return {"reminders": jarvis_engine.db.list_reminders(status="pending")}

    @app.delete("/api/reminders/{reminder_id}")
    async def cancel_reminder(reminder_id: int) -> dict[str, Any]:
        return {"ok": jarvis_engine.db.cancel_reminder(reminder_id)}

    @app.get("/api/memories")
    async def list_memories(namespace: str | None = None) -> dict[str, Any]:
        return {
            "memories": jarvis_engine.vector_store.list_memories(namespace=namespace),
            "stats": jarvis_engine.db.memory_stats(),
        }

    @app.post("/api/memories")
    async def add_memory(req: MemoryCreateRequest) -> dict[str, Any]:
        item = jarvis_engine.vector_store.add_memory(content=req.content, namespace=req.namespace, source="hud")
        return {"memory": item}

    @app.post("/api/memories/ingest")
    async def ingest_knowledge() -> dict[str, Any]:
        summary = await asyncio.to_thread(
            jarvis_engine.vector_store.ingest_directory, jarvis_engine.config.knowledge_dir
        )
        return {"ok": True, **summary}

    @app.delete("/api/memories/{memory_id}")
    async def delete_memory(memory_id: int) -> dict[str, Any]:
        return {"ok": jarvis_engine.vector_store.delete_memory(memory_id)}

    @app.get("/api/audit")
    async def get_audit_logs(limit: int = 40) -> dict[str, Any]:
        return {"logs": jarvis_engine.db.list_audit_logs(limit=limit)}

    @app.get("/api/llm/status")
    async def get_llm_status() -> dict[str, Any]:
        return await jarvis_engine.llm_diagnostics()

    @app.post("/api/llm/configure")
    async def configure_llm_api(req: LLMConfigureRequest) -> dict[str, Any]:
        return await jarvis_engine.configure_llm(req.model_dump())

    @app.get("/api/mcp/servers")
    async def get_mcp_servers() -> dict[str, Any]:
        return jarvis_engine.mcp.status()

    @app.post("/api/mcp/servers")
    async def add_mcp_server(req: MCPServerCreateRequest) -> dict[str, Any]:
        jarvis_engine.mcp.add_server(
            MCPServerConfig(
                name=req.name.strip(),
                transport=req.transport,
                url=req.url.strip(),
                command=req.command.strip(),
                args=req.args,
            )
        )
        disc = await jarvis_engine.mcp.discover_and_register(jarvis_engine.registry)
        return {"ok": True, "discovery": disc, "mcp": jarvis_engine.mcp.status()}

    @app.delete("/api/mcp/servers/{name}")
    async def delete_mcp_server(name: str) -> dict[str, Any]:
        return {"ok": jarvis_engine.mcp.remove_server(name), "mcp": jarvis_engine.mcp.status()}

    @app.post("/api/turn/check")
    async def check_turn_endpoint(req: TurnCheckRequest) -> dict[str, Any]:
        return jarvis_engine.turn_detector.evaluate(req.transcript, req.silence_ms, req.vad_active).to_dict()

    @app.post("/api/proactive/check")
    async def trigger_proactive_check() -> dict[str, Any]:
        alerts = await jarvis_engine.proactive.check_once()
        return {"ok": True, "alerts": alerts}

    @app.post("/api/vision/analyze")
    async def analyze_browser_vision_frame(req: VisionFrameRequest) -> dict[str, Any]:
        import time as _time
        from jarvis.ai.providers import OllamaProvider, resolve_llm_provider

        raw_b64 = req.image_base64.split(",", 1)[-1].strip()
        snap_path = jarvis_engine.config.captures_dir / f"vision_browser_{_time.strftime('%Y%m%d_%H%M%S')}.jpg"
        try:
            snap_path.write_bytes(base64.b64decode(raw_b64))
        except Exception:
            pass

        provider = resolve_llm_provider(jarvis_engine.config)
        if isinstance(provider, OllamaProvider) and not await provider.is_available():
            provider = None

        if provider is not None:
            try:
                resp = await provider.chat_with_image(
                    prompt=req.question,
                    image_b64=raw_b64,
                    mime_type=req.mime_type,
                )
                reply = resp.text or "I captured your screen frame cleanly."
                prov_label, mod_label = resp.provider, resp.model
            except Exception as exc:
                reply = f"Captured your screen frame to {snap_path.name}, but the vision model returned an error: {exc}"
                prov_label, mod_label = "vision", "error"
        else:
            reply = (
                f"Captured your screen frame cleanly ({snap_path.name}). "
                "Connect Gemini, OpenAI, or Ollama Vision in the AI Brain panel at top-right so I can read and explain what's on your screen!"
            )
            prov_label, mod_label = "vision", "browser-capture"

        await jarvis_engine._deliver_reply(
            text=reply,
            session_id=jarvis_engine.default_session_id,
            provider=prov_label,
            model=mod_label,
            tool_events=[
                {
                    "skill": "analyze_screen",
                    "arguments": {"question": req.question},
                    "result": {"ok": True, "message": reply, "data": {"path": str(snap_path), "resolution": "BROWSER"}},
                }
            ],
            speak=req.speak,
            include_browser_audio=True,
            expression="thinking",
        )
        return {"ok": True, "reply": reply, "path": str(snap_path)}

    @app.post("/api/settings")
    async def update_settings(updates: dict[str, Any]) -> dict[str, Any]:
        applied = jarvis_engine.update_settings(updates)
        await jarvis_engine.event_bus.publish(
            EventType.SETTINGS_UPDATED,
            {"applied": applied, "status": jarvis_engine.get_status()},
            source="settings",
        )
        return {"ok": True, "applied": applied, "status": jarvis_engine.get_status()}

    # ------------------------------------------------------------------
    # Real-Time Bidirectional WebSocket HUD Stream
    # ------------------------------------------------------------------
    @app.websocket("/ws")
    async def websocket_hud(ws: WebSocket) -> None:
        await ws_manager.connect(ws)
        try:
            hello_payload = {**jarvis_engine.get_status(), "ui_version": 31}
            await ws.send_text(json.dumps({"type": "hello", "payload": hello_payload}, ensure_ascii=False, default=str))
            while True:
                msg = await ws.receive_json()
                mtype = str(msg.get("type", "")).lower()

                if mtype == "ping":
                    await ws.send_json({"type": "pong"})
                elif mtype == "command":
                    text = str(msg.get("text", "")).strip()
                    if text:
                        asyncio.create_task(
                            jarvis_engine.handle_command(
                                text=text,
                                source=str(msg.get("source", "web_voice")),
                                speak=msg.get("speak"),
                                include_browser_audio=bool(msg.get("browser_audio", False)),
                            )
                        )
                elif mtype == "audio_pcm":
                    b64 = str(msg.get("pcm16_base64", "")).strip()
                    sr_val = int(msg.get("sample_rate", 16000))
                    if b64:
                        pcm_bytes = base64.b64decode(b64)
                        asyncio.create_task(
                            jarvis_engine.handle_audio_pcm(
                                pcm16_bytes=pcm_bytes,
                                sample_rate=sr_val,
                                speak=bool(msg.get("speak", True)),
                                include_browser_audio=bool(msg.get("browser_audio", True)),
                            )
                        )
                elif mtype == "interrupt":
                    jarvis_engine.tts.stop()
                elif mtype == "open_url_os":
                    url = str(msg.get("url", "")).strip()
                    if url.startswith(("http://", "https://")):
                        await asyncio.to_thread(open_url_on_os, url)
                elif mtype == "turn_check":
                    decision = jarvis_engine.turn_detector.evaluate(
                        str(msg.get("transcript", "")),
                        int(msg.get("silence_ms", 0)),
                        bool(msg.get("vad_active", False)),
                    )
                    await ws.send_json({"type": "turn_decision", "payload": decision.to_dict()})
                elif mtype == "confirm":
                    token = str(msg.get("token", ""))
                    approved = bool(msg.get("approved", False))
                    asyncio.create_task(jarvis_engine.resolve_confirmation(token, approved))
                elif mtype == "voice_local":
                    action = str(msg.get("action", "stop"))
                    if action == "start":
                        res = await jarvis_engine.start_local_voice_loop(
                            require_wake_word=bool(msg.get("require_wake_word", True))
                        )
                    elif action == "once":
                        res = await jarvis_engine.listen_once()
                    else:
                        res = await jarvis_engine.stop_local_voice_loop()
                    await ws.send_json({"type": "local_voice_status", "payload": res})
        except WebSocketDisconnect:
            ws_manager.disconnect(ws)
        except Exception as exc:
            logger.debug("WebSocket disconnected: %s", exc)
            ws_manager.disconnect(ws)

    return app
