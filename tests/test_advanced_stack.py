"""Tests for Advanced J.A.R.V.I.S. Stack:
- Semantic Turn Detection (paused-to-think vs finished sentence)
- Supervisor Pattern Multi-Agent Router & Persona / Tone Modulation
- Model Context Protocol (MCP) Client & Registry Bridge
- Vision, PDF & Codebase RAG
- Proactive Background Monitor (file-watcher auto-RAG)
- Smart Home & IoT Control (lights/dimmer/state)
- Coding Workflows & High-Impact Human-in-the-Loop Safety Confirmation
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jarvis.ai.intent_router import IntentRouter
from jarvis.ai.providers import BaseLLMProvider, LLMResponse
from jarvis.ai.supervisor import AgentSupervisor, PERSONA_PROFILES, classify_interaction_mode, get_persona
from jarvis.config import JarvisConfig
from jarvis.core.engine import JarvisEngine
from jarvis.core.events import EventBus
from jarvis.core.mcp_client import MCPManager, MCPServerConfig
from jarvis.core.proactive import ProactiveMonitor
from jarvis.core.safety import SafetyGuard
from jarvis.memory.database import JarvisDatabase
from jarvis.memory.vector_store import SemanticVectorStore, extract_pdf_text
from jarvis.skills.developer_tools import _summarize_pytest_output
from jarvis.skills.loader import create_default_registry
from jarvis.skills.registry import SkillContext
from jarvis.voice.turn_detector import SemanticTurnDetector
from jarvis.voice.voice_cloner import (
    EMOTION_VOICE_PROFILES,
    compute_openvoice_spectrogram,
    get_emotion_voice_profile,
)
from jarvis.web.server import create_app


def test_semantic_turn_detection_distinguishes_pause_from_completion() -> None:
    td = SemanticTurnDetector()

    # Mid-thought trailing word ("remind me to") should HOLD on 700ms pause
    d_pause = td.evaluate("remind me to", silence_ms=700, vad_active=False)
    assert d_pause.is_complete is False
    assert "mid_thought_pause" in d_pause.reason

    # Same incomplete phrase eventually times out after long silence
    d_long = td.evaluate("remind me to", silence_ms=2200, vad_active=False)
    assert d_long.is_complete is True

    # Complete sentence with punctuation completes rapidly
    d_done = td.evaluate("What time is it?", silence_ms=520, vad_active=False)
    assert d_done.is_complete is True

    # Active speech never cuts off
    d_vad = td.evaluate("open youtube", silence_ms=0, vad_active=True)
    assert d_vad.is_complete is False


def test_supervisor_routing_and_personas() -> None:
    sup = AgentSupervisor()
    assert sup.route("what's wrong with this error on my screen").name == "research_web_agent"
    assert sup.route("run pytest and check git branches").name == "code_workspace_agent"
    assert sup.route("turn on desk lamp and set volume to 50").name == "system_os_agent"
    assert sup.route("remember my project notes and remind me").name == "memory_productivity_agent"

    assert set(PERSONA_PROFILES.keys()) == {"calm", "crisp", "witty", "analytical"}
    assert get_persona("witty").name == "witty"
    assert get_persona("unknown_mode").name == "calm"


def test_advanced_intent_routing_vision_iot_dev_persona() -> None:
    router = IntentRouter()

    m_vis = router.match("what's wrong with this error on my screen")
    assert m_vis is not None and m_vis.skill_name == "analyze_screen"

    m_cam = router.match("what do you see on my webcam")
    assert m_cam is not None and m_cam.skill_name == "analyze_camera"

    m_iot_on = router.match("turn on desk lamp")
    assert m_iot_on is not None and m_iot_on.skill_name == "control_smart_device"
    assert m_iot_on.arguments == {"device": "desk lamp", "action": "on"}

    m_iot_dim = router.match("dim room lights to 40 percent")
    assert m_iot_dim is not None and m_iot_dim.skill_name == "control_smart_device"
    assert m_iot_dim.arguments == {"device": "room lights", "action": "dim", "level": 40}

    m_test = router.match("run tests")
    assert m_test is not None and m_test.skill_name == "run_project_tests"

    m_git = router.match("switch branch to feature-voice")
    assert m_git is not None and m_git.skill_name == "git_workflow"
    assert m_git.arguments == {"action": "switch", "branch": "feature-voice"}

    m_shell = router.match("run shell command echo hello")
    assert m_shell is not None and m_shell.skill_name == "run_shell_command"

    m_del = router.match("delete file temp.txt")
    assert m_del is not None and m_del.skill_name == "delete_workspace_file"

    m_kill = router.match("kill process notepad.exe")
    assert m_kill is not None and m_kill.skill_name == "kill_process"

    m_persona = router.match("switch persona to witty")
    assert m_persona is not None and m_persona.skill_name == "switch_persona"
    assert m_persona.arguments == {"persona": "witty"}


def test_pytest_summary_formatting_and_pdf_rag_fallback(tmp_path: Path) -> None:
    msg_ok, data_ok = _summarize_pytest_output("15 passed in 2.30s", "", 0)
    assert "15 tests passed" in msg_ok and data_ok["failed"] == 0

    msg_fail, data_fail = _summarize_pytest_output(
        "FAILED tests/test_demo.py::test_alpha\n10 passed, 1 failed in 1.2s", "", 1
    )
    assert "1 failed" in msg_fail and "test_alpha" in msg_fail and data_fail["failed"] == 1

    # Pure-Python PDF text extraction fallback test
    fake_pdf = tmp_path / "lecture.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\nstream\nBT (Quantum entanglement links particles) Tj ET\nendstream\n%%EOF")
    extracted = extract_pdf_text(fake_pdf)
    assert "Quantum entanglement" in extracted

    db = JarvisDatabase(tmp_path / "rag.db")
    vs = SemanticVectorStore(db)
    res = vs.ingest_file(fake_pdf)
    assert res["chunks_indexed"] >= 1
    hits = vs.search("quantum entanglement")
    assert hits and "Quantum entanglement" in hits[0]["content"]
    db.close()


@pytest.mark.asyncio
async def test_proactive_watcher_iot_mcp_and_guarded_high_impact_skills(tmp_path: Path) -> None:
    kb = tmp_path / "knowledge"
    kb.mkdir()
    cfg = JarvisConfig(
        llm_provider="offline",
        speak_responses=False,
        data_dir=tmp_path / "data",
        knowledge_dir=kb,
        captures_dir=tmp_path / "captures",
    )
    cfg.ensure_directories()
    db = JarvisDatabase(cfg.db_path)
    vs = SemanticVectorStore(db)
    ctx = SkillContext(cfg, db, vs, EventBus(), SafetyGuard(require_confirmation=True))
    reg = create_default_registry()

    # Smart Home IoT state control & persistence
    r_on = await reg.execute("control_smart_device", {"device": "desk lamp", "action": "on"}, ctx)
    assert r_on.ok and r_on.data["state"] == "on"
    r_dim = await reg.execute("control_smart_device", {"device": "room lights", "action": "dim", "level": 35}, ctx)
    assert r_dim.ok and r_dim.data["brightness"] == 35
    r_list = await reg.execute("list_smart_devices", {}, ctx)
    assert r_list.ok and "desk lamp (on)" in r_list.message

    # High-impact actions MUST trigger Human-in-the-Loop confirmation (never execute silently!)
    r_shell = await reg.execute("run_shell_command", {"command": "echo danger"}, ctx)
    assert r_shell.requires_confirmation is True
    ctx.safety.clear_pending()

    r_del = await reg.execute("delete_workspace_file", {"relative_path": "nonexistent.tmp"}, ctx)
    assert r_del.requires_confirmation is True
    ctx.safety.clear_pending()

    r_kill = await reg.execute("kill_process", {"process_name": "notepad.exe"}, ctx)
    assert r_kill.requires_confirmation is True
    ctx.safety.clear_pending()

    # Persona modulation skill updates config cleanly
    r_per = await reg.execute("switch_persona", {"persona": "analytical"}, ctx)
    assert r_per.ok and cfg.persona == "analytical"

    # Proactive background monitor detects newly dropped knowledge file & auto-indexes it!
    alerts_seen: list[dict] = []

    async def _on_alert(a: dict) -> None:
        alerts_seen.append(a)

    async def _on_kb(p: Path) -> dict:
        return vs.ingest_file(p)

    mon = ProactiveMonitor(knowledge_dir=kb, on_alert=_on_alert, on_knowledge_changed=_on_kb)
    note_file = kb / "study_notes.md"
    note_file.write_text("Microservices communicate asynchronously over event buses.", encoding="utf-8")
    fired = await mon.check_once()
    assert any(a["kind"] == "knowledge_updated" for a in fired)
    assert vs.search("event buses")

    # MCP config persistence & server listing
    mcp = MCPManager(cfg.data_dir / "mcp_servers.json")
    mcp.add_server(MCPServerConfig(name="local_fs", transport="http", url="http://127.0.0.1:9999/mcp"))
    assert len(mcp.status()["servers"]) == 1
    assert mcp.remove_server("local_fs") is True

    db.close()


def test_advanced_api_endpoints_mcp_turn_and_status(tmp_path: Path) -> None:
    cfg = JarvisConfig(
        llm_provider="offline",
        speak_responses=False,
        data_dir=tmp_path / "data",
        knowledge_dir=tmp_path / "knowledge",
        captures_dir=tmp_path / "captures",
    )
    engine = JarvisEngine(config=cfg, enable_tts=False)
    app = create_app(engine=engine, start_background_tasks=False)

    with TestClient(app) as client:
        st = client.get("/api/status").json()
        assert len(st["supervisor_agents"]) == 4
        assert st["persona"]["name"] == "calm"
        assert "vector_backend" in st
        assert "mcp" in st

        tc = client.post("/api/turn/check", json={"transcript": "open", "silence_ms": 500}).json()
        assert tc["is_complete"] is False

        tc2 = client.post("/api/turn/check", json={"transcript": "What time is it?", "silence_ms": 600}).json()
        assert tc2["is_complete"] is True

        mcp_st = client.get("/api/mcp/servers").json()
        assert mcp_st["protocol"] == "MCP JSON-RPC 2.0"


class _SpyChatLLM(BaseLLMProvider):
    name = "spy-llm"
    model = "spy-v1"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(
        self,
        messages: list[dict],
        system_prompt: str,
        openai_tools: list[dict],
        gemini_declarations: list[dict],
        temperature: float = 0.4,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "system_prompt": system_prompt,
                "openai_tools": list(openai_tools),
                "gemini_declarations": list(gemini_declarations),
            }
        )
        return LLMResponse(text="Sounds like a solid evening — what vibe are you feeling?", provider=self.name, model=self.model)


@pytest.mark.asyncio
async def test_mode_gating_disables_search_on_casual_chat_and_preserves_rolling_history(tmp_path: Path) -> None:
    # 1. Explicit chat / search / device classification checks
    assert classify_interaction_mode("Hey, long day today.") == "chat"
    assert classify_interaction_mode("I was thinking about movies earlier.") == "chat"
    assert classify_interaction_mode("I got some great news today") == "chat"
    assert classify_interaction_mode("What's the weather in Jabalpur right now?") == "search"
    assert classify_interaction_mode("Who is Nikola Tesla?") == "search"
    assert classify_interaction_mode("open youtube") == "device"
    assert classify_interaction_mode("set volume to 50") == "device"

    # Weather intent cleans trailing "right now"
    router = IntentRouter()
    m_wx = router.match("What's the weather in Jabalpur right now?")
    assert m_wx is not None and m_wx.skill_name == "get_weather" and m_wx.arguments["city"] == "jabalpur"
    # Casual "great news" must NOT trigger BBC headlines
    m_casual_news = router.match("I got some great news today")
    assert m_casual_news is None or m_casual_news.skill_name != "get_top_news"

    # 2. LLM path: tools MUST be physically empty on mode=='chat', enabled on mode=='search', and rolling history preserved
    spy = _SpyChatLLM()
    cfg = JarvisConfig(
        fast_path_routing=False,
        speak_responses=False,
        data_dir=tmp_path / "data_llm",
        knowledge_dir=tmp_path / "kb_llm",
        captures_dir=tmp_path / "cap_llm",
    )
    eng = JarvisEngine(config=cfg, provider_override=spy, enable_tts=False)
    await eng.start(start_background_tasks=False)

    r1 = await eng.handle_command("Hey, long day today.", source="chat_text", speak=False)
    assert r1["mode"] == "chat"
    assert spy.calls[-1]["openai_tools"] == []
    assert spy.calls[-1]["gemini_declarations"] == []
    assert "DO NOT call web_search" in spy.calls[-1]["system_prompt"]

    r2 = await eng.handle_command("I was thinking about movies earlier.", source="chat_text", speak=False)
    assert r2["mode"] == "chat"
    assert spy.calls[-1]["openai_tools"] == []
    # Rolling conversation history carries Turn 1 into Turn 2
    sent_texts = [m["content"] for m in spy.calls[-1]["messages"]]
    assert "Hey, long day today." in sent_texts
    assert "I was thinking about movies earlier." in sent_texts

    # Explicit search turn DOES attach web tools
    r3 = await eng.handle_command("What's the weather in Jabalpur right now?", source="chat_text", speak=False)
    assert r3["mode"] == "search"
    tool_names = {t["function"]["name"] for t in spy.calls[-1]["openai_tools"]}
    assert "get_weather" in tool_names

    # Short STT noise filler on voice source is ignored cleanly
    r_noise = await eng.handle_command("um", source="hands_free_voice", speak=False)
    assert r_noise.get("ignored") is True

    await eng.stop()

    # 3. Strictly offline path: casual chat NEVER dumps Wikipedia and NEVER repeats consecutive replies
    cfg_off = JarvisConfig(
        llm_provider="offline",
        speak_responses=False,
        data_dir=tmp_path / "data_off",
        knowledge_dir=tmp_path / "kb_off",
        captures_dir=tmp_path / "cap_off",
    )
    eng_off = JarvisEngine(config=cfg_off, enable_tts=False)
    await eng_off.start(start_background_tasks=False)

    rc1 = await eng_off.handle_command("Hey there buddy..", source="chat_text", speak=False)
    assert rc1["ok"] is True and rc1["mode"] == "chat" and rc1["tool_events"] == []
    rc2 = await eng_off.handle_command("What's going on..?", source="chat_text", speak=False)
    assert rc2["ok"] is True and rc2["mode"] == "chat" and rc2["tool_events"] == []
    rc3 = await eng_off.handle_command("Random question one?", source="chat_text", speak=False)
    rc4 = await eng_off.handle_command("Random question two?", source="chat_text", speak=False)
    assert rc3["reply"] != rc4["reply"]
    assert "Connect Gemini" not in rc3["reply"]

    await eng_off.stop()

    # 4. Zero-key Free AI fallback path: open-ended banter uses free provider with rolling history & no search tools
    cfg_free = JarvisConfig(
        llm_provider="auto",
        free_ai_fallback=True,
        speak_responses=False,
        data_dir=tmp_path / "data_free",
        knowledge_dir=tmp_path / "kb_free",
        captures_dir=tmp_path / "cap_free",
    )
    eng_free = JarvisEngine(config=cfg_free, enable_tts=False)
    eng_free.brain._ollama_checked = True
    eng_free.brain._ollama_ok = False
    spy_free = _SpyChatLLM()
    eng_free.brain._free_provider = spy_free  # type: ignore[assignment]
    await eng_free.start(start_background_tasks=False)

    rf = await eng_free.handle_command(
        "So, if i wanna give you a treat... what will be the best way..?",
        source="chat_text",
        speak=False,
    )
    assert rf["ok"] is True and rf["mode"] == "chat" and rf["tool_events"] == []
    assert rf["provider"] == "spy-llm"
    assert spy_free.calls[-1]["openai_tools"] == []

    await eng_free.stop()


def test_multi_emotional_cloned_voice_profiles_and_spectrogram() -> None:
    import numpy as np

    expected_emotions = {
        "neutral", "smiling", "happy", "laughing", "surprised",
        "listening", "thinking", "serious", "sad", "angry",
    }
    assert expected_emotions.issubset(set(EMOTION_VOICE_PROFILES.keys()))
    assert get_emotion_voice_profile("happy").pitch.startswith("+")
    assert get_emotion_voice_profile("sad").pitch.startswith("-")
    assert get_emotion_voice_profile("sad").rate.startswith("-")
    assert get_emotion_voice_profile("angry").volume.startswith("+")

    # Verify exact OpenVoice v2 spectrogram geometry (T, 513) with zero NaNs
    dummy_wav = np.sin(2 * np.pi * 220.0 * np.arange(22050, dtype=np.float32) / 22050.0)
    spec = compute_openvoice_spectrogram(dummy_wav)
    assert spec.ndim == 2 and spec.shape[1] == 513 and spec.shape[0] > 70
    assert np.isfinite(spec).all() and float(spec.min()) > 0.0
