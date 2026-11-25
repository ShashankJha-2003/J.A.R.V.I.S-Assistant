"""Tests for EventBus, SkillRegistry, SafetyGuard, WakeWordDetector, and TTS text cleaning."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import JarvisConfig
from jarvis.core.engine import build_action_card
from jarvis.core.events import Event, EventBus, EventType
from jarvis.core.realtime_pipeline import inspect_orchestrators
from jarvis.core.safety import RiskLevel, SafetyGuard, is_app_allowed, resolve_safe_shortcut
from jarvis.memory.database import JarvisDatabase
from jarvis.memory.vector_store import SemanticVectorStore
from jarvis.skills.registry import SkillContext, SkillRegistry, SkillResult
from jarvis.voice.tts import build_viseme_timeline, clean_text_for_speech, word_to_visemes
from jarvis.voice.vad import VoiceActivityDetector
from jarvis.voice.wake_word import WakeWordDetector


@pytest.mark.asyncio
async def test_event_bus_pubsub_and_isolation() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def good_handler(ev: Event) -> None:
        received.append(ev)

    def broken_handler(_ev: Event) -> None:
        raise RuntimeError("boom")

    unsub = bus.subscribe(EventType.STATE_CHANGED, good_handler)
    bus.subscribe(None, broken_handler)

    await bus.publish(EventType.STATE_CHANGED, {"state": "listening"}, source="test")
    assert len(received) == 1
    assert received[0].payload["state"] == "listening"

    unsub()
    await bus.publish(EventType.STATE_CHANGED, {"state": "idle"}, source="test")
    assert len(received) == 1


@pytest.mark.asyncio
async def test_skill_registry_schemas_and_confirmation_gate(tmp_path: Path) -> None:
    cfg = JarvisConfig(data_dir=tmp_path / "data", knowledge_dir=tmp_path / "kb", captures_dir=tmp_path / "cap")
    cfg.ensure_directories()
    db = JarvisDatabase(cfg.db_path)
    vs = SemanticVectorStore(db)
    bus = EventBus()
    safety = SafetyGuard(require_confirmation=True)
    ctx = SkillContext(config=cfg, db=db, vector_store=vs, event_bus=bus, safety=safety)

    registry = SkillRegistry()
    executed: list[str] = []

    @registry.register(
        name="dangerous_op",
        description="A dangerous test operation",
        category="system",
        risk=RiskLevel.DANGEROUS,
        parameters={
            "type": "object",
            "properties": {"target": {"type": "string", "description": "Target name"}},
            "required": ["target"],
        },
    )
    def dangerous_op(c: SkillContext, target: str) -> SkillResult:
        executed.append(target)
        return SkillResult(ok=True, message=f"Executed on {target}")

    # Verify schema export for OpenAI & Gemini
    oa = registry.openai_tools()
    gm = registry.gemini_declarations()
    assert oa[0]["function"]["name"] == "dangerous_op"
    assert gm[0]["parameters"]["type"] == "OBJECT"
    assert gm[0]["parameters"]["properties"]["target"]["type"] == "STRING"

    # First call must trigger confirmation gate
    res1 = await registry.execute("dangerous_op", {"target": "reactor"}, ctx)
    assert res1.requires_confirmation is True
    assert res1.confirmation_token is not None
    assert executed == []

    # Bypass confirmation executes cleanly
    res2 = await registry.execute("dangerous_op", {"target": "reactor"}, ctx, bypass_confirmation=True)
    assert res2.ok is True
    assert executed == ["reactor"]
    assert len(db.list_audit_logs()) >= 2
    db.close()


def test_wake_word_detector_exact_and_fuzzy() -> None:
    det = WakeWordDetector(["jarvis", "hey jarvis", "ok jarvis"])

    found, cmd = det.detect_and_extract("Hey Jarvis, what is the weather in Tokyo?")
    assert found is True
    assert cmd == "what is the weather in Tokyo?"

    found2, cmd2 = det.detect_and_extract("Jarvis")
    assert found2 is True
    assert cmd2 == ""

    found3, cmd3 = det.detect_and_extract("Jervis open notepad")
    assert found3 is True
    assert cmd3 == "open notepad"

    found4, _ = det.detect_and_extract("play some music please")
    assert found4 is False


def test_tts_text_cleaning() -> None:
    raw = "**Status**: Visit [Docs](https://example.com) and run `python main.py` ```code```"
    cleaned = clean_text_for_speech(raw)
    assert "**" not in cleaned
    assert "```" not in cleaned
    assert "https://" not in cleaned
    assert "Docs" in cleaned


def test_tts_phoneme_viseme_timeline() -> None:
    vis_hello = [v for v, _ in word_to_visemes("hello")]
    assert "EE" in vis_hello and "OO" in vis_hello

    vis_five = [v for v, _ in word_to_visemes("five")]
    assert vis_five[0] == "FV"

    vis_boom = [v for v, _ in word_to_visemes("boom")]
    assert vis_boom[0] == "MBP" and vis_boom[-1] == "MBP" and "OO" in vis_boom

    timeline = build_viseme_timeline([(100, 300, "hello"), (500, 250, "world")])
    # Expect an inter-word REST gap between 400ms and 500ms + trailing REST
    rests = [seg for seg in timeline if seg["v"] == "REST"]
    assert len(rests) >= 2
    assert any(seg["v"] == "OO" for seg in timeline)


def test_allowlist_vad_orchestrator_and_bento_cards() -> None:
    import numpy as np

    # Safe app & shortcut allowlist
    assert is_app_allowed("vs code", strict=True)[0] is True
    assert is_app_allowed("chrome", strict=True)[0] is True
    assert is_app_allowed("spotify", strict=True)[0] is True
    assert is_app_allowed("notepad && rm -rf /", strict=False)[0] is False
    assert is_app_allowed("unknown_malware_bin", strict=True)[0] is False
    assert resolve_safe_shortcut("alt+tab") == ("alt", "tab")
    assert resolve_safe_shortcut("show desktop") == ("win", "d")
    assert resolve_safe_shortcut("rm -rf") is None

    # VAD silence vs voice-like signal
    vad = VoiceActivityDetector(engine="energy", energy_threshold=0.015)
    silence = np.zeros(1600, dtype=np.float32)
    assert vad.is_speech(silence)[0] is False
    t = np.linspace(0, 0.1, 1600, dtype=np.float32)
    voice_sig = (0.12 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    assert vad.is_speech(voice_sig)[0] is True

    # Orchestrator detection
    orch = inspect_orchestrators("custom_asyncio")
    assert orch.active == "custom_asyncio"
    assert orch.barge_in_supported is True

    # Bento action card builder
    card = build_action_card(
        [{"skill": "open_application", "arguments": {"app_name": "chrome"}, "result": {"ok": True, "latency_ms": 11.2}}]
    )
    assert card is not None and card["title"] == "Opened Chrome"


