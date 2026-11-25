"""End-to-end tests for JarvisEngine, Agentic LLM Tool Loop, Safety Flow, and FastAPI + WebSocket API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from jarvis.ai.providers import BaseLLMProvider, LLMResponse, ToolCallRequest
from jarvis.config import JarvisConfig
from jarvis.core.engine import JarvisEngine
from jarvis.web.server import create_app


class MockAgentLLMProvider(BaseLLMProvider):
    """Simulates a 2-step ReAct LLM that calls a tool then synthesizes a final spoken reply."""

    name = "mock-gemini"
    model = "mock-2.5-flash"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        openai_tools: list[dict[str, Any]],
        gemini_declarations: list[dict[str, Any]],
        temperature: float = 0.4,
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="",
                tool_calls=[
                    ToolCallRequest(
                        id="call_calc_1",
                        name="calculate",
                        arguments={"expression": "19 * 23"},
                    )
                ],
                provider=self.name,
                model=self.model,
            )
        # Verify tool output reached the LLM
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert "437" in tool_msgs[0]["content"]
        return LLMResponse(
            text="Nineteen multiplied by twenty-three equals 437, Sir.",
            provider=self.name,
            model=self.model,
        )


@pytest.mark.asyncio
async def test_engine_offline_and_confirmation_flow(tmp_path: Path) -> None:
    cfg = JarvisConfig(
        llm_provider="offline",
        speak_responses=False,
        data_dir=tmp_path / "data",
        knowledge_dir=tmp_path / "knowledge",
        captures_dir=tmp_path / "captures",
    )
    engine = JarvisEngine(config=cfg, enable_tts=False)
    await engine.start(start_background_tasks=False)

    # Wake prefix stripping + offline intent
    r1 = await engine.handle_command("Hey Jarvis, calculate 100 / 4", source="test", speak=False)
    assert r1["ok"] is True
    assert "25" in r1["reply"]

    # Dangerous command triggers confirmation gate
    r2 = await engine.handle_command("shutdown my pc", source="test", speak=False)
    assert r2["requires_confirmation"] is True
    assert engine.safety.pending is not None

    # Denying via voice cancels safely without executing shutdown
    r3 = await engine.handle_command("cancel", source="test", speak=False)
    assert "Cancelled" in r3["reply"]
    assert engine.safety.pending is None

    # Remember + semantic RAG fallback
    await engine.handle_command("remember that the launch code is Orion-77", source="test", speak=False)
    r4 = await engine.handle_command("what is the launch code?", source="test", speak=False)
    assert "Orion-77" in r4["reply"]

    await engine.stop()


@pytest.mark.asyncio
async def test_engine_react_llm_tool_calling(tmp_path: Path) -> None:
    cfg = JarvisConfig(
        fast_path_routing=False,
        speak_responses=False,
        data_dir=tmp_path / "data",
        knowledge_dir=tmp_path / "knowledge",
        captures_dir=tmp_path / "captures",
    )
    mock_llm = MockAgentLLMProvider()
    engine = JarvisEngine(config=cfg, provider_override=mock_llm, enable_tts=False)
    await engine.start(start_background_tasks=False)

    res = await engine.handle_command("What is nineteen times twenty-three?", source="voice", speak=False)
    assert res["ok"] is True
    assert res["provider"] == "mock-gemini"
    assert "437" in res["reply"]
    assert len(res["tool_events"]) == 1
    assert res["tool_events"][0]["skill"] == "calculate"

    await engine.stop()


def test_fastapi_hud_rest_and_websocket(tmp_path: Path) -> None:
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
        # HUD page serves cleanly
        home = client.get("/")
        assert home.status_code == 200
        assert "J.A.R.V.I.S." in home.text

        # Status & skills endpoints
        st = client.get("/api/status").json()
        assert st["skills_count"] >= 18

        skills = client.get("/api/skills").json()["skills"]
        assert any(s["name"] == "get_system_status" for s in skills)

        # Command endpoint
        cmd = client.post("/api/command", json={"text": "calculate 7 * 9", "speak": False}).json()
        assert cmd["ok"] is True
        assert "63" in cmd["reply"]

        # Notes & Tasks CRUD via REST
        note = client.post("/api/notes", json={"content": "Test HUD note"}).json()["note"]
        assert note["id"] > 0
        assert len(client.get("/api/notes").json()["notes"]) == 1

        task = client.post("/api/tasks", json={"title": "Test HUD task"}).json()["task"]
        assert client.post(f"/api/tasks/{task['id']}/complete").json()["ok"] is True

        # WebSocket hello + ping/pong + command round-trip
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"
