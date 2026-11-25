"""Real-Time Streaming & Multimodal Orchestration Adapters.

Inspects and configures:
- Custom Asyncio + FastAPI Bidirectional WebSocket pipeline (built-in, instant barge-in)
- Pipecat (`pipecat-ai`) WebSocket/WebRTC streaming frame pipeline (optional)
- LiveKit Agents (`livekit-agents`) WebRTC room pipeline (optional)
- Native Multimodal Live APIs (Gemini Live API / OpenAI Realtime API readiness)
- Silero VAD + Semantic Turn Detection fusion
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrchestratorInfo:
    active: str
    available: list[str]
    streaming_transport: str
    barge_in_supported: bool = True
    turn_detection: str = "Silero/Energy VAD + Semantic Endpointing"
    multimodal_live_ready: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "available": self.available,
            "streaming_transport": self.streaming_transport,
            "barge_in_supported": self.barge_in_supported,
            "turn_detection": self.turn_detection,
            "multimodal_live_ready": self.multimodal_live_ready,
        }


def inspect_orchestrators(
    requested: str = "custom_asyncio",
    gemini_configured: bool = False,
    openai_configured: bool = False,
) -> OrchestratorInfo:
    """Detect available real-time voice/multimodal orchestration frameworks."""
    avail = ["custom_asyncio"]
    has_pipecat = importlib.util.find_spec("pipecat") is not None
    has_livekit = importlib.util.find_spec("livekit") is not None
    if has_pipecat:
        avail.append("pipecat")
    if has_livekit:
        avail.append("livekit")

    live_ready = {
        "gemini_live_api": gemini_configured,
        "openai_realtime_api": openai_configured,
        "webrtc_livekit": has_livekit,
        "pipecat_pipeline": has_pipecat,
    }

    req = requested.lower().strip()
    if req == "pipecat" and has_pipecat:
        return OrchestratorInfo("pipecat", avail, "WebSocket / WebRTC (Pipecat Frame Pipeline)", True, multimodal_live_ready=live_ready)
    if req == "livekit" and has_livekit:
        return OrchestratorInfo("livekit", avail, "WebRTC (LiveKit Agents Room)", True, multimodal_live_ready=live_ready)
    return OrchestratorInfo(
        "custom_asyncio",
        avail,
        "FastAPI Bidirectional WebSocket + Asyncio EventBus",
        True,
        multimodal_live_ready=live_ready,
    )
