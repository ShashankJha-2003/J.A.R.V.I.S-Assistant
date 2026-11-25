"""Supervisor Pattern Multi-Agent Router & Persona / Tone Modulation Engine.

Routes requests to focused domain sub-agents (System/OS, Research/Web/Vision,
Code/Workspace, Memory/Productivity) and modulates persona, tone, and TTS voice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

InteractionMode = Literal["chat", "search", "device"]

_CASUAL_PHRASE_RE = re.compile(
    r"\b(?:what(?:'s|\s+is)\s+up|whats\s+up|what\s+are\s+you\s+doing|what\s+do\s+you\s+think|"
    r"what(?:'s|\s+is)\s+on\s+your\s+mind|what\s+should\s+i|what(?:'s|\s+is)\s+going\s+on|"
    r"how\s+are\s+you|how(?:'s|\s+is)\s+it\s+going|tell\s+me\s+a\s+joke|good\s+news|bad\s+news|"
    r"great\s+news|no\s+news|long\s+day|thinking\s+about|in\s+the\s+mood|let'?s\s+chill|"
    r"just\s+talking|want\s+to\s+chat|talk\s+to\s+me)\b",
    re.IGNORECASE,
)

_DEVICE_MODE_RE = re.compile(
    r"\b(?:"
    r"volume|mute|unmute|louder|quieter|brightness|brighten|dim\s+(?:the\s+)?(?:screen|room|desk|lights?|lamp)|"
    r"screenshot|screen\s*shot|lock\s+(?:my\s+|the\s+)?(?:screen|pc|computer|workstation|system)|"
    r"shut\s*down|power\s+off|restart|reboot|sleep\s+mode|put\s+.*?\s+to\s+sleep|"
    r"system\s+status|telemetry|cpu|ram|memory\s+usage|battery|disk\s+space|diagnostics|"
    r"clipboard|shortcut|alt\s*tab|show\s+desktop|new\s+tab|close\s+tab|reopen\s+tab|"
    r"pause\s+(?:the\s+)?(?:music|media|song|video)|resume\s+(?:the\s+)?(?:music|media|song|video)|"
    r"next\s+(?:song|track|music|video)|previous\s+(?:song|track|music)|"
    r"list\s+(?:all\s+|open\s+)?windows|focus\s+.*?\s+window|minimize|maximize|switch\s+window|"
    r"(?:open|launch)\s+\w+|(?:go\s+to|visit|browse)\s+\w+|play\s+.+?\s+on\s+youtube|"
    r"(?:turn|switch)\s+(?:on|off)\s+(?:the\s+)?\w+|smart\s+home|smart\s+devices?|home\s+assistant|iot|mqtt|esp32|"
    r"git\s+(?:status|branch|branches|log|diff|switch)|switch\s+(?:git\s+)?branch|checkout\s+branch|"
    r"pytest|run\s+(?:project\s+|unit\s+)?tests?|index\s+(?:my\s+)?(?:codebase|knowledge|docs|documents)|"
    r"run\s+shell|execute\s+shell|delete\s+(?:workspace\s+)?file|kill\s+process|terminate\s+process|"
    r"(?:find|search)\s+(?:workspace\s+)?files?|mcp|"
    r"remind\s+me|set\s+(?:a\s+)?timer|my\s+reminders|my\s+timers|"
    r"(?:take|create|add|save|make|show|list|read|delete)\s+(?:a\s+|my\s+)?notes?|"
    r"(?:add|create|show|list|read|complete|finish|delete)\s+(?:a\s+|my\s+)?(?:tasks?|to-?do)|"
    r"remember\s+(?:that\s+)?|what\s+do\s+you\s+remember|recall\s+|search\s+my\s+(?:memory|knowledge|docs|rag)|"
    r"what\s+(?:time|day|date)\s+is\s+it|what(?:'s|\s+is)\s+(?:the\s+)?(?:time|date|day)|current\s+time|today'?s\s+date|"
    r"calculate|compute|evaluate|\b\d+\s*(?:\+|\-|\*|\/|\^|plus|minus|times|multiplied|divided)\s*\d+|"
    r"\b(?:times|multiplied\s+by|divided\s+by|plus|minus|sqrt)\b|"
    r"on\s+my\s+screen|see\s+my\s+screen|analyze\s+(?:my\s+)?screen|inspect\s+(?:my\s+)?screen|read\s+(?:my\s+)?screen|"
    r"summarize\s+what(?:'s|\s+is)\s+open|webcam|camera|"
    r"(?:switch|set)\s+(?:persona|tone|mode)\s+to\s+(?:calm|crisp|witty|analytical)|"
    r"^(?:calm|crisp|witty|analytical)\s+(?:mode|persona|tone)|"
    r"(?:type|dictate)\s+\w+"
    r")\b",
    re.IGNORECASE,
)

_SEARCH_LIVE_RE = re.compile(
    r"\b(?:"
    r"weather|temperature|forecast|"
    r"(?:latest|top|today'?s|breaking|tech|world|science)\s+news|\bheadlines\b|"
    r"(?:read|get|show|check|give\s+me)\s+(?:the\s+)?news|"
    r"stock\s+price|share\s+price|price\s+of|bitcoin\s+price|crypto\s+price|gold\s+price|exchange\s+rate|"
    r"live\s+score|match\s+score|cricket\s+score|football\s+score|"
    r"search\s+(?:google|web|internet|online|github|youtube)\b|search\s+for\b|\bgoogle\s+\w+|"
    r"look\s+up\b|\blookup\b|\bwikipedia\b|\bwiki\s+\w+|https?://|summarize\s+(?:https?://|www\.)"
    r")\b",
    re.IGNORECASE,
)

_SEARCH_QUESTION_RE = re.compile(
    r"^(?:can\s+you\s+|could\s+you\s+|please\s+)?"
    r"(?:"
    r"who\s+(?:is|was|are|were|invented|created|discovered|won|founded)\s+\w+|"
    r"what\s+(?:is|was|are|were)\s+(?:a\s+|an\s+|the\s+)?\w+|"
    r"what's\s+(?:the\s+)?(?:capital|population|meaning|definition|distance|height|speed|difference|weather|temperature|launch\s+code|latest|current)\b|"
    r"tell\s+(?:me\s+)?about\s+(?!yourself\b|you\b|my\s+day\b|your\s+day\b)\w+|"
    r"(?:give\s+me\s+)?(?:info|information|details|facts)\s+(?:about|on)\s+\w+|"
    r"what\s+do\s+you\s+know\s+about\s+\w+|"
    r"explain(?:\s+to\s+me)?\s+\w+|describe\s+\w+|define\s+\w+|"
    r"(?:what(?:'s|\s+is)\s+the\s+)?(?:meaning|definition)\s+of\s+\w+|"
    r"where\s+(?:is|are|was)\s+\w+|when\s+(?:was|did)\s+\w+|"
    r"how\s+(?:many|far|tall|old|fast|long|much\s+does)\s+\w+|how\s+(?:does|do)\s+.+?\s+work\b"
    r")",
    re.IGNORECASE,
)


def classify_interaction_mode(utterance: str) -> InteractionMode:
    """Explicitly classify a user turn into 'chat' | 'search' | 'device'.

    Prevents casual conversation from ever receiving or triggering web search
    or OS tools, while routing explicit factual lookups to 'search' and
    system/dev/productivity/vision/IoT actions to 'device'.
    """
    raw = (utterance or "").strip()
    if not raw:
        return "chat"
    low = raw.lower().strip(" .!?")

    # 1. Explicit device / OS / code / vision / IoT / memory / productivity / math actions
    if _DEVICE_MODE_RE.search(low):
        return "device"

    # 2. Explicit live info / external search requests (weather, news, prices, search, URLs)
    if _SEARCH_LIVE_RE.search(low) and not re.search(r"\b(?:good|bad|great|awesome|no)\s+news\b", low):
        return "search"

    # 3. Casual banter / opinion / feeling statements stay strictly in CHAT mode
    if _CASUAL_PHRASE_RE.search(low):
        return "chat"

    # 4. Explicit factual / encyclopedic question starters -> SEARCH mode
    if _SEARCH_QUESTION_RE.search(raw):
        # Guard against casual personal questions ("what is your favorite...", "what is wrong with you")
        if re.search(r"\b(?:your\s+(?:favorite|name|opinion|mood|day)|wrong\s+with\s+you|up\s+with\s+you)\b", low):
            return "chat"
        return "search"

    # 5. Default to natural CHAT mode (never guess a search on everyday conversation!)
    return "chat"


@dataclass(frozen=True)
class PersonaProfile:
    name: str
    label: str
    description: str
    system_style: str
    edge_voice: str
    tts_rate_delta: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "edge_voice": self.edge_voice,
        }


PERSONA_PROFILES: dict[str, PersonaProfile] = {
    "calm": PersonaProfile(
        name="calm",
        label="Calm & Natural",
        description="Warm, poised, natural conversational assistant.",
        system_style="Speak in a calm, warm, reassuring, and natural human tone (1-3 crisp sentences).",
        edge_voice="en-GB-RyanNeural",
    ),
    "crisp": PersonaProfile(
        name="crisp",
        label="Crisp Executive",
        description="Direct, zero-fluff, ultra-fast executive brevity.",
        system_style="Be ultra-concise, direct, and efficient. Lead immediately with the answer or action in 1-2 short sentences.",
        edge_voice="en-GB-RyanNeural",
        tts_rate_delta=15,
    ),
    "witty": PersonaProfile(
        name="witty",
        label="Witty Stark-Style",
        description="Sharp, charming British wit with helpful precision.",
        system_style="Respond with subtle, charming Stark-style dry wit while remaining accurate and concise (1-3 sentences).",
        edge_voice="en-GB-RyanNeural",
    ),
    "analytical": PersonaProfile(
        name="analytical",
        label="Analytical Engineer",
        description="Structured, technical, root-cause engineering clarity.",
        system_style="Adopt a precise, analytical engineering tone: highlight root causes, concrete metrics, and exact next steps clearly.",
        edge_voice="en-US-AndrewNeural",
    ),
}


@dataclass(frozen=True)
class SubAgentSpec:
    name: str
    title: str
    categories: tuple[str, ...]
    system_focus: str
    keywords: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "categories": list(self.categories),
        }


SUB_AGENTS: dict[str, SubAgentSpec] = {
    "system_os_agent": SubAgentSpec(
        name="system_os_agent",
        title="System, OS & Smart Home Agent",
        categories=("system", "automation", "iot"),
        system_focus=(
            "You are acting as the System/OS & Smart Home Sub-Agent. Focus on laptop control, "
            "window/app automation, audio/brightness, hardware telemetry, smart home lights/plugs/MQTT, "
            "and strictly obey human-in-the-loop safety confirmation for destructive actions."
        ),
        keywords=(
            "volume", "mute", "brightness", "screen", "screenshot", "lock", "shutdown", "restart",
            "sleep", "battery", "cpu", "ram", "disk", "clipboard", "open", "launch", "window",
            "minimize", "maximize", "close", "music", "pause", "next track", "shortcut", "type",
            "light", "lights", "lamp", "plug", "fan", "thermostat", "ac", "smart home", "home assistant",
            "iot", "mqtt", "esp32", "kill process",
        ),
    ),
    "code_workspace_agent": SubAgentSpec(
        name="code_workspace_agent",
        title="Code & Workspace Workflow Agent",
        categories=("developer", "automation", "mcp"),
        system_focus=(
            "You are acting as the Code & Workspace Sub-Agent. Focus on developer workflows: "
            "running test suites and summarizing failures cleanly without dumping raw logs, "
            "managing Git branches/status, searching workspace files, setting up project sessions, "
            "and requiring explicit confirmation before running shell commands or deleting files."
        ),
        keywords=(
            "git", "branch", "commit", "diff", "pytest", "test", "tests", "unit test", "workspace",
            "repo", "repository", "file", "files", "code", "vs code", "vscode", "dev server",
            "coding", "project", "shell", "terminal", "command", "delete file", "mcp",
        ),
    ),
    "research_web_agent": SubAgentSpec(
        name="research_web_agent",
        title="Research, Web & Vision Agent",
        categories=("web", "vision", "mcp"),
        system_focus=(
            "You are acting as the Research, Web & Vision Sub-Agent. Focus on live factual lookup, "
            "weather, news, summarizing URLs, and inspecting screen or camera captures to explain "
            "errors or active content clearly."
        ),
        keywords=(
            "weather", "temperature", "forecast", "news", "headline", "wikipedia", "wiki",
            "google", "youtube", "github", "search", "who is", "what is", "tell about", "explain",
            "define", "summarize", "url", "http", "screen", "what's on my screen", "error on my screen",
            "camera", "webcam", "see", "look at", "vision",
        ),
    ),
    "memory_productivity_agent": SubAgentSpec(
        name="memory_productivity_agent",
        title="Memory, RAG & Productivity Agent",
        categories=("memory", "productivity", "mcp"),
        system_focus=(
            "You are acting as the Memory, RAG & Productivity Sub-Agent. Focus on semantic vector memory, "
            "local document/PDF/notes RAG retrieval, reminders, timers, notes, tasks, and calculations."
        ),
        keywords=(
            "remember", "recall", "memory", "knowledge", "rag", "document", "pdf", "index",
            "note", "notes", "task", "tasks", "todo", "to-do", "remind", "reminder", "timer",
            "time", "date", "calculate", "compute", "solve", "math",
        ),
    ),
}


def get_persona(name: str) -> PersonaProfile:
    return PERSONA_PROFILES.get((name or "calm").strip().lower(), PERSONA_PROFILES["calm"])


class AgentSupervisor:
    """Routes commands to the best specialized sub-agent (Supervisor Pattern) and interaction mode."""

    def classify_mode(self, utterance: str) -> InteractionMode:
        return classify_interaction_mode(utterance)

    def route(self, utterance: str) -> SubAgentSpec:
        low = (utterance or "").lower().strip()
        if not low:
            return SUB_AGENTS["memory_productivity_agent"]

        # High-priority direct vision routing ("what's on my screen", "look at camera")
        if re.search(r"\b(?:on\s+my\s+screen|see\s+my\s+screen|error\s+on\s+screen|what'?s\s+open\s+right\s+now|webcam|camera)\b", low):
            return SUB_AGENTS["research_web_agent"]

        # High-priority coding / git / test / shell routing
        if re.search(r"\b(?:pytest|run\s+tests?|git\s+\w+|switch\s+branch|coding\s+workspace|dev\s+workflow|run\s+shell|delete\s+file|mcp)\b", low):
            return SUB_AGENTS["code_workspace_agent"]

        scores: dict[str, int] = {k: 0 for k in SUB_AGENTS}
        for key, spec in SUB_AGENTS.items():
            for kw in spec.keywords:
                if kw in low:
                    scores[key] += 2 if " " in kw else 1

        best_key = max(scores, key=lambda k: scores[k])
        if scores[best_key] > 0:
            return SUB_AGENTS[best_key]
        return SUB_AGENTS["research_web_agent"]

    def list_agents(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in SUB_AGENTS.values()]
