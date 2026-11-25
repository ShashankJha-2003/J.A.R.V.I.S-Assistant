"""Tests for Offline NLP Intent Router, Safe Calculator, Duration Parser, and Built-in Skills."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.ai.emotions import choose_reply_expression, detect_expression_from_text
from jarvis.ai.intent_router import IntentRouter
from jarvis.config import JarvisConfig
from jarvis.core.events import EventBus
from jarvis.core.safety import SafetyGuard
from jarvis.memory.database import JarvisDatabase
from jarvis.memory.vector_store import SemanticVectorStore
from jarvis.skills.loader import create_default_registry
from jarvis.skills.productivity import parse_duration_seconds, safe_calculate
from jarvis.skills.registry import SkillContext


def test_intent_router_covers_core_voice_commands() -> None:
    router = IntentRouter()

    cases = [
        ("set volume to 65 percent", "control_volume", {"action": "set", "level": 65}),
        ("mute audio", "control_volume", {"action": "mute"}),
        ("set brightness to 70 percent", "control_brightness", {"action": "set", "level": 70}),
        ("dim screen", "control_brightness", {"action": "down"}),
        ("switch window", "press_shortcut", {"shortcut": "alt+tab"}),
        ("show desktop", "press_shortcut", {"shortcut": "win+d"}),
        ("pause music", "media_control", {"command": "play_pause"}),
        ("next track", "media_control", {"command": "next"}),
        ("take a screenshot", "take_screenshot", {}),
        ("lock my computer", "lock_screen", {}),
        ("shutdown my pc", "power_control", {"mode": "shutdown"}),
        ("what time is it", "get_time_and_date", {}),
        ("check my cpu and battery status", "get_system_status", {}),
        ("remind me in 2 minutes to stretch", "set_reminder", {"duration_seconds": 120, "message": "stretch"}),
        ("take a note: ship version 2", "manage_notes", {"action": "add", "content": "ship version 2"}),
        ("add task write unit tests", "manage_tasks", {"action": "add", "title": "write unit tests"}),
        ("complete task 3", "manage_tasks", {"action": "complete", "task_id": 3}),
        ("remember that my deploy port is 9000", "remember_fact", {"fact": "my deploy port is 9000"}),
        ("what do you remember about deploy port", "recall_memory", {"query": "deploy port"}),
        ("what is the weather in Tokyo", "get_weather", {"city": "tokyo"}),
        ("give me the latest tech news", "get_top_news", {"category": "tech"}),
        ("calculate sqrt(144) + 25", "calculate", {"expression": "sqrt(144) + 25"}),
        ("open notepad", "open_application", {"app_name": "notepad"}),
        ("open github", "web_search_and_open", {"query": "github", "platform": "url"}),
        ("open google", "web_search_and_open", {"query": "google", "platform": "url"}),
        ("open youtube", "web_search_and_open", {"query": "youtube", "platform": "url"}),
        ("open instagram", "web_search_and_open", {"query": "instagram", "platform": "url"}),
        ("go to chatgpt", "web_search_and_open", {"query": "chatgpt", "platform": "url"}),
        ("play lofi beats on youtube", "web_search_and_open", {"query": "lofi beats", "platform": "youtube"}),
        ("list open windows", "manage_window", {"action": "list"}),
        ("check git status", "git_status", {}),
        ("tell about MS Dhoni", "search_wikipedia", {"topic": "MS Dhoni"}),
        ("explain quantum computing", "search_wikipedia", {"topic": "quantum computing"}),
        ("define serendipity", "search_wikipedia", {"topic": "serendipity"}),
    ]

    for phrase, expected_skill, expected_args in cases:
        match = router.match(phrase)
        assert match is not None, f"Failed to match: {phrase}"
        assert match.skill_name == expected_skill, f"Wrong skill for '{phrase}': {match.skill_name}"
        for k, v in expected_args.items():
            assert match.arguments.get(k) == v, f"Mismatch for '{phrase}' arg '{k}': {match.arguments}"


def test_safe_calculator_and_duration_parser() -> None:
    assert safe_calculate("sqrt(144) + 25") == 37.0
    assert safe_calculate("12 times 8 minus 6") == 90.0
    assert round(safe_calculate("2 ^ 10"), 2) == 1024.0
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('echo hi')")

    assert parse_duration_seconds("45 seconds") == 45
    assert parse_duration_seconds("2 minutes 30 seconds") == 150
    assert parse_duration_seconds("1 hour 5 min") == 3900


@pytest.mark.asyncio
async def test_builtin_skills_execute_cleanly(tmp_path: Path) -> None:
    cfg = JarvisConfig(data_dir=tmp_path / "data", knowledge_dir=tmp_path / "kb", captures_dir=tmp_path / "cap")
    cfg.ensure_directories()
    db = JarvisDatabase(cfg.db_path)
    vs = SemanticVectorStore(db)
    ctx = SkillContext(cfg, db, vs, EventBus(), SafetyGuard(True))
    reg = create_default_registry()

    assert len(reg.list_skills()) >= 18

    r_time = await reg.execute("get_time_and_date", {}, ctx)
    assert r_time.ok and "time" in r_time.data

    r_sys = await reg.execute("get_system_status", {}, ctx)
    assert r_sys.ok and "ram_total_gb" in r_sys.data

    r_calc = await reg.execute("calculate", {"expression": "15 * 4"}, ctx)
    assert r_calc.ok and r_calc.data["formatted"] == "60"

    r_rem = await reg.execute("remember_fact", {"fact": "Project codename is StarkNet"}, ctx)
    assert r_rem.ok
    r_rec = await reg.execute("recall_memory", {"query": "codename"}, ctx)
    assert r_rec.ok and "StarkNet" in r_rec.message

    db.close()


def test_emotion_and_expression_intelligence() -> None:
    router = IntentRouter()

    m_joke = router.match("tell me a joke")
    assert m_joke is not None and m_joke.expression == "laughing"

    m_sad = router.match("I had a bad day")
    assert m_sad is not None and m_sad.expression == "sad"

    m_angry = router.match("look angry")
    assert m_angry is not None and m_angry.expression == "angry"

    m_serious = router.match("be serious")
    assert m_serious is not None and m_serious.expression == "serious"

    m_surprised = router.match("surprise me")
    assert m_surprised is not None and m_surprised.expression == "surprised"

    m_greet = router.match("hey there")
    assert m_greet is not None and m_greet.expression == "smiling" and m_greet.direct_reply

    assert detect_expression_from_text("Hahaha that is hilarious!") == "laughing"
    assert detect_expression_from_text("This is so frustrating and annoying") == "angry"
    assert choose_reply_expression("shutdown my pc", "Confirmation required", requires_confirmation=True) == "serious"

