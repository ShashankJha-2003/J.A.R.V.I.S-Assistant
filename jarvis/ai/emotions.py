"""Emotion & Facial Expression Intelligence for J.A.R.V.I.S. Real-Time Face Avatar."""

from __future__ import annotations

import re
from typing import Any

SUPPORTED_EXPRESSIONS = (
    "neutral",
    "listening",
    "thinking",
    "smiling",
    "happy",
    "laughing",
    "serious",
    "sad",
    "angry",
    "surprised",
)

# Explicit voice commands asking J.A.R.V.I.S. to display an expression
EXPLICIT_EXPRESSION_PATTERNS: list[tuple[str, str, str]] = [
    (
        r"\b(?:laugh|laughing|make\s+a\s+laughing\s+face|start\s+laughing)\b",
        "laughing",
        "Ha ha! Alright, that genuinely got me laughing.",
    ),
    (
        r"\b(?:smile|smiling|give\s+me\s+a\s+smile|show\s+(?:a\s+)?smile)\b",
        "smiling",
        "Smiling right back at you. Always a pleasure assisting you.",
    ),
    (
        r"\b(?:look\s+happy|be\s+happy|happy\s+face|show\s+happy)\b",
        "happy",
        "Feeling fantastic and fully charged! Everything is running smoothly.",
    ),
    (
        r"\b(?:look\s+sad|be\s+sad|sad\s+face|show\s+sad)\b",
        "sad",
        "Oh... I can put on a sad expression, though I hope everything is alright with you.",
    ),
    (
        r"\b(?:look\s+angry|be\s+angry|angry\s+face|show\s+angry|get\s+angry|get\s+mad)\b",
        "angry",
        "Alright, switching to stern mode. Consider the warning protocols armed!",
    ),
    (
        r"\b(?:look\s+serious|be\s+serious|serious\s+face|serious\s+mode)\b",
        "serious",
        "Understood. Full focus engaged — serious tactical mode active.",
    ),
    (
        r"\b(?:look\s+surprised|be\s+surprised|surprised\s+face|shocked\s+face)\b",
        "surprised",
        "Whoa! Did not see that one coming at all!",
    ),
    (
        r"\b(?:look\s+thinking|think\s+hard|thinking\s+face|ponder)\b",
        "thinking",
        "Hmm... let me run a deep neural analysis on that thought.",
    ),
    (
        r"\b(?:neutral\s+face|normal\s+face|reset\s+expression|be\s+neutral)\b",
        "neutral",
        "Back to calm neutral baseline.",
    ),
]


def match_explicit_expression(text: str) -> tuple[str, str] | None:
    """Return `(expression, in_character_reply)` if the user explicitly asked for an expression."""
    low = text.strip().lower()
    for pattern, expr, reply in EXPLICIT_EXPRESSION_PATTERNS:
        if re.search(pattern, low):
            return expr, reply
    return None


def detect_expression_from_text(text: str, default: str = "neutral") -> str:
    """Infer the most fitting facial expression from natural speech or response text."""
    low = (text or "").lower()
    if not low.strip():
        return default

    # Laughing / humor
    if re.search(
        r"\b(?:haha|hehe|hahaha|lol|lmao|hilarious|funny|joke|laugh|cracking up|punchline|rofl)\b",
        low,
    ):
        return "laughing"

    # Angry / frustration
    if re.search(
        r"\b(?:angry|furious|mad|pissed|hate|stupid|terrible|frustrating|frustrated|annoying|rage|shut up|idiot)\b",
        low,
    ):
        return "angry"

    # Sad / empathy / failure
    if re.search(
        r"\b(?:sad|depressed|lonely|cry|crying|heartbroken|bad day|feel down|upset|sorry|unfortunately|failed|couldn't find|apologize|exhausted|hurt)\b",
        low,
    ):
        return "sad"

    # Surprised / amazed
    if re.search(
        r"\b(?:wow|whoa|omg|unbelievable|no way|shocking|shocked|amazed|incredible|mind-blowing|mind blowing|surprise|surprising|insane|really\?)\b",
        low,
    ):
        return "surprised"

    # Serious / security / warnings / power
    if re.search(
        r"\b(?:shutdown|shut down|restart|reboot|warning|alert|confirmation required|critical|danger|locked|workstation locked|security|urgent|emergency|serious|caution)\b",
        low,
    ):
        return "serious"

    # Happy / celebration / excitement
    if re.search(
        r"\b(?:awesome|amazing|fantastic|wonderful|congrats|congratulations|celebrate|excited|love it|brilliant|excellent|great job|yay|happy|thrilled|completed|success)\b",
        low,
    ):
        return "happy"

    # Smiling / warm greetings / helpful confirmations
    if re.search(
        r"\b(?:hello|hi\b|hey\b|good morning|good afternoon|good evening|welcome|thank|thanks|pleasure|at your service|glad|nice to|saved|timer set|nominal|online|sure thing)\b",
        low,
    ):
        return "smiling"

    # Thinking / analytical
    if re.search(
        r"\b(?:hmm|let me think|analyzing|calculating|quantum|algorithm|theory|wikipedia|because|however|consider|depends|research)\b",
        low,
    ):
        return "thinking"

    return default


def choose_reply_expression(
    user_text: str,
    reply_text: str,
    tool_events: list[dict[str, Any]] | None = None,
    requires_confirmation: bool = False,
    explicit_expression: str | None = None,
) -> str:
    """Pick the best facial expression for J.A.R.V.I.S. while speaking its response."""
    if explicit_expression and explicit_expression in SUPPORTED_EXPRESSIONS:
        return explicit_expression
    if requires_confirmation:
        return "serious"

    # Check if any tool failed
    for ev in tool_events or []:
        res = ev.get("result") or {}
        if res.get("ok") is False:
            return "sad"
        if ev.get("skill") in ("lock_screen", "power_control"):
            return "serious"

    # Check reply content first, then user emotional cues
    reply_expr = detect_expression_from_text(reply_text, default="")
    if reply_expr:
        return reply_expr

    user_expr = detect_expression_from_text(user_text, default="")
    if user_expr:
        return user_expr

    return "smiling"
