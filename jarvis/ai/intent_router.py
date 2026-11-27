"""Offline Natural-Language Intent Router & Fast-Path Reflex Engine.

Parses natural voice commands directly into registered skill invocations with
sub-millisecond latency. Serves two purposes:
1. Fast-Path Reflexes: Instant execution for hardware/media commands even when using a cloud LLM.
2. Full Offline Brain: Enables 30+ natural voice commands with ZERO API keys or internet LLMs.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

from jarvis.ai.emotions import match_explicit_expression
from jarvis.skills.app_automation import APP_CATALOG
from jarvis.skills.productivity import parse_duration_seconds
from jarvis.skills.web_intelligence import SITE_SHORTCUTS


JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs! Ha ha!",
    "Why did the neural network go to therapy? It had too many deep issues and kept overfitting its past! Ha ha!",
    "I told my Python script a joke about recursion, and it's still laughing... and laughing... and laughing!",
    "Why did the AI break up with the database? There was no real connection — just too many unindexed arguments! Ha ha!",
]

FUN_FACTS = [
    "Whoa, here is a wild fact: a single bolt of lightning carries enough energy to toast about 100,000 slices of bread!",
    "Incredible fact: octopuses have three hearts, blue blood, and nine brains — one central brain and one in each arm!",
    "Mind-blowing fact: there are more possible games of chess than there are atoms in the observable universe!",
]


@dataclass
class IntentMatch:
    skill_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    fast_path: bool = False
    direct_reply: str | None = None
    expression: str | None = None
    confidence: float = 0.95


def strip_wake_prefix(text: str, wake_words: list[str]) -> str:
    """Remove leading wake word ('hey jarvis, ...') from a voice command."""
    cleaned = text.strip()
    lower = cleaned.lower()
    for word in sorted(wake_words, key=len, reverse=True):
        w = word.lower().strip()
        if lower == w:
            return ""
        if lower.startswith(w) and len(lower) > len(w) and lower[len(w)] in " ,.:!?":
            return cleaned[len(w) :].lstrip(" ,.:!?")
    return cleaned


class IntentRouter:
    """Deterministic + NLP pattern router for voice commands."""

    def match(self, utterance: str, fast_path_only: bool = False) -> IntentMatch | None:
        raw = utterance.strip()
        if not raw:
            return None
        low = raw.lower().strip(" .!?")

        # 1. Fast-path hardware / reflex commands (always sub-50ms)
        reflex = self._match_fast_path(raw, low)
        if reflex is not None:
            return reflex
        if fast_path_only:
            return None

        # 2. Full offline natural-language command parser
        return self._match_general(raw, low)

    def _match_fast_path(self, raw: str, low: str) -> IntentMatch | None:
        # Volume controls
        m = re.search(r"\b(?:set|change|adjust)\s+(?:the\s+)?volume\s+(?:to\s+)?(\d{1,3})\s*%?", low)
        if m:
            return IntentMatch("control_volume", {"action": "set", "level": int(m.group(1))}, fast_path=True)
        if re.search(r"\b(unmute)\b", low):
            return IntentMatch("control_volume", {"action": "unmute"}, fast_path=True)
        if re.search(r"\b(mute)\b", low) and "mic" not in low:
            return IntentMatch("control_volume", {"action": "mute"}, fast_path=True)
        if re.search(r"\b(?:volume\s+up|turn\s+(?:up\s+)?(?:the\s+)?volume|louder|increase\s+volume)\b", low):
            return IntentMatch("control_volume", {"action": "up"}, fast_path=True)
        if re.search(r"\b(?:volume\s+down|turn\s+(?:down\s+)?(?:the\s+)?volume|quieter|decrease\s+volume|lower\s+volume)\b", low):
            return IntentMatch("control_volume", {"action": "down"}, fast_path=True)

        # Screen brightness controls
        m = re.search(r"\b(?:set|change|adjust)\s+(?:the\s+)?brightness\s+(?:to\s+)?(\d{1,3})\s*%?", low)
        if m:
            return IntentMatch("control_brightness", {"action": "set", "level": int(m.group(1))}, fast_path=True)
        if re.search(r"\b(?:brighten|increase\s+brightness|brightness\s+up|turn\s+up\s+brightness)\b", low):
            return IntentMatch("control_brightness", {"action": "up"}, fast_path=True)
        if re.search(r"\b(?:dim\s+(?:the\s+)?screen|decrease\s+brightness|lower\s+brightness|brightness\s+down)\b", low):
            return IntentMatch("control_brightness", {"action": "down"}, fast_path=True)

        # Safe allowlisted keyboard shortcuts
        m = re.match(r"^(?:press|hit|shortcut)\s+(.+)$", low)
        if m:
            return IntentMatch("press_shortcut", {"shortcut": m.group(1).strip()}, fast_path=True)
        if low in {"switch window", "alt tab", "alt+tab"}:
            return IntentMatch("press_shortcut", {"shortcut": "alt+tab"}, fast_path=True)
        if low in {"show desktop", "minimize everything", "hide windows"}:
            return IntentMatch("press_shortcut", {"shortcut": "win+d"}, fast_path=True)
        if low in {"new tab", "open new tab"}:
            return IntentMatch("press_shortcut", {"shortcut": "ctrl+t"}, fast_path=True)
        if low in {"close tab", "close this tab"}:
            return IntentMatch("press_shortcut", {"shortcut": "ctrl+w"}, fast_path=True)
        if low in {"reopen tab", "reopen closed tab"}:
            return IntentMatch("press_shortcut", {"shortcut": "ctrl+shift+t"}, fast_path=True)

        # Media playback
        if re.search(r"\b(?:pause|resume|play/pause)\s*(?:the\s+)?(?:music|media|song|video|playback)?$", low):
            return IntentMatch("media_control", {"command": "play_pause"}, fast_path=True)
        if re.search(r"\b(?:next|skip)\s+(?:song|track|music|video)\b", low):
            return IntentMatch("media_control", {"command": "next"}, fast_path=True)
        if re.search(r"\b(?:previous|prev|last)\s+(?:song|track|music)\b", low):
            return IntentMatch("media_control", {"command": "previous"}, fast_path=True)

        # Screenshot
        if re.search(r"\b(?:take|capture|grab)\s+(?:a\s+)?(?:screenshot|screen\s*shot|screen)\b", low):
            return IntentMatch("take_screenshot", {}, fast_path=True)

        # Lock workstation
        if re.search(r"\block\s+(?:my\s+|the\s+)?(?:screen|computer|pc|workstation|system)\b", low):
            return IntentMatch("lock_screen", {}, fast_path=True)

        # Power management (always gated by SafetyGuard confirmation!)
        if re.search(r"\b(?:shut\s*down|power\s+off)\s+(?:my\s+|the\s+)?(?:pc|computer|system|machine)\b", low):
            return IntentMatch("power_control", {"mode": "shutdown"}, fast_path=True)
        if re.search(r"\b(?:restart|reboot)\s+(?:my\s+|the\s+)?(?:pc|computer|system|machine)\b", low):
            return IntentMatch("power_control", {"mode": "restart"}, fast_path=True)
        if re.search(r"\b(?:put\s+(?:my\s+|the\s+)?(?:pc|computer|system)\s+to\s+sleep|sleep\s+mode)\b", low):
            return IntentMatch("power_control", {"mode": "sleep"}, fast_path=True)

        return None

    def _match_general(self, raw: str, low: str) -> IntentMatch | None:
        # Explicit facial expression triggers ("smile", "laugh", "look angry", etc.)
        expr_hit = match_explicit_expression(low)
        if expr_hit is not None:
            expr, reply = expr_hit
            return IntentMatch("", direct_reply=reply, expression=expr)

        # Conversational entrance greetings, small talk, humor, empathy, and emotional reactions
        if low in {
            "hi", "hello", "hey", "hey there", "hi there", "hello there", "yo", "namaste",
            "hello jarvis", "hi jarvis", "hey jarvis", "wake up", "are you there", "you there",
        } or re.match(
            r"^(?:hi|hello|hey|yo|namaste)(?:\s+(?:there|jarvis|buddy|bud|man|bro|mate|friend|pal))+$",
            low,
        ):
            return IntentMatch(
                "",
                direct_reply=random.choice([
                    "Hey there! How's it going? What can I help you with today?",
                    "Hey! Good to see you. What's on your mind?",
                    "Hey there! I'm all ears — talk or type whenever you're ready.",
                    "Hi! Ready to help with anything you need.",
                ]),
                expression="smiling",
            )
        if low in {"good morning", "morning"}:
            return IntentMatch(
                "",
                direct_reply=random.choice([
                    "Good morning! Hope you're off to a great start today. How can I help?",
                    "Hey, good morning! Ready to tackle the day together?",
                ]),
                expression="smiling",
            )
        if low in {"good afternoon", "afternoon"}:
            return IntentMatch(
                "",
                direct_reply="Good afternoon! How's your day going so far?",
                expression="smiling",
            )
        if low in {"good evening", "evening"}:
            return IntentMatch(
                "",
                direct_reply="Good evening! How was your day? Let me know if you need anything.",
                expression="smiling",
            )
        if re.search(r"\b(?:how are you|how's it going|how are things|how do you feel|how've you been)\b", low):
            return IntentMatch(
                "",
                direct_reply=random.choice([
                    "I'm doing great, thanks for asking! How are you doing today?",
                    "Feeling sharp and ready to help! How's everything on your end?",
                    "I'm doing awesome — glad you checked in. What are you up to?",
                ]),
                expression="happy",
            )
        if re.search(r"\b(?:what are you doing|whatcha doing|what you doing)\b", low):
            return IntentMatch(
                "",
                direct_reply=random.choice([
                    "Just hanging out here on standby, waiting to help you out! What's up?",
                    "Keeping an eye on things and staying ready for your next idea!",
                ]),
                expression="smiling",
            )
        if re.search(r"\b(?:nice to meet you|pleased to meet you|glad to meet you)\b", low):
            return IntentMatch(
                "",
                direct_reply="Nice to meet you too! Looking forward to working together.",
                expression="happy",
            )
        if re.search(r"\b(?:you look good|looking good|you look great|you look nice|cool avatar)\b", low):
            return IntentMatch(
                "",
                direct_reply="Haha, thanks! Gotta keep the studio lighting looking sharp.",
                expression="happy",
            )
        if re.search(r"\b(?:i am bored|i'm bored|so bored|feeling bored)\b", low):
            return IntentMatch(
                "",
                direct_reply="Let's fix that! I can tell you a joke, share a wild fact, open YouTube, or chat about any topic you like.",
                expression="smiling",
            )
        if re.search(r"\b(?:i am tired|i'm tired|so tired|exhausted|sleepy)\b", low):
            return IntentMatch(
                "",
                direct_reply="Sounds like you've been working hard. Take a breather and grab some water — I'll handle the busywork.",
                expression="sad",
            )
        if re.search(r"\b(?:i am happy|i'm happy|feeling great|having a great day|good news)\b", low):
            return IntentMatch(
                "",
                direct_reply="That's awesome to hear! Love that energy — tell me all about it!",
                expression="happy",
            )
        if re.search(r"\b(?:who made you|who created you|who built you)\b", low):
            return IntentMatch(
                "",
                direct_reply="I was built right here in Python as your personal J.A.R.V.I.S. AI assistant — customized with your own voice interface and avatar!",
                expression="smiling",
            )
        if low in {"ok", "okay", "cool", "nice", "great", "alright", "got it", "sounds good", "fine"}:
            return IntentMatch(
                "",
                direct_reply=random.choice([
                    "Sounds good! Let me know whenever you need anything else.",
                    "Got it — I'm right here if you need me.",
                ]),
                expression="smiling",
            )
        if low in {"sorry", "my bad", "never mind", "nevermind", "nothing", "forget it"}:
            return IntentMatch(
                "",
                direct_reply="No worries at all! Just give me a shout whenever you want.",
                expression="smiling",
            )
        if re.search(r"\b(?:tell me a joke|make me laugh|say something funny|another joke)\b", low):
            return IntentMatch("", direct_reply=random.choice(JOKES), expression="laughing")
        if re.search(r"\b(?:surprise me|fun fact|amazing fact|mind blowing fact)\b", low):
            return IntentMatch("", direct_reply=random.choice(FUN_FACTS), expression="surprised")
        if re.search(r"\b(?:i am sad|i'm sad|feel sad|feeling down|had a bad day|i feel lonely|i'm upset)\b", low):
            return IntentMatch(
                "",
                direct_reply="I'm really sorry you're feeling that way. Take a deep breath — I'm right here with you. Want to talk about it, or hear some music?",
                expression="sad",
            )
        if re.search(r"\b(?:i am angry|i'm angry|i am mad|so frustrated|this is annoying|i hate this)\b", low):
            return IntentMatch(
                "",
                direct_reply="I can hear how frustrating that is. Let's focus and solve whatever is getting in your way right now.",
                expression="angry",
            )
        if re.search(r"\b(?:thank you|thanks|you're awesome|good job|well done|love you)\b", low):
            return IntentMatch(
                "",
                direct_reply="You're very welcome! Always glad to make your day easier.",
                expression="happy",
            )
        if low in {"bye", "goodbye", "see you", "see you later", "good night", "goodnight"}:
            return IntentMatch(
                "",
                direct_reply="Goodbye! I'll be right here on standby whenever you need me.",
                expression="smiling",
            )
        if re.search(r"\b(?:what's up|whats up|wassup|sup|what's going on|whats going on|what is going on)\b", low):
            return IntentMatch(
                "",
                direct_reply=random.choice([
                    "Not much, just keeping everything running smoothly! What's going on with you?",
                    "All good over here — just chilling on standby. What are we up to?",
                    "Just hanging out and ready whenever you need me! How's your day going?",
                ]),
                expression="smiling",
            )
        if re.search(r"\b(?:flip a coin|toss a coin|heads or tails)\b", low):
            side = random.choice(["Heads", "Tails"])
            return IntentMatch("", direct_reply=f"I flipped a coin — it landed on {side}!", expression="happy")
        if re.search(r"\b(?:roll a dice|roll a die|roll dice)\b", low):
            val = random.randint(1, 6)
            return IntentMatch("", direct_reply=f"You rolled a {val}!", expression="happy")
        if re.search(r"\b(?:motivate me|inspire me|motivational quote|give me a quote)\b", low):
            quote = random.choice([
                "Success is the sum of small efforts repeated day in and day out. Keep building!",
                "The best way to predict the future is to create it — one bold step at a time.",
                "Discipline beats motivation when the road gets steep. You've got this!",
            ])
            return IntentMatch("", direct_reply=quote, expression="smiling")
        if re.search(r"\b(?:who are you|what is your name|introduce yourself)\b", low):
            return IntentMatch(
                "",
                direct_reply=(
                    "I am J.A.R.V.I.S., your Python AI voice control assistant — integrating real-time "
                    "voice recognition, neural speech synthesis, multi-LLM tool calling, semantic RAG memory, "
                    "and live expressive avatar interaction."
                ),
                expression="smiling",
            )
        if re.search(r"\b(?:what can you do|your capabilities|list skills|help me|^help$)\b", low):
            return IntentMatch("jarvis_capabilities", {})

        if re.search(r"^(?:क्या\s+हाल\s+है|क्या\s+हाल|कैसे\s+हो|नमस्ते|नमस्कार|और\s+भाई|सब\s+बढ़िया)$", raw.strip(" .!?।")):
            return IntentMatch(
                "",
                direct_reply=random.choice([
                    "सब बढ़िया है भाई! आप बताओ, आज क्या चल रहा है?",
                    "एकदम मस्त हूँ यार! बोलो, आज क्या प्लान है?",
                    "नमस्ते भाई! सब फर्स्ट क्लास है, आपका दिन कैसा चल रहा है?",
                ]),
                expression="smiling",
            )
        if re.search(r"^(?:kya\s+haal\s+hai|kya\s+haal|kaise\s+ho|aur\s+bhai|sab\s+badhiya|kya\s+chal\s+raha\s+hai)(?:\s+(?:bhai|yaar|bro|dost))?$", low):
            return IntentMatch(
                "",
                direct_reply=random.choice([
                    "Sab badhiya hai bhai! Tum batao, aaj kya chal raha hai?",
                    "Ekdum mast hoon yaar! Bolo, aaj ka kya plan hai?",
                ]),
                expression="smiling",
            )

        # Time & Date (English + Hindi/Hinglish)
        if re.search(
            r"\b(?:what\s+(?:time|day|date)\s+is\s+it|what(?:'s|\s+is)\s+(?:the\s+)?(?:time|date|day)|"
            r"current\s+time|today'?s\s+date|tell\s+me\s+the\s+(?:time|date)|"
            r"kitne\s+baje|time\s+kya\s+hua|samay\s+kya\s+hua)\b|(?:कितने\s+बजे|समय\s+क्या\s+हुआ)",
            low,
        ):
            return IntentMatch("get_time_and_date", {})

        # System Telemetry
        if re.search(r"\b(?:system\s+status|telemetry|cpu|ram|memory\s+usage|battery|disk\s+space|diagnostics|health\s+check)\b", low):
            return IntentMatch("get_system_status", {})

        # Clipboard
        if re.search(r"\b(?:read|what'?s\s+(?:in|on)|show)\s+(?:my\s+)?clipboard\b", low):
            return IntentMatch("clipboard_action", {"action": "read"})
        m = re.match(r"^(?:copy)\s+(.+?)\s+to\s+(?:my\s+)?clipboard$", raw, re.IGNORECASE)
        if m:
            return IntentMatch("clipboard_action", {"action": "copy", "text": m.group(1).strip()})

        # Timers & Reminders
        if re.search(r"\b(?:list|show|what(?:'s| are))\s+(?:my\s+)?(?:active\s+)?(?:reminders|timers)\b", low):
            return IntentMatch("list_reminders", {})
        m = re.search(r"\bremind\s+me\s+in\s+(.+?)\s+(?:to|that)\s+(.+)$", raw, re.IGNORECASE)
        if m:
            secs = parse_duration_seconds(m.group(1))
            if secs > 0:
                return IntentMatch("set_reminder", {"duration_seconds": secs, "message": m.group(2).strip()})
        m = re.search(r"\bremind\s+me\s+(?:to\s+)?(.+?)\s+in\s+(.+)$", raw, re.IGNORECASE)
        if m:
            secs = parse_duration_seconds(m.group(2))
            if secs > 0:
                return IntentMatch("set_reminder", {"duration_seconds": secs, "message": m.group(1).strip()})
        m = re.search(r"\b(?:set\s+(?:a\s+)?timer|timer)\s+(?:for\s+)?(.+?)(?:\s+(?:for|called|named)\s+(.+))?$", raw, re.IGNORECASE)
        if m:
            secs = parse_duration_seconds(m.group(1))
            if secs > 0:
                label = (m.group(2) or "Countdown timer complete").strip()
                return IntentMatch("set_reminder", {"duration_seconds": secs, "message": label})

        # Semantic Memory & RAG
        m = re.match(r"^(?:please\s+)?remember\s+(?:that\s+)?(.+)$", raw, re.IGNORECASE)
        if m:
            return IntentMatch("remember_fact", {"fact": m.group(1).strip()})
        if re.search(r"\b(?:index|refresh|reload|ingest)\s+(?:my\s+|the\s+)?(?:knowledge|documents|docs|rag)\b", low):
            return IntentMatch("index_knowledge_base", {})
        m = re.search(
            r"\b(?:what\s+do\s+you\s+remember\s+about|recall|search\s+(?:my\s+)?(?:memory|knowledge\s*base|docs|rag)\s+(?:for|about))\s+(.+)$",
            raw,
            re.IGNORECASE,
        )
        if m:
            return IntentMatch("recall_memory", {"query": m.group(1).strip()})

        # Notes
        if re.search(r"\b(?:show|list|read)\s+(?:my\s+)?notes\b", low):
            return IntentMatch("manage_notes", {"action": "list"})
        m = re.match(r"^(?:take|create|add|save|make)\s+(?:a\s+)?note(?:\s+that|\s*:\s*|\s+)(.+)$", raw, re.IGNORECASE)
        if m:
            return IntentMatch("manage_notes", {"action": "add", "content": m.group(1).strip()})
        m = re.search(r"\bdelete\s+note\s+#?(\d+)\b", low)
        if m:
            return IntentMatch("manage_notes", {"action": "delete", "note_id": int(m.group(1))})

        # Tasks / To-Do
        if re.search(r"\b(?:show|list|read)\s+(?:my\s+)?(?:tasks|to-?do(?:\s+list)?)\b", low):
            return IntentMatch("manage_tasks", {"action": "list"})
        m = re.match(r"^(?:add|create)\s+(?:a\s+)?(?:task|to-?do)(?:\s*:\s*|\s+)(.+)$", raw, re.IGNORECASE)
        if m:
            return IntentMatch("manage_tasks", {"action": "add", "title": m.group(1).strip()})
        m = re.search(r"\b(?:complete|finish|check\s+off|done)\s+task\s+#?(\d+)\b", low)
        if m:
            return IntentMatch("manage_tasks", {"action": "complete", "task_id": int(m.group(1))})
        m = re.search(r"\bdelete\s+task\s+#?(\d+)\b", low)
        if m:
            return IntentMatch("manage_tasks", {"action": "delete", "task_id": int(m.group(1))})

        # Weather (check Hinglish '<city> ka/mein weather/mausam' FIRST so optional-city English regex doesn't shadow it)
        m = re.search(
            r"^(?:(?:bhai|yaar|jarvis)[\s,]+)?([a-zA-Z\s,]+?)\s+(?:ka|ki|mein|me)\s+"
            r"(?:weather|temperature|mausam)(?:\s+(?:batao|kaisa\s+hai|kya\s+hai|dikhao))?$",
            low,
        )
        if not m:
            m = re.search(
                r"\b(?:weather|temperature|forecast|mausam)(?:\s+(?:in|for|at)\s+([a-zA-Z\s,]+?))?"
                r"(?:\s+(?:right\s+now|currently|today|now|outside|batao|kaisa\s+hai))?$",
                low,
            )
        if m:
            city = re.sub(
                r"\s+(?:right\s+now|currently|today|now|outside|batao|kaisa\s+hai|kya\s+hai)$",
                "",
                (m.group(1) or "").strip(),
            ).strip()
            return IntentMatch("get_weather", {"city": city})

        # News (ignore casual banter like "good news", "bad news", "no news")
        if re.search(
            r"\b(?:headlines|(?:latest|top|today'?s|breaking|tech|world|science)\s+news|"
            r"(?:read|get|show|check|give\s+me)\s+(?:the\s+)?news)\b",
            low,
        ) and not re.search(r"\b(?:good|bad|great|awesome|no)\s+news\b", low):
            cat = "science" if "science" in low else ("world" if "world" in low else "tech")
            return IntentMatch("get_top_news", {"category": cat})

        # Calculator (check before Wikipedia so "what is sqrt(144) + 25" calculates accurately)
        m = re.match(r"^(?:calculate|compute|solve|what\s+is|evaluate)\s+([0-9\s\+\-\*/\^\(\)\.a-z]+)$", low)
        if m and any(ch.isdigit() for ch in m.group(1)) and re.search(r"[\+\-\*/\^]|plus|minus|times|divided|sqrt|sin|cos|log", m.group(1)):
            return IntentMatch("calculate", {"expression": m.group(1).strip()})

        # Live Knowledge & Wikipedia Lookup ("tell about MS Dhoni", "who is ...", "explain ...", "define ...")
        m = re.search(r"\b(?:wikipedia|wiki)\s+(?:for\s+|about\s+)?(.+)$", raw, re.IGNORECASE)
        if m:
            return IntentMatch("search_wikipedia", {"topic": m.group(1).strip()})
        m = re.match(
            r"^(?:can\s+you\s+|please\s+)?(?:who\s+(?:is|was|are|were)|tell\s+(?:me\s+)?about|talk\s+about|"
            r"(?:give\s+me\s+)?(?:info|information|details|facts)\s+(?:about|on)|"
            r"what\s+do\s+you\s+know\s+about|explain(?:\s+to\s+me)?|describe|define|"
            r"(?:what(?:'s|\s+is)\s+the\s+)?meaning\s+of|where\s+(?:is|are))\s+(.+)$",
            raw,
            re.IGNORECASE,
        )
        if m and not re.search(r"\b(?:time|date|weather|status|cpu|ram|battery|clipboard|my\s+(?:notes|tasks|reminders|pc|computer|screen|memory))\b", low):
            topic = m.group(1).strip().rstrip("?.!")
            if topic and len(topic.split()) <= 8 and topic.lower() not in {
                "it", "this", "that", "today", "yourself", "you",
                "my day", "your day", "movies", "life", "work", "stuff", "things", "anything", "something",
            }:
                return IntentMatch("search_wikipedia", {"topic": topic})

        # YouTube / GitHub / Google Search
        m = re.search(r"\b(?:play|search)\s+(.+?)\s+on\s+youtube\b", raw, re.IGNORECASE)
        if m:
            return IntentMatch("web_search_and_open", {"query": m.group(1).strip(), "platform": "youtube"})
        m = re.search(r"\bsearch\s+github\s+(?:for\s+)?(.+)$", raw, re.IGNORECASE)
        if m:
            return IntentMatch("web_search_and_open", {"query": m.group(1).strip(), "platform": "github"})
        m = re.search(r"\b(?:search\s+(?:google\s+)?for|google)\s+(.+)$", raw, re.IGNORECASE)
        if m:
            return IntentMatch("web_search_and_open", {"query": m.group(1).strip(), "platform": "google"})

        # Window Management
        if re.search(r"\b(?:list|show)\s+(?:all\s+)?(?:open\s+)?windows\b", low):
            return IntentMatch("manage_window", {"action": "list"})
        m = re.match(r"^(focus|switch\s+to|minimize|maximize|close)\s+(?:the\s+)?(.+?)(?:\s+window)?$", raw, re.IGNORECASE)
        if m:
            verb = m.group(1).lower()
            action = "focus" if "switch" in verb else verb
            return IntentMatch("manage_window", {"action": action, "title": m.group(2).strip()})

        # Dictate / Type text
        m = re.match(r"^(?:type|dictate)\s+(.+)$", raw, re.IGNORECASE)
        if m:
            return IntentMatch("type_text", {"text": m.group(1).strip()})

        # Persona, Tone & Voice Modulation ("witty mode", "switch persona to analytical", "be more crisp")
        m = re.search(r"\b(?:switch\s+(?:persona|tone|mode)\s+to|be\s+more|set\s+persona\s+to)\s+(calm|crisp|witty|analytical)\b", low)
        if m:
            return IntentMatch("switch_persona", {"persona": m.group(1)})
        m = re.match(r"^(calm|crisp|witty|analytical)\s+(?:mode|persona|tone)$", low)
        if m:
            return IntentMatch("switch_persona", {"persona": m.group(1)})

        # Vision: Screen & Webcam Inspection
        if re.search(
            r"\b(?:what(?:'s|\s+is)\s+(?:wrong\s+with\s+.*?)?on\s+my\s+screen|see\s+my\s+screen|"
            r"analyze\s+(?:my\s+)?screen|inspect\s+(?:my\s+)?screen|read\s+(?:my\s+)?screen|"
            r"summarize\s+what(?:'s|\s+is)\s+open\s+(?:right\s+now|on\s+my\s+screen)?)\b",
            low,
        ):
            return IntentMatch("analyze_screen", {"question": raw})
        if re.search(r"\b(?:look\s+through\s+(?:my\s+)?(?:camera|webcam)|what\s+do\s+you\s+see\s+on\s+(?:my\s+)?(?:camera|webcam)|analyze\s+(?:my\s+)?(?:camera|webcam)|check\s+(?:my\s+)?webcam)\b", low):
            return IntentMatch("analyze_camera", {"question": raw})

        # Smart Home & IoT Control (Home Assistant / ESP32 / MQTT / Local State)
        if re.search(r"\b(?:list|show)\s+(?:my\s+)?smart(?:\s+home)?\s+devices\b|\bsmart\s+home\s+status\b", low):
            return IntentMatch("list_smart_devices", {})
        m = re.search(r"\bdim\s+(?:the\s+)?([a-z0-9\s]+?)\s+to\s+(\d{1,3})\s*%?(?:\s+percent)?$", low)
        if m and "screen" not in m.group(1) and "brightness" not in m.group(1):
            return IntentMatch("control_smart_device", {"device": m.group(1).strip(), "action": "dim", "level": int(m.group(2))})
        m = re.match(r"^(?:turn|switch)\s+(on|off)\s+(?:the\s+)?([a-z0-9\s]+)$", low)
        if m and not re.search(r"\b(?:volume|brightness|pc|computer|system|voice)\b", m.group(2)):
            return IntentMatch("control_smart_device", {"device": m.group(2).strip(), "action": m.group(1)})

        # Developer tools, Coding Workflows & Human-in-the-Loop Guarded Operations
        if re.search(r"\b(?:run\s+(?:project\s+|unit\s+)?tests?|run\s+pytest|check\s+(?:if\s+)?tests?\s+pass)\b", low):
            return IntentMatch("run_project_tests", {})
        if re.search(r"\b(?:index|ingest)\s+(?:my\s+)?(?:codebase|code|workspace)\b", low):
            return IntentMatch("index_codebase_rag", {})
        if re.search(r"\b(?:list|show)\s+git\s+branches\b", low):
            return IntentMatch("git_workflow", {"action": "branches"})
        m = re.search(r"\b(?:switch\s+(?:git\s+)?branch\s+to|git\s+switch|checkout\s+branch)\s+([a-zA-Z0-9._/\-]+)\b", raw, re.IGNORECASE)
        if m:
            return IntentMatch("git_workflow", {"action": "switch", "branch": m.group(1).strip()})
        if re.search(r"\b(?:git\s+log|recent\s+commits?)\b", low):
            return IntentMatch("git_workflow", {"action": "log"})
        if re.search(r"\b(?:git\s+diff|diff\s+summary)\b", low):
            return IntentMatch("git_workflow", {"action": "diff_summary"})
        m = re.match(r"^(?:run\s+shell(?:\s+command)?|execute\s+shell(?:\s+command)?)\s+(.+)$", raw, re.IGNORECASE)
        if m:
            return IntentMatch("run_shell_command", {"command": m.group(1).strip()})
        m = re.match(r"^delete\s+(?:workspace\s+)?file\s+(.+)$", raw, re.IGNORECASE)
        if m:
            return IntentMatch("delete_workspace_file", {"relative_path": m.group(1).strip()})
        m = re.match(r"^(?:kill|terminate)\s+process\s+(.+)$", raw, re.IGNORECASE)
        if m:
            return IntentMatch("kill_process", {"process_name": m.group(1).strip()})
        if re.search(r"\b(?:git\s+status|git\s+branch)\b", low):
            return IntentMatch("git_status", {})
        m = re.search(r"\b(?:find|search)\s+(?:workspace\s+)?files?\s+(?:matching|named|for)?\s*(.+)$", raw, re.IGNORECASE)
        if m:
            return IntentMatch("search_workspace_files", {"pattern": m.group(1).strip()})

        # Open apps or websites (English + Hinglish: "open google", "youtube khol do", "notepad open karo")
        m = re.match(r"^(open|launch|start|run|go\s+to|visit|browse)\s+(?:the\s+)?(.+)$", raw, re.IGNORECASE)
        if not m:
            hm = re.match(r"^(.+?)\s+(?:khol\s+do|kholo|open\s+karo|chalu\s+karo)$", raw.strip(" .!?"), re.IGNORECASE)
            if hm:
                target = hm.group(1).strip().rstrip(" .!?")
                t_low = target.lower()
                if t_low in SITE_SHORTCUTS or t_low.startswith(("http://", "https://", "www.")):
                    return IntentMatch("web_search_and_open", {"query": target, "platform": "url"})
                if t_low in APP_CATALOG or len(t_low.split()) <= 3:
                    return IntentMatch("open_application", {"app_name": target})
        if m:
            verb = m.group(1).lower()
            target = m.group(2).strip().rstrip(" .!?")
            t_low = target.lower()
            if (
                t_low in SITE_SHORTCUTS
                or t_low.startswith(("http://", "https://", "www."))
                or re.match(r"^[a-z0-9\-]+\.(com|org|net|io|dev|ai|in|co|app|tv|edu|gov)(?:/.*)?$", t_low)
            ):
                return IntentMatch("web_search_and_open", {"query": target, "platform": "url"})
            if t_low in APP_CATALOG or (verb in {"open", "launch"} and len(t_low.split()) <= 3):
                return IntentMatch("open_application", {"app_name": target})

        return None
