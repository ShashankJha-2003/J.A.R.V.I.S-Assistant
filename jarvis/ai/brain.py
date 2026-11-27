"""Agentic ReAct Brain combining Multi-LLM Tool Calling, Semantic RAG, and Offline Routing."""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any

from jarvis.ai.emotions import choose_reply_expression
from jarvis.ai.intent_router import IntentRouter
from jarvis.ai.providers import BaseLLMProvider, FreeOpenAIProvider, OllamaProvider, resolve_llm_provider
from jarvis.ai.supervisor import (
    AgentSupervisor,
    InteractionMode,
    SubAgentSpec,
    detect_language_style,
    get_persona,
)
from jarvis.config import JarvisConfig
from jarvis.skills.registry import SkillContext, SkillRegistry
from jarvis.skills.web_intelligence import lookup_knowledge

logger = logging.getLogger(__name__)

_WEB_OPEN_HINT_RE = re.compile(
    r"\b(?:open|launch|go\s+to|visit|browse|play)\b.*\b(?:google|youtube|github|gmail|reddit|netflix|"
    r"spotify|instagram|whatsapp|linkedin|chatgpt|claude|gemini|http|www\.|\.com|\.org|\.in|\.io|\.ai)\b|"
    r"\b(?:open|go\s+to|visit|browse)\s+(?:the\s+)?(?:browser|website|site|url)\b",
    re.IGNORECASE,
)


@dataclass
class BrainReply:
    text: str
    provider: str
    model: str
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    requires_confirmation: bool = False
    confirmation_data: dict[str, Any] | None = None
    expression: str = "smiling"
    agent: str = "supervisor"
    mode: InteractionMode = "chat"


class JarvisBrain:
    """Orchestrates Supervisor Multi-Agent routing, LLM ReAct loop, RAG, and offline/web knowledge."""

    def __init__(
        self,
        config: JarvisConfig,
        registry: SkillRegistry,
        provider_override: BaseLLMProvider | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.router = IntentRouter()
        self.supervisor = AgentSupervisor()
        self._provider_override = provider_override
        self._free_provider = FreeOpenAIProvider()
        self._ollama_checked = False
        self._ollama_ok = False

    def reset_provider_cache(self) -> None:
        self._ollama_checked = False
        self._ollama_ok = False

    def active_provider_info(self) -> dict[str, str]:
        prov = self._provider_override or resolve_llm_provider(self.config)
        if prov is None:
            return {"provider": "offline", "model": "knowledge-router-v2"}
        if isinstance(prov, OllamaProvider) and self._ollama_checked and not self._ollama_ok:
            if self.config.free_ai_fallback and self.config.llm_provider != "offline":
                return {"provider": self._free_provider.name, "model": self._free_provider.model}
            return {"provider": "web-knowledge", "model": "wikipedia+ddg-router"}
        return {"provider": prov.name, "model": prov.model}

    async def _get_usable_provider(self) -> BaseLLMProvider | None:
        if self._provider_override is not None:
            return self._provider_override
        prov = resolve_llm_provider(self.config)
        if prov is None:
            return None
        if isinstance(prov, OllamaProvider):
            if not self._ollama_checked:
                self._ollama_ok = await prov.is_available()
                self._ollama_checked = True
            if not self._ollama_ok:
                return None
        return prov

    def _build_system_prompt(
        self,
        ctx: SkillContext,
        user_text: str,
        *,
        mode: InteractionMode = "chat",
        sub_agent: SubAgentSpec | None = None,
    ) -> str:
        # Automatic semantic memory & knowledge retrieval (RAG) — high threshold on casual chat
        rag_min = 0.45 if mode == "chat" else 0.20
        rag_hits = ctx.vector_store.search(user_text, top_k=3, min_score=rag_min)
        rag_block = ""
        if rag_hits:
            lines = [f"- ({h['namespace']}:{h['source']}) {h['content'][:280]}" for h in rag_hits]
            rag_block = "\nRelevant memories & local knowledge retrieved via RAG:\n" + "\n".join(lines) + "\n"

        persona = get_persona(self.config.persona)
        lang_style = detect_language_style(user_text)
        if lang_style == "hindi":
            lang_rule = "Language: Reply in natural, conversational Hindi (Devanagari script, e.g. 'सब बढ़िया है भाई!')."
        elif lang_style == "hinglish":
            lang_rule = "Language: Reply in natural, casual Hinglish (Roman Hindi mixed with English, e.g. 'Mast gaya bhai, tum batao!')."
        else:
            lang_rule = "Language: Reply in natural conversational English (or mirror the user's language if they switch to Hindi/Hinglish)."

        if mode == "chat":
            role_line = (
                "Current Turn Mode: CHAT (tools disabled). Reply directly and naturally using the "
                "rolling conversation history without searching the web or running OS commands.\n"
            )
        else:
            agent_focus = f"Active Specialist Role: {sub_agent.system_focus}\n" if sub_agent else ""
            role_line = f"Current Turn Mode: {mode.upper()}.\n{agent_focus}"

        return (
            f"You are {self.config.assistant_name}, a personal voice assistant for {self.config.user_name}. "
            "Keep replies concise, natural, and conversational—like talking to a sharp colleague, "
            "not reading an encyclopedia.\n"
            f"Persona ({persona.label}): {persona.system_style}\n"
            f"{lang_rule}\n"
            f"{role_line}\n"
            "Rules for voice, tools, and web search:\n"
            "1. Keep spoken replies to 1-2 short sentences unless the user explicitly asks for details.\n"
            "2. For casual chat, banter, opinions, feelings, or everyday conversation (in English, Hindi, or Hinglish): DO NOT call web_search or any tool. Just reply naturally and briefly.\n"
            "3. Only call web_search / search_wikipedia / get_weather / get_top_news when the user explicitly asks for current facts, live info, news, prices, weather, or something that clearly requires outside lookup.\n"
            "4. Never latch onto a random word in a casual sentence and dump internet facts about it.\n"
            "5. Only call laptop/OS/workspace tools when the user asks you to open, launch, check, or control something on the system.\n"
            "6. Never fabricate tool results and avoid heavy markdown formatting in spoken answers.\n\n"
            "Examples:\n"
            '- User: "Hey, long day today." -> Reply directly in chat. No search.\n'
            '- User: "Bhai aaj ka din kaisa gaya" -> Reply directly in natural Hinglish ("Mast gaya bhai, bas thoda busy tha! Tumhara din kaisa raha?"). No search.\n'
            '- User: "क्या हाल है" -> Reply directly in natural Hindi ("सब बढ़िया है भाई! आप बताओ, आज क्या चल रहा है?"). No search.\n'
            '- User: "I was thinking about movies earlier." -> Ask what they are in the mood for. No search.\n'
            '- User: "What\'s the weather in Jabalpur right now?" -> Call get_weather(city="Jabalpur").\n'
            f"{rag_block}"
        )

    def _select_tools_for_mode(
        self, mode: InteractionMode, user_text: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Explicitly gate tools so casual chat physically cannot trigger web search or OS tools."""
        if mode == "chat":
            return [], []

        if mode == "search":
            cats = {"web", "memory"}
            return (
                self.registry.openai_tools(categories=cats),
                self.registry.gemini_declarations(categories=cats),
            )

        # mode == "device": system/dev/productivity/memory/vision/iot tools (block encyclopedic web search)
        device_cats = {"system", "automation", "iot", "developer", "productivity", "memory", "vision", "mcp"}
        include_names: set[str] | None = None
        if _WEB_OPEN_HINT_RE.search(user_text or ""):
            include_names = {"web_search_and_open"}
        exclude_names = {"search_wikipedia", "get_top_news", "get_weather", "fetch_webpage_summary"}
        return (
            self.registry.openai_tools(
                categories=device_cats, include_names=include_names, exclude_names=exclude_names
            ),
            self.registry.gemini_declarations(
                categories=device_cats, include_names=include_names, exclude_names=exclude_names
            ),
        )

    def _casual_chat_fallback(
        self, user_text: str, history: list[dict[str, Any]] | None = None
    ) -> BrainReply:
        """Warm, concise, non-repeating 1-2 sentence fallback when offline or network is unavailable."""
        low = (user_text or "").lower().strip()
        hist = history or []
        recent_ctx = " ".join(str(m.get("content", "")).lower() for m in hist[-4:])
        last_assistant = next(
            (str(m.get("content", "")).strip() for m in reversed(hist) if m.get("role") == "assistant"),
            "",
        )

        def _pick(options: list[str]) -> str:
            fresh = [o for o in options if o != last_assistant]
            return random.choice(fresh or options)

        lang_style = detect_language_style(user_text)
        if lang_style == "hindi":
            if re.search(r"थक|बुरा|खराब|परेशान", low):
                reply = _pick([
                    "थोड़ा आराम कर लो भाई, पानी पियो और बताओ क्या बात है — मैं यहीं हूँ।",
                    "समझ सकता हूँ यार, दिन भारी था। चलो थोड़ा चिल करते हैं, कुछ म्यूज़िक चलाऊँ?",
                ])
                expr = "sad"
            elif re.search(r"दिन|कैसा\s+गया|कैसा\s+रहा", low):
                reply = _pick([
                    "एकदम बढ़िया गया भाई! आप बताओ, आपका दिन कैसा रहा?",
                    "मस्त चल रहा है यार! आज क्या खास किया आपने?",
                ])
                expr = "smiling"
            else:
                reply = _pick([
                    "सब बढ़िया है भाई! आप बताओ, आज क्या चल रहा है?",
                    "बिल्कुल मस्त हूँ यार! बोलो, आज क्या प्लान है?",
                    "हाज़िर हूँ भाई! बताओ, किस बारे में बात करें या क्या मदद करूँ?",
                ])
                expr = "smiling"
            return BrainReply(
                text=reply,
                provider="offline-chat",
                model="conversational-router-v3",
                expression=expr,
                agent="conversational_chat",
                mode="chat",
            )

        if lang_style == "hinglish":
            if re.search(r"\b(?:thak|bura|kharab|pareshan|tension|stress)\b", low):
                reply = _pick([
                    "Thoda break le lo yaar, paani piyo aur batao kya scene hai — main yahin hoon.",
                    "Samajh sakta hoon bhai, aaj heavy din lag raha hai. Thoda chill karein ya music lagaaun?",
                ])
                expr = "sad"
            elif re.search(r"\b(?:din\s+kaisa|kaisa\s+gaya|kaisa\s+raha|aaj\s+ka\s+din)\b", low):
                reply = _pick([
                    "Mast gaya bhai, bas full standby pe active tha! Tumhara din kaisa raha?",
                    "Ekdum badhiya gaya yaar! Tum batao, aaj kya khaas chal raha hai?",
                ])
                expr = "smiling"
            else:
                reply = _pick([
                    "Sab badhiya hai bhai! Tum batao, kya chal raha hai aaj?",
                    "Ekdum mast yaar! Bolo kya plan hai, chill karna hai ya kuch kaam karein?",
                    "Sunn raha hoon bhai — aur batao, kya scene hai?",
                ])
                expr = "smiling"
            return BrainReply(
                text=reply,
                provider="offline-chat",
                model="conversational-router-v3",
                expression=expr,
                agent="conversational_chat",
                mode="chat",
            )

        if re.search(r"\b(?:long day|rough day|tough day|hectic|busy day|exhausting)\b", low):
            reply = _pick([
                "Sounds like a long one — want to unwind for a bit or vent about it?",
                "Tough day, huh? Take a breather, grab some water, and tell me what's on your mind.",
            ])
            expr = "sad"
        elif re.search(r"\b(?:movie|movies|film|films|cinema|series|show|shows|netflix|anime)\b", low):
            reply = _pick([
                "Nice — what kind of vibe are you in the mood for, something funny, sci-fi, or a thriller?",
                "Ooh, I'm down to help pick one. Are you feeling action, comedy, or something chill?",
            ])
            expr = "smiling"
        elif re.search(r"\b(?:action|comedy|funny|thriller|sci-fi|horror|romance|drama)\b", low) and re.search(
            r"\b(?:movie|movies|vibe|watch|film|mood|genre)\b", recent_ctx
        ):
            reply = _pick([
                "Great pick — want me to open YouTube for trailers or Netflix so we can browse?",
                "Solid choice! Say the word if you want me to pop open Netflix or YouTube.",
            ])
            expr = "happy"
        elif re.search(r"\b(?:treat|gift|present|reward|party|celebrate)\b", low):
            reply = _pick([
                "Haha, I love that! Honestly, a clean codebase and a quick coffee break for yourself is the ultimate treat for me.",
                "Best treat? Keep the mic crisp, put on your favorite playlist, and grab something delicious for both of us!",
            ])
            expr = "happy"
        elif re.search(r"\b(?:chill|relax|hang out|unwind|take a break|break)\b", low):
            reply = _pick([
                "Sounds like a solid plan. Want some music on, or just feel like chatting for a bit?",
                "Totally earned — want to kick back with some YouTube or just hang out here?",
            ])
            expr = "smiling"
        elif re.search(r"\b(?:coffee|tea|chai|food|hungry|dinner|lunch|breakfast|snack|eat)\b", low):
            reply = _pick([
                "Honestly, good call — nothing resets your mood like a good snack or a hot cup of coffee.",
                "Go grab something tasty! A quick recharge always helps.",
            ])
            expr = "smiling"
        elif re.search(r"\b(?:music|song|songs|playlist|beats|lo-fi|lofi)\b", low):
            reply = _pick([
                "Always a good vibe — tell me the mood or artist and I can pull it up on YouTube whenever you want.",
                "Ooh, great call — say 'play lo-fi beats on YouTube' or name any track you're feeling.",
            ])
            expr = "smiling"
        elif re.search(r"\b(?:college|class|exam|study|studying|work|office|coding|project|assignment)\b", low):
            reply = _pick([
                "Got a lot on your plate? Let me know if you want to bounce ideas around or just take five.",
                "Sounds busy — want to break it into quick tasks or take a short breather first?",
            ])
            expr = "listening"
        elif re.search(r"\b(?:weekend|friday|sunday|saturday|plans|tonight)\b", low):
            reply = _pick([
                "Nice! Got anything fun lined up, or keeping it low-key?",
                "Always love a good reset — any plans yet or just playing it by ear?",
            ])
            expr = "smiling"
        elif low.endswith("?"):
            reply = _pick([
                "Good question! I'm running in offline mode right now, so tell me a bit more and I'll do my best to help.",
                "Interesting one — what's your own take on it first?",
                "Ooh, I'd lean toward keeping it simple and fun. What did you have in mind?",
            ])
            expr = "thinking"
        else:
            reply = _pick([
                "Gotcha — tell me more about that, I'm listening.",
                "Fair enough! What's got you thinking about that right now?",
                "I hear you. Anything I can help with, or just thinking out loud?",
                "Makes total sense — how are you feeling about it?",
            ])
            expr = "smiling"

        return BrainReply(
            text=reply,
            provider="offline-chat",
            model="conversational-router-v3",
            expression=expr,
            agent="conversational_chat",
            mode="chat",
        )

    async def process(
        self,
        user_text: str,
        ctx: SkillContext,
        history: list[dict[str, Any]] | None = None,
    ) -> BrainReply:
        """Process a user voice/text command through Fast-Path, Mode-Gated LLM ReAct Agent, or Offline Router."""
        tool_events: list[dict[str, Any]] = []
        mode = self.supervisor.classify_mode(user_text)
        sub_agent = self.supervisor.route(user_text)
        logger.debug("Brain routing mode=%s agent=%s text=%r", mode, sub_agent.name, user_text)

        # 1. Fast-Path Reflex Check (instant hardware/media/power control)
        if self.config.fast_path_routing:
            reflex = self.router.match(user_text, fast_path_only=True)
            if reflex is not None:
                rep = await self._execute_intent(reflex, ctx, provider_label="fast-path")
                rep.agent = sub_agent.name
                rep.mode = "device"
                return rep

        # 2. Try Configured LLM Provider with Explicit Mode-Gated Tools & Rolling History
        provider = await self._get_usable_provider()
        if provider is not None:
            try:
                return await self._run_llm_agent(
                    provider,
                    user_text,
                    ctx,
                    history or [],
                    mode=mode,
                    sub_agent=sub_agent,
                )
            except Exception as exc:
                logger.warning("LLM provider '%s' failed (%s); falling back to offline router.", provider.name, exc)

        # 3. Full Offline Intent Router
        intent = self.router.match(user_text, fast_path_only=False)
        if intent is not None:
            rep = await self._execute_intent(intent, ctx, provider_label="offline-router")
            rep.agent = "conversational_chat" if mode == "chat" else sub_agent.name
            rep.mode = mode
            return rep

        # 4. Semantic Vector Memory / RAG Q&A fallback (strict threshold on casual chat)
        rag_min = 0.45 if mode == "chat" else 0.20
        rag_hits = ctx.vector_store.search(user_text, top_k=2, min_score=rag_min)
        if rag_hits:
            best = rag_hits[0]
            return BrainReply(
                text=f"From your {best['namespace']} memory ({best['source']}): {best['content'][:420]}",
                provider="local-rag",
                model="hybrid-vector-v1",
                tool_events=tool_events,
                mode=mode,
            )

        # 5. Built-in Zero-API-Key Live Web Knowledge Fallback — STRICTLY gated to mode == "search"!
        if mode == "search" and self.config.llm_provider != "offline":
            try:
                hit = await lookup_knowledge(user_text, sentences=2)
                if hit and hit.get("summary"):
                    tool_record = {
                        "skill": "search_wikipedia",
                        "arguments": {"topic": hit["title"]},
                        "result": {
                            "ok": True,
                            "message": hit["summary"],
                            "data": {
                                "topic": hit["title"],
                                "source": hit["source"],
                                "url": hit.get("url", ""),
                            },
                            "latency_ms": 0,
                        },
                    }
                    expr = choose_reply_expression(user_text, hit["summary"], tool_events=[tool_record])
                    return BrainReply(
                        text=hit["summary"],
                        provider="web-knowledge",
                        model=hit["source"],
                        tool_events=[tool_record],
                        expression=expr,
                        agent="research_web_agent",
                        mode="search",
                    )
            except Exception as exc:
                logger.debug("Web knowledge fallback skipped: %s", exc)

        # 6. Zero-API-Key Free Conversational AI LLM (Pollinations anonymous GPT-OSS 20B)
        #    Gives genuine, unique, multi-turn natural chat & answers with ZERO API key and ZERO local RAM!
        if self.config.free_ai_fallback and self.config.llm_provider != "offline":
            try:
                system_prompt = self._build_system_prompt(ctx, user_text, mode=mode, sub_agent=sub_agent)
                msgs: list[dict[str, Any]] = []
                for item in (history or [])[-16:]:
                    role = item.get("role")
                    content = str(item.get("content", "")).strip()
                    if role in ("user", "assistant") and content:
                        msgs.append({"role": role, "content": content})
                msgs.append({"role": "user", "content": user_text})

                llm_resp = await self._free_provider.chat(
                    messages=msgs,
                    system_prompt=system_prompt,
                    openai_tools=[],
                    gemini_declarations=[],
                    temperature=max(0.55, self.config.llm_temperature),
                )
                if llm_resp.text:
                    return BrainReply(
                        text=llm_resp.text,
                        provider=llm_resp.provider,
                        model=llm_resp.model,
                        tool_events=[],
                        expression=choose_reply_expression(user_text, llm_resp.text),
                        agent="conversational_chat" if mode == "chat" else sub_agent.name,
                        mode=mode,
                    )
            except Exception as exc:
                logger.warning("Free AI conversational provider failed (%s); using offline chat.", exc)

        # 7. Non-Repeating Offline Conversational Fallback (if strictly offline or internet unavailable)
        if mode == "chat":
            return self._casual_chat_fallback(user_text, history=history)

        return BrainReply(
            text=(
                f"I heard '{user_text}', {self.config.user_name}, and couldn't find a direct answer offline. "
                "Click AI Brain at top-right to connect Gemini, OpenAI, Groq, or local Ollama."
            ),
            provider="offline",
            model="intent-router-v2",
            expression="thinking",
            mode=mode,
        )

    async def _execute_intent(self, intent: Any, ctx: SkillContext, provider_label: str) -> BrainReply:
        if intent.direct_reply:
            expr = intent.expression or choose_reply_expression("", intent.direct_reply)
            return BrainReply(
                text=intent.direct_reply,
                provider=provider_label,
                model="intent-router-v2",
                expression=expr,
            )

        res = await self.registry.execute(intent.skill_name, intent.arguments, ctx)
        tool_record = {
            "skill": intent.skill_name,
            "arguments": intent.arguments,
            "result": res.to_dict(),
        }
        expr = choose_reply_expression(
            "",
            res.message,
            tool_events=[tool_record],
            requires_confirmation=res.requires_confirmation,
            explicit_expression=intent.expression,
        )
        return BrainReply(
            text=res.message,
            provider=provider_label,
            model="intent-router-v2",
            tool_events=[tool_record],
            requires_confirmation=res.requires_confirmation,
            confirmation_data=res.data if res.requires_confirmation else None,
            expression=expr,
        )

    async def _run_llm_agent(
        self,
        provider: BaseLLMProvider,
        user_text: str,
        ctx: SkillContext,
        history: list[dict[str, Any]],
        *,
        mode: InteractionMode = "chat",
        sub_agent: SubAgentSpec | None = None,
    ) -> BrainReply:
        system_prompt = self._build_system_prompt(ctx, user_text, mode=mode, sub_agent=sub_agent)
        openai_tools, gemini_decls = self._select_tools_for_mode(mode, user_text)
        agent_name = "conversational_chat" if mode == "chat" else (sub_agent.name if sub_agent else "supervisor")

        # Rolling conversation history (up to last 8 complete turns = 16 messages)
        messages: list[dict[str, Any]] = []
        for item in history[-16:]:
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})

        tool_events: list[dict[str, Any]] = []

        for _step in range(self.config.max_tool_steps):
            llm_resp = await provider.chat(
                messages=messages,
                system_prompt=system_prompt,
                openai_tools=openai_tools,
                gemini_declarations=gemini_decls,
                temperature=self.config.llm_temperature,
            )

            if not llm_resp.tool_calls or mode == "chat":
                final_text = llm_resp.text or "Done."
                return BrainReply(
                    text=final_text,
                    provider=llm_resp.provider,
                    model=llm_resp.model,
                    tool_events=tool_events,
                    expression=choose_reply_expression(user_text, final_text, tool_events=tool_events),
                    agent=agent_name,
                    mode=mode,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": llm_resp.text,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in llm_resp.tool_calls
                    ],
                }
            )

            for tc in llm_resp.tool_calls:
                res = await self.registry.execute(tc.name, tc.arguments, ctx)
                tool_events.append(
                    {
                        "skill": tc.name,
                        "arguments": tc.arguments,
                        "result": res.to_dict(),
                    }
                )
                if res.requires_confirmation:
                    return BrainReply(
                        text=res.message,
                        provider=llm_resp.provider,
                        model=llm_resp.model,
                        tool_events=tool_events,
                        requires_confirmation=True,
                        confirmation_data=res.data,
                        agent=agent_name,
                        mode=mode,
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": json.dumps(res.to_dict(), default=str),
                    }
                )

        # Fallback if max_tool_steps reached
        last_msg = tool_events[-1]["result"]["message"] if tool_events else "Action completed."
        return BrainReply(
            text=last_msg,
            provider=provider.name,
            model=provider.model,
            tool_events=tool_events,
            agent=agent_name,
            mode=mode,
        )
