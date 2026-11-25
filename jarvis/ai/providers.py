"""Async Multi-Provider LLM Adapters with unified function/tool calling.

Supports:
- Google Gemini (REST v1beta with native functionDeclarations)
- OpenAI / Groq / OpenRouter / any OpenAI-compatible API
- Local Ollama (http://localhost:11434/api/chat)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from jarvis.config import JarvisConfig

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    provider: str = "offline"
    model: str = "local-router"
    raw: dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider:
    name: str = "base"
    model: str = ""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        openai_tools: list[dict[str, Any]],
        gemini_declarations: list[dict[str, Any]],
        temperature: float = 0.4,
    ) -> LLMResponse:
        raise NotImplementedError

    async def chat_with_image(
        self,
        prompt: str,
        image_b64: str,
        mime_type: str = "image/jpeg",
        system_prompt: str = "Analyze this image concisely for voice playback.",
    ) -> LLMResponse:
        raise NotImplementedError("This provider does not implement multimodal vision.")


class GeminiProvider(BaseLLMProvider):
    """Google Gemini REST provider with native tool calling."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        self.name = "gemini"
        self.api_key = api_key
        self.model = model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        openai_tools: list[dict[str, Any]],
        gemini_declarations: list[dict[str, Any]],
        temperature: float = 0.4,
    ) -> LLMResponse:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            if role == "user":
                contents.append({"role": "user", "parts": [{"text": msg.get("content", "")}]})
            elif role == "assistant":
                parts: list[dict[str, Any]] = []
                if msg.get("content"):
                    parts.append({"text": msg["content"]})
                for tc in msg.get("tool_calls") or []:
                    parts.append({"functionCall": {"name": tc["name"], "args": tc.get("arguments", {})}})
                if parts:
                    contents.append({"role": "model", "parts": parts})
            elif role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.get("name", "tool"),
                                    "response": {"result": msg.get("content", "")},
                                }
                            }
                        ],
                    }
                )

        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 700},
        }
        if gemini_declarations:
            body["tools"] = [{"functionDeclarations": gemini_declarations}]

        async with httpx.AsyncClient(timeout=35.0) as client:
            resp = await client.post(url, params={"key": self.api_key}, json=body)
            resp.raise_for_status()
            data = resp.json()

        candidates = data.get("candidates") or []
        if not candidates:
            return LLMResponse(text="", provider=self.name, model=self.model, raw=data)

        parts = candidates[0].get("content", {}).get("parts", [])
        text_chunks: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        for part in parts:
            if "text" in part and part["text"]:
                text_chunks.append(part["text"])
            if "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCallRequest(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name=fc.get("name", ""),
                        arguments=fc.get("args") or {},
                    )
                )

        return LLMResponse(
            text=" ".join(text_chunks).strip(),
            tool_calls=tool_calls,
            provider=self.name,
            model=self.model,
            raw=data,
        )

    async def chat_with_image(
        self,
        prompt: str,
        image_b64: str,
        mime_type: str = "image/jpeg",
        system_prompt: str = "Analyze this screen or camera frame concisely for voice playback.",
    ) -> LLMResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": mime_type, "data": image_b64}},
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 500},
        }
        async with httpx.AsyncClient(timeout=35.0) as client:
            resp = await client.post(url, params={"key": self.api_key}, json=body)
            resp.raise_for_status()
            data = resp.json()
        parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        text = " ".join(p.get("text", "") for p in parts if p.get("text")).strip()
        return LLMResponse(text=text, provider=self.name, model=self.model, raw=data)


class OpenAICompatibleProvider(BaseLLMProvider):
    """Works with OpenAI, Groq, OpenRouter, Together, vLLM, and LM Studio."""

    def __init__(
        self,
        name: str,
        api_key: str,
        model: str,
        base_url: str,
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        openai_tools: list[dict[str, Any]],
        gemini_declarations: list[dict[str, Any]],
        temperature: float = 0.4,
    ) -> LLMResponse:
        formatted: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = msg["role"]
            if role == "assistant" and msg.get("tool_calls"):
                formatted.append(
                    {
                        "role": "assistant",
                        "content": msg.get("content") or None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc.get("arguments", {})),
                                },
                            }
                            for tc in msg["tool_calls"]
                        ],
                    }
                )
            elif role == "tool":
                formatted.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id", "call_0"),
                        "name": msg.get("name", "tool"),
                        "content": msg.get("content", ""),
                    }
                )
            else:
                formatted.append({"role": role, "content": msg.get("content", "")})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted,
            "temperature": temperature,
            "max_tokens": 700,
        }
        if openai_tools:
            payload["tools"] = openai_tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=35.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = (data.get("choices") or [{}])[0].get("message", {})
        tool_calls: list[ToolCallRequest] = []
        for raw_tc in choice.get("tool_calls") or []:
            fn = raw_tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                parsed_args = {}
            tool_calls.append(
                ToolCallRequest(
                    id=raw_tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    name=fn.get("name", ""),
                    arguments=parsed_args,
                )
            )

        return LLMResponse(
            text=(choice.get("content") or "").strip(),
            tool_calls=tool_calls,
            provider=self.name,
            model=self.model,
            raw=data,
        )

    async def chat_with_image(
        self,
        prompt: str,
        image_b64: str,
        mime_type: str = "image/jpeg",
        system_prompt: str = "Analyze this screen or camera frame concisely for voice playback.",
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                    ],
                },
            ],
            "temperature": 0.3,
            "max_tokens": 500,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=35.0) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = ((data.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()
        return LLMResponse(text=text, provider=self.name, model=self.model, raw=data)


class OllamaProvider(BaseLLMProvider):
    """100% local private LLM provider via Ollama `/api/chat`."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2") -> None:
        self.name = "ollama"
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        openai_tools: list[dict[str, Any]],
        gemini_declarations: list[dict[str, Any]],
        temperature: float = 0.4,
    ) -> LLMResponse:
        formatted: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            if msg["role"] == "tool":
                formatted.append({"role": "tool", "content": msg.get("content", "")})
            else:
                formatted.append({"role": msg["role"], "content": msg.get("content", "")})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if openai_tools:
            payload["tools"] = openai_tools

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            if resp.status_code == 400 and "tools" in payload:
                # Some local Ollama models do not support native tool calling; retry plain chat
                payload.pop("tools", None)
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        msg_obj = data.get("message") or {}
        tool_calls: list[ToolCallRequest] = []
        for raw_tc in msg_obj.get("tool_calls") or []:
            fn = raw_tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(
                ToolCallRequest(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=fn.get("name", ""),
                    arguments=args,
                )
            )
        return LLMResponse(
            text=(msg_obj.get("content") or "").strip(),
            tool_calls=tool_calls,
            provider=self.name,
            model=self.model,
            raw=data,
        )

    async def chat_with_image(
        self,
        prompt: str,
        image_b64: str,
        mime_type: str = "image/jpeg",
        system_prompt: str = "Analyze this screen or camera frame concisely for voice playback.",
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt, "images": [image_b64]},
            ],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = ((data.get("message") or {}).get("content") or "").strip()
        return LLMResponse(text=text, provider=self.name, model=self.model, raw=data)


class FreeOpenAIProvider(BaseLLMProvider):
    """Zero-API-key, zero-local-RAM public OpenAI-compatible chat provider (Pollinations anonymous tier).

    Used automatically when no paid cloud API key or local Ollama server is active so
    J.A.R.V.I.S. can hold genuine, context-aware multi-turn conversations out of the box.
    """

    def __init__(
        self,
        model: str = "openai-fast",
        endpoint: str = "https://text.pollinations.ai/openai",
    ) -> None:
        self.name = "free-ai"
        self.model = model
        self.endpoint = endpoint

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        openai_tools: list[dict[str, Any]],
        gemini_declarations: list[dict[str, Any]],
        temperature: float = 0.65,
    ) -> LLMResponse:
        formatted: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = msg.get("role")
            content = str(msg.get("content", "")).strip()
            if role in ("user", "assistant") and content:
                formatted.append({"role": role, "content": content})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted,
            "temperature": max(0.3, min(1.0, float(temperature))),
            "max_tokens": 500,
        }

        last_err: Exception | None = None
        async with httpx.AsyncClient(timeout=18.0, follow_redirects=True) as client:
            for _attempt in range(2):
                try:
                    resp = await client.post(
                        self.endpoint,
                        headers={"Content-Type": "application/json"},
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    msg_obj = (data.get("choices") or [{}])[0].get("message") or {}
                    text = str(msg_obj.get("content") or "").strip()
                    # Strip any leaked <think>...</think> blocks if a reasoning model emits them
                    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
                    if text:
                        return LLMResponse(
                            text=text,
                            tool_calls=[],
                            provider=self.name,
                            model=self.model,
                            raw=data,
                        )
                except Exception as exc:
                    last_err = exc
        if last_err is not None:
            raise last_err
        return LLMResponse(text="", tool_calls=[], provider=self.name, model=self.model)


def resolve_llm_provider(config: JarvisConfig) -> BaseLLMProvider | None:
    """Resolve the active LLM provider based on config and available API keys."""
    choice = config.llm_provider.lower().strip()

    if choice == "offline":
        return None
    if choice == "gemini" and config.gemini_api_key:
        return GeminiProvider(config.gemini_api_key, config.gemini_model)
    if choice == "openai" and config.openai_api_key:
        return OpenAICompatibleProvider("openai", config.openai_api_key, config.openai_model, config.openai_base_url)
    if choice == "groq" and config.groq_api_key:
        return OpenAICompatibleProvider("groq", config.groq_api_key, config.groq_model, "https://api.groq.com/openai/v1")
    if choice == "ollama":
        return OllamaProvider(config.ollama_url, config.ollama_model)

    # Auto-resolution priority: Gemini -> OpenAI -> Groq -> Ollama (probed lazily) -> Offline
    if choice == "auto":
        if config.gemini_api_key:
            return GeminiProvider(config.gemini_api_key, config.gemini_model)
        if config.openai_api_key:
            return OpenAICompatibleProvider("openai", config.openai_api_key, config.openai_model, config.openai_base_url)
        if config.groq_api_key:
            return OpenAICompatibleProvider("groq", config.groq_api_key, config.groq_model, "https://api.groq.com/openai/v1")
        return OllamaProvider(config.ollama_url, config.ollama_model)

    return None
