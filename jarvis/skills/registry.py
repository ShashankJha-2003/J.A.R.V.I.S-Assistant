"""Decorator-based Skill & Tool Registry with automatic LLM JSON Schema export."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from jarvis.config import JarvisConfig
from jarvis.core.events import EventBus, EventType
from jarvis.core.safety import RiskLevel, SafetyGuard
from jarvis.memory.database import JarvisDatabase
from jarvis.memory.vector_store import SemanticVectorStore

logger = logging.getLogger(__name__)


@dataclass
class SkillContext:
    """Shared runtime dependencies injected into every skill handler."""

    config: JarvisConfig
    db: JarvisDatabase
    vector_store: SemanticVectorStore
    event_bus: EventBus
    safety: SafetyGuard
    session_id: str = "default"
    defer_browser_open: bool = False


@dataclass
class SkillResult:
    """Standardized return object from any executed skill."""

    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    confirmation_token: str | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "data": self.data,
            "requires_confirmation": self.requires_confirmation,
            "confirmation_token": self.confirmation_token,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class SkillDefinition:
    name: str
    description: str
    category: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    risk: RiskLevel = RiskLevel.SAFE
    enabled: bool = True
    examples: list[str] = field(default_factory=list)

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_gemini_declaration(self) -> dict[str, Any]:
        """Convert JSON Schema parameters to Google Gemini functionDeclarations schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": _convert_schema_for_gemini(self.parameters),
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "risk": self.risk.value,
            "enabled": self.enabled,
            "examples": self.examples,
            "parameters": self.parameters,
        }


def _convert_schema_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    type_map = {
        "object": "OBJECT",
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
    }
    out: dict[str, Any] = {}
    if "type" in schema:
        out["type"] = type_map.get(str(schema["type"]).lower(), "STRING")
    if "description" in schema:
        out["description"] = schema["description"]
    if "enum" in schema:
        out["enum"] = list(schema["enum"])
    if "properties" in schema and isinstance(schema["properties"], dict):
        out["properties"] = {
            k: _convert_schema_for_gemini(v) for k, v in schema["properties"].items()
        }
    if "items" in schema and isinstance(schema["items"], dict):
        out["items"] = _convert_schema_for_gemini(schema["items"])
    if "required" in schema and schema["required"]:
        out["required"] = list(schema["required"])
    return out


class SkillRegistry:
    """Central registry managing J.A.R.V.I.S. voice-callable tools and plugins."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        category: str,
        parameters: dict[str, Any] | None = None,
        risk: RiskLevel = RiskLevel.SAFE,
        examples: list[str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register an async or sync skill function."""
        schema = parameters or {"type": "object", "properties": {}, "required": []}

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._skills[name] = SkillDefinition(
                name=name,
                description=description,
                category=category,
                parameters=schema,
                handler=func,
                risk=risk,
                examples=examples or [],
            )
            return func

        return decorator

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(name)

    def list_skills(self, enabled_only: bool = False) -> list[SkillDefinition]:
        skills = list(self._skills.values())
        if enabled_only:
            skills = [s for s in skills if s.enabled]
        return sorted(skills, key=lambda s: (s.category, s.name))

    def set_enabled(self, name: str, enabled: bool) -> bool:
        skill = self._skills.get(name)
        if not skill:
            return False
        skill.enabled = enabled
        return True

    def _filter_skills(
        self,
        *,
        categories: set[str] | tuple[str, ...] | None = None,
        include_names: set[str] | None = None,
        exclude_names: set[str] | None = None,
    ) -> list[SkillDefinition]:
        skills = self.list_skills(enabled_only=True)
        cat_set = set(categories) if categories is not None else None
        out: list[SkillDefinition] = []
        for s in skills:
            if exclude_names and s.name in exclude_names:
                continue
            if cat_set is not None and s.category not in cat_set:
                if not (include_names and s.name in include_names):
                    continue
            elif include_names is not None and cat_set is None and s.name not in include_names:
                continue
            out.append(s)
        return out

    def openai_tools(
        self,
        *,
        categories: set[str] | tuple[str, ...] | None = None,
        include_names: set[str] | None = None,
        exclude_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            s.to_openai_tool()
            for s in self._filter_skills(
                categories=categories,
                include_names=include_names,
                exclude_names=exclude_names,
            )
        ]

    def gemini_declarations(
        self,
        *,
        categories: set[str] | tuple[str, ...] | None = None,
        include_names: set[str] | None = None,
        exclude_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            s.to_gemini_declaration()
            for s in self._filter_skills(
                categories=categories,
                include_names=include_names,
                exclude_names=exclude_names,
            )
        ]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        ctx: SkillContext,
        bypass_confirmation: bool = False,
    ) -> SkillResult:
        """Execute a registered skill with safety checks, events, and audit logging."""
        skill = self._skills.get(name)
        if not skill:
            return SkillResult(ok=False, message=f"Unknown skill '{name}'.")
        if not skill.enabled:
            return SkillResult(ok=False, message=f"Skill '{name}' is currently disabled.")

        safe_args = arguments or {}

        # Safety confirmation gate for dangerous system commands
        if not bypass_confirmation and ctx.safety.needs_confirmation(skill.risk):
            desc = f"Execute {skill.name} with {safe_args}" if safe_args else f"Execute {skill.name}"
            pending = ctx.safety.request_confirmation(
                skill_name=skill.name,
                arguments=safe_args,
                description=desc,
            )
            await ctx.event_bus.publish(
                EventType.CONFIRMATION_REQUESTED,
                pending.to_dict(),
                source="safety",
            )
            ctx.db.log_audit(skill.name, safe_args, "pending_confirmation", desc, 0.0)
            return SkillResult(
                ok=True,
                message=f"Confirmation required before running '{skill.name}'. Please say 'confirm' to proceed or 'cancel' to abort.",
                data=pending.to_dict(),
                requires_confirmation=True,
                confirmation_token=pending.token,
            )

        await ctx.event_bus.publish(
            EventType.TOOL_CALLED,
            {"skill": skill.name, "category": skill.category, "arguments": safe_args},
            source="skills",
        )

        started = time.perf_counter()
        try:
            sig = inspect.signature(skill.handler)
            valid_kwargs: dict[str, Any] = {}
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            for k, v in safe_args.items():
                if accepts_var_kw or (k in sig.parameters and k != "ctx"):
                    valid_kwargs[k] = v

            if inspect.iscoroutinefunction(skill.handler):
                raw = await skill.handler(ctx, **valid_kwargs)
            else:
                raw = await asyncio.to_thread(skill.handler, ctx, **valid_kwargs)

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if isinstance(raw, SkillResult):
                result = raw
                result.latency_ms = elapsed_ms
            elif isinstance(raw, dict):
                result = SkillResult(
                    ok=bool(raw.get("ok", True)),
                    message=str(raw.get("message", "Completed.")),
                    data=raw,
                    latency_ms=elapsed_ms,
                )
            else:
                result = SkillResult(ok=True, message=str(raw), latency_ms=elapsed_ms)

            ctx.db.log_audit(
                skill.name,
                safe_args,
                "ok" if result.ok else "failed",
                result.message,
                elapsed_ms,
            )
            await ctx.event_bus.publish(
                EventType.TOOL_RESULT,
                {
                    "skill": skill.name,
                    "ok": result.ok,
                    "message": result.message,
                    "data": result.data,
                    "latency_ms": round(elapsed_ms, 2),
                },
                source="skills",
            )
            return result
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.exception("Skill '%s' raised an error: %s", skill.name, exc)
            err_msg = f"Failed to execute {skill.name}: {exc}"
            ctx.db.log_audit(skill.name, safe_args, "error", err_msg, elapsed_ms)
            await ctx.event_bus.publish(
                EventType.TOOL_RESULT,
                {"skill": skill.name, "ok": False, "message": err_msg, "latency_ms": round(elapsed_ms, 2)},
                source="skills",
            )
            return SkillResult(ok=False, message=err_msg, latency_ms=elapsed_ms)
