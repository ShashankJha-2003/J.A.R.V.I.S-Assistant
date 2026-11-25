"""Productivity skills: Timers, Reminders, Notes, To-Do Tasks, Safe Calculator, and Time."""

from __future__ import annotations

import ast
import datetime as dt
import math
import operator
import re
import time
from typing import Any

from jarvis.core.safety import RiskLevel
from jarvis.skills.registry import SkillContext, SkillRegistry, SkillResult

# Safe AST math evaluator (no eval())
_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MATH_FUNCS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
}
_MATH_CONSTS: dict[str, float] = {"pi": math.pi, "e": math.e}


def safe_calculate(expression: str) -> float:
    """Safely evaluate arithmetic and basic scientific math expressions via AST."""
    expr = (
        expression.lower()
        .replace("^", "**")
        .replace("×", "*")
        .replace("÷", "/")
        .replace(" plus ", " + ")
        .replace(" minus ", " - ")
        .replace(" times ", " * ")
        .replace(" multiplied by ", " * ")
        .replace(" divided by ", " / ")
        .strip()
    )
    tree = ast.parse(expr, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in _MATH_CONSTS:
            return _MATH_CONSTS[node.id]
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return float(_UNARY_OPS[type(node.op)](_eval(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("Exponent too large.")
            return float(_BIN_OPS[type(node.op)](left, right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _MATH_FUNCS:
            args = [_eval(arg) for arg in node.args]
            return float(_MATH_FUNCS[node.func.id](*args))
        raise ValueError(f"Unsupported expression element: {ast.dump(node)}")

    return _eval(tree)


def parse_duration_seconds(text: str) -> int:
    """Parse durations like '10 seconds', '5 minutes', '1 hour 30 minutes', '90s'."""
    total = 0
    for amount, unit in re.findall(
        r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)\b",
        text.lower(),
    ):
        val = float(amount)
        if unit.startswith("h"):
            total += int(val * 3600)
        elif unit.startswith("m"):
            total += int(val * 60)
        else:
            total += int(val)
    if total == 0 and text.strip().isdigit():
        total = int(text.strip())
    return total


def _format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if mins:
        parts.append(f"{mins} minute{'s' if mins != 1 else ''}")
    if secs:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")
    return " and ".join(parts)


def register_productivity_skills(registry: SkillRegistry) -> None:
    @registry.register(
        name="get_time_and_date",
        description="Get the current local time, day of the week, and calendar date.",
        category="productivity",
        risk=RiskLevel.SAFE,
        examples=["what time is it", "what is today's date"],
    )
    def get_time_and_date(ctx: SkillContext) -> SkillResult:
        now = dt.datetime.now()
        msg = f"It is currently {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}."
        return SkillResult(
            ok=True,
            message=msg,
            data={"time": now.strftime("%H:%M:%S"), "date": now.strftime("%Y-%m-%d"), "weekday": now.strftime("%A")},
        )

    @registry.register(
        name="calculate",
        description="Evaluate arithmetic or scientific math expressions safely (supports +, -, *, /, ^, sqrt, sin, cos, log, pi).",
        category="productivity",
        risk=RiskLevel.SAFE,
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to calculate."}
            },
            "required": ["expression"],
        },
        examples=["calculate 45 times 18", "what is sqrt(144) + 25"],
    )
    def calculate(ctx: SkillContext, expression: str) -> SkillResult:
        value = safe_calculate(expression)
        formatted = str(int(value)) if value.is_integer() else f"{value:.6g}"
        return SkillResult(
            ok=True,
            message=f"The result of {expression} is {formatted}.",
            data={"expression": expression, "result": value, "formatted": formatted},
        )

    @registry.register(
        name="set_reminder",
        description="Set a timed reminder or countdown timer that proactively alerts and speaks when time is up.",
        category="productivity",
        risk=RiskLevel.SAFE,
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "What to remind the user about."},
                "duration_seconds": {
                    "type": "integer",
                    "description": "Delay in seconds from now.",
                },
                "duration_text": {
                    "type": "string",
                    "description": "Natural duration string such as '5 minutes' or '30 seconds'.",
                },
            },
            "required": ["message"],
        },
        examples=["remind me in 5 minutes to stretch", "set a timer for 30 seconds"],
    )
    def set_reminder(
        ctx: SkillContext,
        message: str,
        duration_seconds: int | None = None,
        duration_text: str = "",
    ) -> SkillResult:
        secs = int(duration_seconds) if duration_seconds and duration_seconds > 0 else parse_duration_seconds(duration_text)
        if secs <= 0:
            return SkillResult(ok=False, message="Please specify a positive duration, such as '2 minutes' or '30 seconds'.")
        trigger_at = time.time() + secs
        rem = ctx.db.add_reminder(message=message or "Timer finished", trigger_at=trigger_at)
        friendly = _format_duration(secs)
        return SkillResult(
            ok=True,
            message=f"Timer set for {friendly}: '{rem['message']}'.",
            data=rem,
        )

    @registry.register(
        name="list_reminders",
        description="List all pending timers and scheduled reminders.",
        category="productivity",
        risk=RiskLevel.SAFE,
        examples=["show my reminders", "what timers are active"],
    )
    def list_reminders(ctx: SkillContext) -> SkillResult:
        reminders = ctx.db.list_reminders(status="pending")
        if not reminders:
            return SkillResult(ok=True, message="You have no active reminders or timers.", data={"reminders": []})
        now = time.time()
        parts = []
        for r in reminders[:8]:
            remaining = max(0, int(r["trigger_at"] - now))
            parts.append(f"#{r['id']} '{r['message']}' in {_format_duration(remaining)}")
        return SkillResult(
            ok=True,
            message="Active reminders: " + "; ".join(parts) + ".",
            data={"reminders": reminders},
        )

    @registry.register(
        name="manage_notes",
        description="Create, search, list, or delete persistent personal notes.",
        category="productivity",
        risk=RiskLevel.SAFE,
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "delete"]},
                "title": {"type": "string", "description": "Short note title (for add)."},
                "content": {"type": "string", "description": "Body of the note (for add) or search query (for list)."},
                "note_id": {"type": "integer", "description": "Note ID (for delete)."},
            },
            "required": ["action"],
        },
        examples=["take a note: finish voice assistant demo", "show my notes"],
    )
    def manage_notes(
        ctx: SkillContext,
        action: str,
        title: str = "",
        content: str = "",
        note_id: int | None = None,
    ) -> SkillResult:
        act = action.lower().strip()
        if act == "add":
            body = (content or title).strip()
            if not body:
                return SkillResult(ok=False, message="Please provide note content to save.")
            note_title = title.strip() or (body[:42] + ("..." if len(body) > 42 else ""))
            note = ctx.db.add_note(title=note_title, content=body)
            return SkillResult(ok=True, message=f"Saved note #{note['id']}: {note['title']}.", data={"note": note})
        if act == "list":
            notes = ctx.db.list_notes(query=content or title)
            if not notes:
                return SkillResult(ok=True, message="No notes found.", data={"notes": []})
            summary = "; ".join(f"#{n['id']} {n['title']}" for n in notes[:6])
            return SkillResult(ok=True, message=f"You have {len(notes)} note(s): {summary}.", data={"notes": notes})
        if act == "delete" and note_id is not None:
            deleted = ctx.db.delete_note(int(note_id))
            return SkillResult(
                ok=deleted,
                message=f"Deleted note #{note_id}." if deleted else f"Note #{note_id} was not found.",
            )
        return SkillResult(ok=False, message="Invalid note action or missing note_id.")

    @registry.register(
        name="manage_tasks",
        description="Manage your to-do task list: add a task, list tasks, complete a task, or delete a task.",
        category="productivity",
        risk=RiskLevel.SAFE,
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "complete", "delete"]},
                "title": {"type": "string", "description": "Task description (for add)."},
                "priority": {"type": "string", "enum": ["low", "normal", "high"]},
                "task_id": {"type": "integer", "description": "Task ID (for complete/delete)."},
            },
            "required": ["action"],
        },
        examples=["add task prepare presentation", "show my todo list", "complete task 1"],
    )
    def manage_tasks(
        ctx: SkillContext,
        action: str,
        title: str = "",
        priority: str = "normal",
        task_id: int | None = None,
    ) -> SkillResult:
        act = action.lower().strip()
        if act == "add":
            if not title.strip():
                return SkillResult(ok=False, message="Please provide a task description.")
            task = ctx.db.add_task(title=title, priority=priority)
            return SkillResult(ok=True, message=f"Added task #{task['id']}: {task['title']}.", data={"task": task})
        if act == "list":
            tasks = ctx.db.list_tasks(include_completed=False)
            if not tasks:
                return SkillResult(ok=True, message="Your to-do list is completely clear.", data={"tasks": []})
            summary = "; ".join(f"#{t['id']} {t['title']}" for t in tasks[:8])
            return SkillResult(ok=True, message=f"Open tasks: {summary}.", data={"tasks": tasks})
        if act == "complete" and task_id is not None:
            ok = ctx.db.complete_task(int(task_id))
            return SkillResult(ok=ok, message=f"Marked task #{task_id} as completed." if ok else f"Task #{task_id} not found.")
        if act == "delete" and task_id is not None:
            ok = ctx.db.delete_task(int(task_id))
            return SkillResult(ok=ok, message=f"Deleted task #{task_id}." if ok else f"Task #{task_id} not found.")
        return SkillResult(ok=False, message="Invalid task action or missing task_id.")
