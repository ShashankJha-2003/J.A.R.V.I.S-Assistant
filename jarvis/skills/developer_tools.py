"""Developer, Workspace Workflow, Persona Modulation & High-Impact Guarded Skills."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import psutil

from jarvis.ai.supervisor import PERSONA_PROFILES, get_persona
from jarvis.config import PROJECT_ROOT
from jarvis.core.safety import RiskLevel
from jarvis.skills.registry import SkillContext, SkillRegistry, SkillResult


def _summarize_pytest_output(stdout: str, stderr: str, returncode: int) -> tuple[str, dict]:
    """Convert raw pytest output into a crisp spoken engineering summary (zero log dumping)."""
    combined = (stdout or "") + "\n" + (stderr or "")
    passed = int(m.group(1)) if (m := re.search(r"(\d+)\s+passed", combined)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+)\s+failed", combined)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+)\s+error", combined)) else 0
    failed_names = re.findall(r"FAILED\s+([\w/\\.\-]+::[\w\[\]\-]+)", combined)[:5]

    if returncode == 0 and failed == 0 and errors == 0:
        msg = f"All {passed} tests passed cleanly with zero failures!"
    elif failed or errors:
        names = ", ".join(n.split("::")[-1] for n in failed_names)
        detail = f" Failing tests: {names}." if names else ""
        msg = f"Test run finished with {passed} passed, {failed} failed, and {errors} errors.{detail}"
    else:
        tail = [ln.strip() for ln in combined.splitlines() if ln.strip()][-1:] or ["Test runner exited with errors."]
        msg = f"Tests exited with code {returncode}: {tail[0][:160]}"

    return msg, {
        "returncode": returncode,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "failed_tests": failed_names,
    }


def register_developer_skills(registry: SkillRegistry) -> None:
    @registry.register(
        name="search_workspace_files",
        description="Search for files by filename pattern inside the current project workspace.",
        category="developer",
        risk=RiskLevel.SAFE,
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Filename substring or glob pattern (e.g., '*.py' or 'voice').",
                }
            },
            "required": ["pattern"],
        },
        examples=["find files matching voice", "search python files in workspace"],
    )
    def search_workspace_files(ctx: SkillContext, pattern: str) -> SkillResult:
        pat = pattern.strip().lower()
        ignored = {".git", ".venv", "venv", "__pycache__", "node_modules"}
        hits: list[str] = []
        for path in PROJECT_ROOT.rglob("*"):
            if any(part in ignored for part in path.parts):
                continue
            if path.is_file():
                rel = path.relative_to(PROJECT_ROOT).as_posix()
                if pat in path.name.lower() or path.match(pattern):
                    hits.append(rel)
                    if len(hits) >= 20:
                        break
        if not hits:
            return SkillResult(ok=True, message=f"No workspace files matched '{pattern}'.", data={"files": []})
        preview = ", ".join(hits[:8])
        return SkillResult(
            ok=True,
            message=f"Found {len(hits)} file(s): {preview}.",
            data={"files": hits},
        )

    @registry.register(
        name="git_status",
        description="Check Git branch and working tree status in the current workspace.",
        category="developer",
        risk=RiskLevel.SAFE,
        examples=["check git status", "what git branch am i on"],
    )
    def git_status(ctx: SkillContext) -> SkillResult:
        try:
            proc = subprocess.run(
                ["git", "status", "--short", "--branch"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=6,
            )
            if proc.returncode != 0:
                return SkillResult(ok=False, message="This directory is not initialized as a Git repository.")
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            branch = lines[0].replace("## ", "") if lines else "unknown"
            changed = len(lines) - 1 if len(lines) > 1 else 0
            msg = f"Git branch is '{branch}' with {changed} modified or untracked item(s)."
            return SkillResult(ok=True, message=msg, data={"branch": branch, "changed_count": changed, "raw": proc.stdout})
        except Exception as exc:
            return SkillResult(ok=False, message=f"Git status failed: {exc}")

    @registry.register(
        name="git_workflow",
        description="Perform Git workspace actions: list branches, switch branches, show recent commits, or summarize diffs.",
        category="developer",
        risk=RiskLevel.MODERATE,
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["branches", "switch", "log", "diff_summary"]},
                "branch": {"type": "string", "description": "Target branch name when action is 'switch'."},
            },
            "required": ["action"],
        },
        examples=["list git branches", "switch branch to main", "show git log"],
    )
    def git_workflow(ctx: SkillContext, action: str, branch: str = "") -> SkillResult:
        act = action.strip().lower()
        try:
            if act == "branches":
                p = subprocess.run(["git", "branch", "--list"], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=6)
                if p.returncode != 0:
                    return SkillResult(ok=False, message="Not a Git repository.")
                items = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
                return SkillResult(ok=True, message=f"Git branches: {', '.join(items[:8])}.", data={"branches": items})
            if act == "switch":
                target = branch.strip()
                if not target or not re.match(r"^[A-Za-z0-9._/\-]+$", target):
                    return SkillResult(ok=False, message="Please provide a valid Git branch name.")
                p = subprocess.run(["git", "switch", target], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=8)
                if p.returncode != 0:
                    return SkillResult(ok=False, message=f"Could not switch to branch '{target}': {(p.stderr or p.stdout).strip()[:120]}")
                return SkillResult(ok=True, message=f"Switched Git branch to '{target}'.", data={"branch": target})
            if act == "log":
                p = subprocess.run(["git", "log", "-n", "5", "--oneline"], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=6)
                if p.returncode != 0:
                    return SkillResult(ok=False, message="No Git commits found in this workspace.")
                commits = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
                return SkillResult(ok=True, message="Recent commits: " + "; ".join(commits[:3]) + ".", data={"commits": commits})
            if act == "diff_summary":
                p = subprocess.run(["git", "diff", "--stat"], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=6)
                lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
                summary = lines[-1] if lines else "Working tree has no unstaged diffs."
                return SkillResult(ok=True, message=summary, data={"diff_stat": p.stdout})
        except Exception as exc:
            return SkillResult(ok=False, message=f"Git workflow error: {exc}")
        return SkillResult(ok=False, message=f"Unsupported Git workflow action '{action}'.")

    @registry.register(
        name="run_project_tests",
        description="Run the project pytest suite and speak back a crisp pass/fail summary instead of raw logs.",
        category="developer",
        risk=RiskLevel.SAFE,
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Optional specific test file or pytest -k filter."},
            },
        },
        examples=["run tests", "run pytest", "check if unit tests pass"],
    )
    def run_project_tests(ctx: SkillContext, target: str = "") -> SkillResult:
        cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short"]
        cleaned = target.strip()
        if cleaned and re.match(r"^[A-Za-z0-9_./\\\-]+$", cleaned):
            cmd.append(cleaned)
        try:
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=45)
            msg, data = _summarize_pytest_output(proc.stdout, proc.stderr, proc.returncode)
            return SkillResult(ok=(proc.returncode == 0), message=msg, data=data)
        except subprocess.TimeoutExpired:
            return SkillResult(ok=False, message="Test run timed out after 45 seconds.")
        except Exception as exc:
            return SkillResult(ok=False, message=f"Could not run pytest: {exc}")

    @registry.register(
        name="index_codebase_rag",
        description="Index project source code, notes, markdown, and PDFs into J.A.R.V.I.S.'s semantic RAG vector store.",
        category="developer",
        risk=RiskLevel.SAFE,
        examples=["index my codebase", "index workspace docs into rag"],
    )
    def index_codebase_rag(ctx: SkillContext) -> SkillResult:
        summary = ctx.vector_store.ingest_directory(PROJECT_ROOT / "jarvis", include_code=True)
        kb = ctx.vector_store.ingest_directory(ctx.config.knowledge_dir, include_code=False)
        files = summary["files_indexed"] + kb["files_indexed"]
        chunks = summary["total_chunks"] + kb["total_chunks"]
        return SkillResult(
            ok=True,
            message=f"Indexed {files} codebase and knowledge files ({chunks} semantic chunks) into RAG memory.",
            data={"files_indexed": files, "total_chunks": chunks},
        )

    @registry.register(
        name="run_shell_command",
        description="Execute a shell command in the workspace (HIGH-IMPACT: always requires explicit human confirmation).",
        category="developer",
        risk=RiskLevel.DANGEROUS,
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute after explicit user approval."},
            },
            "required": ["command"],
        },
        examples=["run shell command python --version"],
    )
    def run_shell_command(ctx: SkillContext, command: str) -> SkillResult:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=20,
        )
        out = (proc.stdout or proc.stderr or "").strip()
        preview = out[:240] if out else f"Command exited with code {proc.returncode}."
        return SkillResult(
            ok=(proc.returncode == 0),
            message=f"Shell command finished (code {proc.returncode}): {preview}",
            data={"returncode": proc.returncode, "output": out[:1500]},
        )

    @registry.register(
        name="delete_workspace_file",
        description="Delete a file inside the workspace (HIGH-IMPACT: always requires explicit human confirmation).",
        category="developer",
        risk=RiskLevel.DANGEROUS,
        parameters={
            "type": "object",
            "properties": {
                "relative_path": {"type": "string", "description": "Workspace-relative file path to delete."},
            },
            "required": ["relative_path"],
        },
        examples=["delete file scratch.txt"],
    )
    def delete_workspace_file(ctx: SkillContext, relative_path: str) -> SkillResult:
        target = (PROJECT_ROOT / relative_path.strip()).resolve()
        if PROJECT_ROOT.resolve() not in target.parents:
            return SkillResult(ok=False, message="Blocked: file deletion is restricted strictly to workspace files.")
        if not target.exists() or not target.is_file():
            return SkillResult(ok=False, message=f"File '{relative_path}' does not exist.")
        target.unlink()
        return SkillResult(ok=True, message=f"Deleted workspace file '{relative_path}'.", data={"deleted": relative_path})

    @registry.register(
        name="kill_process",
        description="Terminate a running desktop process by name (HIGH-IMPACT: always requires explicit human confirmation).",
        category="system",
        risk=RiskLevel.DANGEROUS,
        parameters={
            "type": "object",
            "properties": {
                "process_name": {"type": "string", "description": "Process executable name (e.g., 'notepad.exe')."},
            },
            "required": ["process_name"],
        },
        examples=["kill process notepad.exe"],
    )
    def kill_process(ctx: SkillContext, process_name: str) -> SkillResult:
        needle = process_name.strip().lower()
        protected = {"explorer.exe", "csrss.exe", "wininit.exe", "lsass.exe", "system", "python.exe"}
        if needle in protected:
            return SkillResult(ok=False, message=f"Blocked: '{process_name}' is a protected system or assistant process.")
        killed = 0
        for p in psutil.process_iter(["pid", "name"]):
            try:
                pname = (p.info.get("name") or "").lower()
                if pname == needle or pname == f"{needle}.exe":
                    p.terminate()
                    killed += 1
            except Exception:
                continue
        if killed == 0:
            return SkillResult(ok=False, message=f"No running process matched '{process_name}'.")
        return SkillResult(ok=True, message=f"Terminated {killed} instance(s) of '{process_name}'.", data={"killed": killed})

    @registry.register(
        name="switch_persona",
        description="Switch J.A.R.V.I.S.'s persona, tone, and TTS voice modulation (calm, crisp, witty, analytical).",
        category="developer",
        risk=RiskLevel.SAFE,
        parameters={
            "type": "object",
            "properties": {
                "persona": {"type": "string", "enum": list(PERSONA_PROFILES.keys())},
            },
            "required": ["persona"],
        },
        examples=["switch persona to witty", "be more analytical", "crisp mode", "calm mode"],
    )
    def switch_persona(ctx: SkillContext, persona: str) -> SkillResult:
        prof = get_persona(persona)
        ctx.config.apply_runtime_updates({"persona": prof.name, "edge_tts_voice": prof.edge_voice})
        return SkillResult(
            ok=True,
            message=f"Switched to {prof.label} mode — {prof.description}",
            data=prof.to_dict(),
        )

    @registry.register(
        name="jarvis_capabilities",
        description="List J.A.R.V.I.S.'s installed voice skills, capabilities, and sample voice commands.",
        category="developer",
        risk=RiskLevel.SAFE,
        examples=["what can you do", "help", "list your skills"],
    )
    def jarvis_capabilities(ctx: SkillContext) -> SkillResult:
        stats = ctx.db.memory_stats()
        msg = (
            "I combine multi-agent supervision (System/OS, Research/Web/Vision, Code/Workspace, and Memory/RAG), "
            "screen & camera vision, semantic PDF & codebase RAG, MCP tool plugins, proactive battery & file alerts, "
            "smart home IoT control, pytest & Git dev workflows, switchable personas (calm, crisp, witty, analytical), "
            "and human-in-the-loop safety confirmations."
        )
        return SkillResult(ok=True, message=msg, data={"memory_stats": stats})
