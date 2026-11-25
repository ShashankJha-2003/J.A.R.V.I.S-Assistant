"""Model Context Protocol (MCP) Client & Tool Bridge (JSON-RPC 2.0 over stdio or HTTP).

Allows J.A.R.V.I.S. to connect to standardized external MCP servers defined in
`data/mcp_servers.json` (or added dynamically via API/voice), discover their tools,
and expose them cleanly in SkillRegistry with safety guardrails.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from jarvis.core.safety import RiskLevel
from jarvis.skills.registry import SkillContext, SkillRegistry, SkillResult

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    name: str
    transport: str = "http"  # "http" | "stdio"
    url: str = ""
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "transport": self.transport,
            "url": self.url,
            "command": self.command,
            "args": list(self.args),
            "enabled": self.enabled,
        }


class MCPManager:
    """Discovers and invokes tools from configured MCP JSON-RPC 2.0 servers."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = Path(config_path)
        self._servers: dict[str, MCPServerConfig] = {}
        self._discovered_tools: dict[str, list[dict[str, Any]]] = {}
        self._rpc_id = 0
        self.load_config()

    def _next_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id

    def load_config(self) -> None:
        self._servers.clear()
        if not self.config_path.exists():
            return
        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
            entries = raw.get("mcpServers") or raw.get("servers") or {}
            if isinstance(entries, dict):
                for name, cfg in entries.items():
                    if not isinstance(cfg, dict):
                        continue
                    transport = str(cfg.get("transport") or ("stdio" if cfg.get("command") else "http")).lower()
                    self._servers[name] = MCPServerConfig(
                        name=name,
                        transport=transport,
                        url=str(cfg.get("url") or "").strip(),
                        command=str(cfg.get("command") or "").strip(),
                        args=[str(a) for a in (cfg.get("args") or [])],
                        env={str(k): str(v) for k, v in (cfg.get("env") or {}).items()},
                        enabled=bool(cfg.get("enabled", True)),
                    )
        except Exception as exc:
            logger.warning("Failed to parse MCP config %s: %s", self.config_path, exc)

    def save_config(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"mcpServers": {name: s.to_dict() for name, s in self._servers.items()}}
        self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add_server(self, cfg: MCPServerConfig) -> None:
        self._servers[cfg.name] = cfg
        self.save_config()

    def remove_server(self, name: str) -> bool:
        if name not in self._servers:
            return False
        self._servers.pop(name, None)
        self._discovered_tools.pop(name, None)
        self.save_config()
        return True

    def status(self) -> dict[str, Any]:
        return {
            "protocol": "MCP JSON-RPC 2.0",
            "config_path": str(self.config_path),
            "servers": [
                {
                    **s.to_dict(),
                    "tools_count": len(self._discovered_tools.get(s.name, [])),
                    "tools": [t.get("name") for t in self._discovered_tools.get(s.name, [])],
                }
                for s in self._servers.values()
            ],
            "total_tools": sum(len(v) for v in self._discovered_tools.values()),
        }

    async def _rpc_http(self, server: MCPServerConfig, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        body = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or {},
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                server.url,
                json=body,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        if "error" in data and data["error"]:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result") or {}

    async def _rpc_stdio(self, server: MCPServerConfig, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not server.command:
            raise ValueError(f"MCP stdio server '{server.name}' missing command.")
        proc = await asyncio.create_subprocess_exec(
            server.command,
            *server.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            init_req = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "JARVIS", "version": "2.0"},
                    },
                }
            )
            call_id = self._next_id()
            call_req = json.dumps(
                {"jsonrpc": "2.0", "id": call_id, "method": method, "params": params or {}}
            )
            stdin_bytes = (init_req + "\n" + call_req + "\n").encode("utf-8")
            stdout, _ = await asyncio.wait_for(proc.communicate(stdin_bytes), timeout=10.0)
            for line in stdout.decode("utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") == call_id:
                    if "error" in msg and msg["error"]:
                        raise RuntimeError(f"MCP stdio error: {msg['error']}")
                    return msg.get("result") or {}
            return {}
        finally:
            if proc.returncode is None:
                proc.kill()

    async def _rpc(self, server: MCPServerConfig, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if server.transport == "stdio":
            return await self._rpc_stdio(server, method, params)
        return await self._rpc_http(server, method, params)

    async def discover_and_register(self, registry: SkillRegistry) -> dict[str, Any]:
        """Connect to all enabled MCP servers, list their tools, and register them as JARVIS skills."""
        registered: list[str] = []
        errors: dict[str, str] = {}
        for name, server in self._servers.items():
            if not server.enabled:
                continue
            try:
                res = await self._rpc(server, "tools/list", {})
                tools = res.get("tools") or []
                self._discovered_tools[name] = tools
                for tool in tools:
                    t_name = str(tool.get("name") or "").strip()
                    if not t_name:
                        continue
                    skill_name = f"mcp_{name}_{t_name}".replace("-", "_").replace(" ", "_")
                    desc = str(tool.get("description") or f"MCP tool {t_name} from server {name}.")
                    schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
                    # High-impact verbs on external MCP tools get human-in-the-loop confirmation!
                    low_t = t_name.lower()
                    risk = (
                        RiskLevel.DANGEROUS
                        if any(v in low_t for v in ("delete", "remove", "drop", "send", "exec", "shell", "kill"))
                        else RiskLevel.MODERATE
                    )

                    def _make_handler(srv: MCPServerConfig, raw_tool: str):
                        async def _invoke(ctx: SkillContext, **kwargs: Any) -> SkillResult:
                            out = await self._rpc(srv, "tools/call", {"name": raw_tool, "arguments": kwargs})
                            content = out.get("content") or []
                            texts = [
                                str(item.get("text", ""))
                                for item in content
                                if isinstance(item, dict) and item.get("text")
                            ]
                            msg = " ".join(texts).strip() or json.dumps(out, default=str)[:400]
                            return SkillResult(ok=not out.get("isError", False), message=msg, data={"mcp_server": srv.name, "tool": raw_tool, "raw": out})

                        return _invoke

                    registry.register(
                        name=skill_name,
                        description=f"[MCP:{name}] {desc}",
                        category="mcp",
                        risk=risk,
                        parameters=schema,
                    )(_make_handler(server, t_name))
                    registered.append(skill_name)
            except Exception as exc:
                errors[name] = str(exc)
                logger.debug("MCP server '%s' discovery skipped: %s", name, exc)
        return {"registered": registered, "errors": errors}
