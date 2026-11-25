"""Smart Home & IoT Control Skill Pack (Home Assistant REST, ESP32/RPi HTTP, MQTT, & Local Registry)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from jarvis.core.safety import RiskLevel
from jarvis.skills.registry import SkillContext, SkillRegistry, SkillResult

try:
    import paho.mqtt.publish as mqtt_publish  # type: ignore
except ImportError:
    mqtt_publish = None


DEFAULT_IOT_DEVICES: dict[str, dict[str, Any]] = {
    "desk lamp": {"domain": "light", "entity_id": "light.desk_lamp", "state": "off", "brightness": 80, "endpoint": "", "mqtt_topic": "home/desk_lamp/set"},
    "room lights": {"domain": "light", "entity_id": "light.room_lights", "state": "off", "brightness": 100, "endpoint": "", "mqtt_topic": "home/room_lights/set"},
    "fan": {"domain": "fan", "entity_id": "fan.room_fan", "state": "off", "brightness": None, "endpoint": "", "mqtt_topic": "home/fan/set"},
    "smart plug": {"domain": "switch", "entity_id": "switch.smart_plug", "state": "off", "brightness": None, "endpoint": "", "mqtt_topic": "home/smart_plug/set"},
    "ac": {"domain": "climate", "entity_id": "climate.room_ac", "state": "off", "brightness": None, "endpoint": "", "mqtt_topic": "home/ac/set"},
}


def _iot_store_path(ctx: SkillContext) -> Path:
    return ctx.config.data_dir / "iot_devices.json"


def _load_devices(ctx: SkillContext) -> dict[str, dict[str, Any]]:
    path = _iot_store_path(ctx)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {k: dict(v) for k, v in DEFAULT_IOT_DEVICES.items()}


def _save_devices(ctx: SkillContext, devices: dict[str, dict[str, Any]]) -> None:
    path = _iot_store_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(devices, indent=2), encoding="utf-8")


def register_smart_home_skills(registry: SkillRegistry) -> None:
    @registry.register(
        name="control_smart_device",
        description=(
            "Control smart home & IoT devices (lights, desk lamp, smart plugs, fans, ESP32/Raspberry Pi boards) "
            "via Home Assistant REST API, local HTTP webhooks, MQTT, or built-in device state registry."
        ),
        category="iot",
        risk=RiskLevel.MODERATE,
        parameters={
            "type": "object",
            "properties": {
                "device": {"type": "string", "description": "Device name (e.g. 'desk lamp', 'room lights', 'fan', 'smart plug')."},
                "action": {"type": "string", "enum": ["on", "off", "toggle", "dim", "status"], "description": "Power or dimmer action."},
                "level": {"type": "integer", "description": "Optional brightness/speed percentage (0-100) when dimming."},
            },
            "required": ["device", "action"],
        },
        examples=["turn on desk lamp", "turn off room lights", "dim room lights to 40 percent"],
    )
    async def control_smart_device(ctx: SkillContext, device: str, action: str = "toggle", level: int | None = None) -> SkillResult:
        devices = _load_devices(ctx)
        key = device.strip().lower()
        dev = devices.get(key)
        if dev is None:
            # Fuzzy match partial device name
            for k, v in devices.items():
                if key in k or k in key:
                    key, dev = k, v
                    break
        if dev is None:
            dev = {"domain": "switch", "entity_id": f"switch.{key.replace(' ', '_')}", "state": "off", "brightness": 100, "endpoint": "", "mqtt_topic": f"home/{key.replace(' ', '_')}/set"}
            devices[key] = dev

        act = action.strip().lower()
        if act == "status":
            lvl = f" at {dev['brightness']}%" if dev.get("brightness") is not None and dev.get("state") == "on" else ""
            return SkillResult(
                ok=True,
                message=f"{key.title()} is currently {dev.get('state', 'off')}{lvl}.",
                data={"device": key, **dev},
            )

        new_state = "on" if act in ("on", "dim") else ("off" if act == "off" else ("off" if dev.get("state") == "on" else "on"))
        dev["state"] = new_state
        if level is not None:
            dev["brightness"] = max(0, min(100, int(level)))

        transport_used = "local-state"
        ha_url = getattr(ctx.config, "home_assistant_url", "") or os.getenv("HOME_ASSISTANT_URL", "").strip()
        ha_token = getattr(ctx.config, "home_assistant_token", "") or os.getenv("HOME_ASSISTANT_TOKEN", "").strip()
        mqtt_host = getattr(ctx.config, "mqtt_broker_host", "") or os.getenv("MQTT_BROKER_HOST", "").strip()
        mqtt_port = int(getattr(ctx.config, "mqtt_broker_port", 1883) or os.getenv("MQTT_BROKER_PORT", "1883"))

        # 1. Home Assistant REST API
        if ha_url and ha_token:
            domain = dev.get("domain", "homeassistant")
            service = "turn_on" if new_state == "on" else "turn_off"
            body: dict[str, Any] = {"entity_id": dev.get("entity_id")}
            if new_state == "on" and level is not None and domain == "light":
                body["brightness_pct"] = dev["brightness"]
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    r = await client.post(
                        f"{ha_url.rstrip('/')}/api/services/{domain}/{service}",
                        headers={"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"},
                        json=body,
                    )
                    r.raise_for_status()
                transport_used = "home_assistant"
            except Exception as exc:
                return SkillResult(ok=False, message=f"Home Assistant error controlling {key}: {exc}")

        # 2. Direct ESP32 / Raspberry Pi HTTP endpoint
        elif dev.get("endpoint"):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    r = await client.post(
                        str(dev["endpoint"]),
                        json={"device": key, "state": new_state, "level": dev.get("brightness")},
                    )
                    r.raise_for_status()
                transport_used = "esp32_http"
            except Exception as exc:
                return SkillResult(ok=False, message=f"IoT endpoint error for {key}: {exc}")

        # 3. MQTT broker publish (if configured & paho-mqtt installed)
        elif mqtt_host and mqtt_publish is not None:
            try:
                payload = json.dumps({"state": new_state, "brightness": dev.get("brightness")})
                mqtt_publish.single(dev.get("mqtt_topic") or f"home/{key}/set", payload=payload, hostname=mqtt_host, port=mqtt_port)
                transport_used = "mqtt"
            except Exception as exc:
                return SkillResult(ok=False, message=f"MQTT publish failed for {key}: {exc}")

        _save_devices(ctx, devices)
        detail = f" to {dev['brightness']}%" if act == "dim" and dev.get("brightness") is not None else ""
        verb = "Dimmed" if act == "dim" else f"Turned {new_state}"
        suffix = "" if transport_used != "local-state" else " (local smart-home state — configure Home Assistant/ESP32/MQTT in .env to drive physical hardware)"
        return SkillResult(
            ok=True,
            message=f"{verb} {key}{detail}.{suffix}",
            data={"device": key, "state": new_state, "brightness": dev.get("brightness"), "transport": transport_used},
        )

    @registry.register(
        name="list_smart_devices",
        description="List all configured smart home / IoT devices and their current states.",
        category="iot",
        risk=RiskLevel.SAFE,
        examples=["list smart home devices", "show my smart devices"],
    )
    def list_smart_devices(ctx: SkillContext) -> SkillResult:
        devices = _load_devices(ctx)
        summary = ", ".join(f"{name} ({d.get('state', 'off')})" for name, d in devices.items())
        return SkillResult(
            ok=True,
            message=f"Smart devices: {summary}.",
            data={"devices": devices},
        )
