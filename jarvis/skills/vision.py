"""Multimodal Vision skills: Screen Capture & Webcam Inspection with Vision LLMs."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageGrab

from jarvis.ai.providers import OllamaProvider, resolve_llm_provider
from jarvis.core.safety import RiskLevel
from jarvis.skills.registry import SkillContext, SkillRegistry, SkillResult

try:
    import mss  # type: ignore
except ImportError:
    mss = None

try:
    import pygetwindow as gw  # type: ignore
except ImportError:
    gw = None

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None


def capture_screen_jpeg(save_dir: Path, max_dim: int = 1280) -> tuple[Path, str, tuple[int, int]]:
    """Capture primary display via `mss` (fastest) or `PIL.ImageGrab`, returning `(path, base64_jpeg, (w, h))`."""
    save_dir.mkdir(parents=True, exist_ok=True)
    img: Image.Image
    if mss is not None:
        with mss.mss() as sct:
            mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            raw = sct.grab(mon)
            img = Image.frombytes("RGB", raw.size, raw.rgb)
    else:
        img = ImageGrab.grab().convert("RGB")

    orig_size = img.size
    out_path = save_dir / f"vision_screen_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    img.save(out_path, format="JPEG", quality=82)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return out_path, b64, orig_size


def capture_webcam_jpeg(save_dir: Path) -> tuple[Path, str] | None:
    """Capture a single webcam snapshot via OpenCV if installed and camera is accessible."""
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(0)
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((960, 720), Image.Resampling.LANCZOS)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"vision_cam_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        img.save(out_path, format="JPEG", quality=85)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return out_path, base64.b64encode(buf.getvalue()).decode("ascii")
    finally:
        cap.release()


def _active_windows_context() -> dict[str, Any]:
    if gw is None:
        return {"active": "", "open": []}
    try:
        active = gw.getActiveWindow()
        active_title = (active.title or "").strip() if active else ""
        titles = [t.strip() for t in gw.getAllTitles() if t and t.strip()][:10]
        return {"active": active_title, "open": titles}
    except Exception:
        return {"active": "", "open": []}


def register_vision_skills(registry: SkillRegistry) -> None:
    @registry.register(
        name="analyze_screen",
        description=(
            "Capture the user's current screen (and active windows) and analyze it using a vision-capable LLM "
            "(Gemini / OpenAI / Ollama vision) to explain errors, summarize open apps, or answer visual questions."
        ),
        category="vision",
        risk=RiskLevel.SAFE,
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Question about what is currently on the user's screen.",
                }
            },
        },
        examples=[
            "what's on my screen",
            "what's wrong with this error on my screen",
            "summarize what's open right now",
        ],
    )
    async def analyze_screen(ctx: SkillContext, question: str = "Summarize what is currently open on my screen.") -> SkillResult:
        win_ctx = _active_windows_context()
        snap_path: Path | None = None
        img_b64 = ""
        size = (0, 0)
        capture_err = ""
        try:
            snap_path, img_b64, size = capture_screen_jpeg(ctx.config.captures_dir)
        except Exception as exc:
            capture_err = str(exc)

        if img_b64:
            provider = resolve_llm_provider(ctx.config)
            if isinstance(provider, OllamaProvider) and not await provider.is_available():
                provider = None

            if provider is not None:
                try:
                    prompt = (
                        f"User question: {question}\n"
                        f"Active window title: {win_ctx['active'] or 'Unknown'}\n"
                        f"Open windows: {', '.join(win_ctx['open'][:6])}\n"
                        "Describe clearly and concisely (2-3 spoken sentences) what you see and answer the user's question."
                    )
                    resp = await provider.chat_with_image(prompt=prompt, image_b64=img_b64, mime_type="image/jpeg")
                    if resp.text:
                        return SkillResult(
                            ok=True,
                            message=resp.text,
                            data={
                                "path": str(snap_path),
                                "resolution": f"{size[0]}x{size[1]}",
                                "active_window": win_ctx["active"],
                                "vision_provider": f"{resp.provider}/{resp.model}",
                            },
                        )
                except Exception as exc:
                    win_ctx["vision_error"] = str(exc)

            active = win_ctx["active"] or "your desktop"
            others = ", ".join(t for t in win_ctx["open"][:5] if t != active)
            extra = f" Other open windows include: {others}." if others else ""
            msg = (
                f"I captured your screen ({size[0]}x{size[1]}). Your active window is '{active}'.{extra} "
                "Connect Gemini, OpenAI, or Ollama Vision in the AI Brain panel at top-right for deep visual code and error reading."
            )
            return SkillResult(
                ok=True,
                message=msg,
                data={
                    "path": str(snap_path),
                    "resolution": f"{size[0]}x{size[1]}",
                    "active_window": active,
                    "open_windows": win_ctx["open"],
                },
            )

        # Graceful fallback when backend runs on a headless/background Windows desktop session
        active = win_ctx["active"]
        open_list = ", ".join(win_ctx["open"][:5])
        if active or open_list:
            detail = f"Your active window is '{active or win_ctx['open'][0]}'." + (f" Open windows: {open_list}." if open_list else "")
        else:
            detail = "Direct OS screen capture isn't available from this background runner session."
        return SkillResult(
            ok=True,
            message=f"{detail} Click the 👁 Screen button in the dock to share a browser screen frame directly, or run `python main.py` from your desktop terminal.",
            data={"active_window": active, "open_windows": win_ctx["open"], "capture_error": capture_err, "use_browser_capture": True},
        )

    @registry.register(
        name="analyze_camera",
        description="Capture a live frame from the user's webcam and analyze it with a vision-capable model.",
        category="vision",
        risk=RiskLevel.MODERATE,
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What the user wants analyzed in the webcam frame.",
                }
            },
        },
        examples=["look through my camera", "what do you see on my webcam"],
    )
    async def analyze_camera(ctx: SkillContext, question: str = "Describe what you see in front of the camera.") -> SkillResult:
        cap = capture_webcam_jpeg(ctx.config.captures_dir)
        if cap is None:
            return SkillResult(
                ok=False,
                message="Webcam capture requires `opencv-python` and an accessible camera (or check camera permissions).",
                data={"opencv_installed": cv2 is not None},
            )
        snap_path, img_b64 = cap
        provider = resolve_llm_provider(ctx.config)
        if isinstance(provider, OllamaProvider) and not await provider.is_available():
            provider = None
        if provider is not None:
            try:
                resp = await provider.chat_with_image(prompt=question, image_b64=img_b64, mime_type="image/jpeg")
                if resp.text:
                    return SkillResult(
                        ok=True,
                        message=resp.text,
                        data={"path": str(snap_path), "vision_provider": f"{resp.provider}/{resp.model}"},
                    )
            except Exception as exc:
                return SkillResult(ok=False, message=f"Captured camera snapshot to {snap_path.name}, but vision model failed: {exc}")

        return SkillResult(
            ok=True,
            message=f"Captured webcam frame to {snap_path.name}. Connect Gemini or OpenAI in AI Brain to describe live camera visuals.",
            data={"path": str(snap_path)},
        )
