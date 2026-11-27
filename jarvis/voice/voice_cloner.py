"""Pure-ONNX Multi-Emotional Neural Voice Cloning Engine (OpenVoice v2).

Clones the user's real voice (`My Voice.mpeg`) locally on CPU using OpenVoice v2's
`SynthesizerTrn` + HiFi-GAN ONNX converter (~0.35x real-time on CPU, zero PyTorch,
zero API keys) while preserving full 10-emotion prosody modulation from the base
expressive speech generator.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort  # type: ignore
except ImportError:
    ort = None

try:
    import imageio_ffmpeg  # type: ignore
except ImportError:
    imageio_ffmpeg = None


SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
PAD_SIZE = (N_FFT - HOP_LENGTH) // 2  # 384

# Periodic Hann window matching upstream PyTorch `torch.hann_window(1024)`
_HANN_WINDOW = (
    0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(WIN_LENGTH, dtype=np.float32) / float(WIN_LENGTH))
).astype(np.float32)


@dataclass(frozen=True)
class EmotionVoiceProfile:
    """Prosodic & vocal energy modulation driving multi-emotional delivery."""

    rate: str
    pitch: str
    volume: str
    gain_db: float = 0.0
    presence_db: float = 0.0  # gentle clarity/warmth EQ tilt


# 10 expressive voice profiles mapped 1-to-1 with J.A.R.V.I.S. facial expressions
EMOTION_VOICE_PROFILES: dict[str, EmotionVoiceProfile] = {
    "neutral": EmotionVoiceProfile(rate="+0%", pitch="+0Hz", volume="+0%", gain_db=0.0, presence_db=0.5),
    "smiling": EmotionVoiceProfile(rate="+4%", pitch="+3Hz", volume="+2%", gain_db=0.5, presence_db=1.2),
    "happy": EmotionVoiceProfile(rate="+10%", pitch="+6Hz", volume="+6%", gain_db=1.2, presence_db=2.0),
    "laughing": EmotionVoiceProfile(rate="+14%", pitch="+9Hz", volume="+8%", gain_db=1.5, presence_db=2.5),
    "surprised": EmotionVoiceProfile(rate="+12%", pitch="+10Hz", volume="+6%", gain_db=1.2, presence_db=2.2),
    "listening": EmotionVoiceProfile(rate="-2%", pitch="+2Hz", volume="+0%", gain_db=0.0, presence_db=0.8),
    "thinking": EmotionVoiceProfile(rate="-8%", pitch="-2Hz", volume="-2%", gain_db=-0.5, presence_db=0.0),
    "serious": EmotionVoiceProfile(rate="-6%", pitch="-4Hz", volume="+2%", gain_db=0.4, presence_db=-0.5),
    "sad": EmotionVoiceProfile(rate="-14%", pitch="-6Hz", volume="-6%", gain_db=-1.5, presence_db=-1.5),
    "angry": EmotionVoiceProfile(rate="+9%", pitch="-3Hz", volume="+10%", gain_db=1.8, presence_db=1.8),
}


def get_emotion_voice_profile(expression: str | None) -> EmotionVoiceProfile:
    key = (expression or "neutral").strip().lower()
    return EMOTION_VOICE_PROFILES.get(key, EMOTION_VOICE_PROFILES["neutral"])


def compute_openvoice_spectrogram(wav_f32: np.ndarray) -> np.ndarray:
    """Compute exact upstream OpenVoice v2 linear STFT magnitude spectrogram `(T, 513)`."""
    wav = np.asarray(wav_f32, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if len(wav) < WIN_LENGTH:
        wav = np.pad(wav, (0, WIN_LENGTH - len(wav)))
    y = np.pad(wav, (PAD_SIZE, PAD_SIZE), mode="reflect")
    n_frames = 1 + (len(y) - WIN_LENGTH) // HOP_LENGTH
    strides = (y.strides[0] * HOP_LENGTH, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=(n_frames, WIN_LENGTH), strides=strides)
    fft = np.fft.rfft(frames * _HANN_WINDOW[None, :], n=N_FFT, axis=1)
    return np.sqrt(fft.real ** 2 + fft.imag ** 2 + 1e-6).astype(np.float32)


def _get_ffmpeg_exe() -> str | None:
    if imageio_ffmpeg is not None:
        try:
            return str(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            pass
    return "ffmpeg"


def decode_audio_to_22050_f32(src: Path | bytes, *, denoise: bool = False) -> np.ndarray:
    """Decode any audio file/bytes (MPEG/MP3/WAV) into normalized 22050 Hz mono float32 PCM."""
    ff = _get_ffmpeg_exe()
    if not ff:
        raise RuntimeError("FFmpeg binary is required for neural voice cloning.")
    af = "highpass=f=85,afftdn=nf=-25" if denoise else "anull"
    is_bytes = isinstance(src, (bytes, bytearray))
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0" if is_bytes else str(src),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-af", af,
        "-f", "f32le", "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        input=bytes(src) if is_bytes else None,
        capture_output=True,
        check=True,
    )
    wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if len(wav):
        peak = float(np.max(np.abs(wav)))
        if peak > 0.99:
            wav = wav * (0.95 / peak)
    return wav


def encode_22050_f32_to_mp3(
    wav_f32: np.ndarray,
    *,
    expression: str = "neutral",
) -> bytes:
    """Master and encode 22050 Hz float32 cloned audio into crisp 128kbps MP3 bytes."""
    ff = _get_ffmpeg_exe()
    if not ff:
        raise RuntimeError("FFmpeg binary is required to encode cloned audio.")

    audio = np.asarray(wav_f32, dtype=np.float32).squeeze()
    if len(audio):
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-5:
            audio = audio * min(1.8, 0.92 / peak)
    pcm16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16).tobytes()

    prof = get_emotion_voice_profile(expression)
    filters = ["highpass=f=75"]
    if abs(prof.presence_db) >= 0.3:
        filters.append(f"treble=g={prof.presence_db:.1f}:f=3200")
    if abs(prof.gain_db) >= 0.2:
        filters.append(f"volume={prof.gain_db:.1f}dB")
    filters.append("alimiter=limit=0.95")

    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
        "-af", ",".join(filters),
        "-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3", "pipe:1",
    ]
    proc = subprocess.run(cmd, input=pcm16, capture_output=True, check=True)
    return proc.stdout


class NeuralVoiceCloner:
    """Local pure-ONNX OpenVoice v2 ToneColorConverter cloning `My Voice.mpeg`."""

    def __init__(
        self,
        reference_audio_path: Path,
        models_dir: Path,
    ) -> None:
        self.reference_audio_path = Path(reference_audio_path)
        self.models_dir = Path(models_dir)
        self.ref_encoder_path = self.models_dir / "tone_ref_encoder.onnx"
        self.converter_path = self.models_dir / "tone_converter.onnx"
        self.target_se_cache = self.models_dir / "my_voice_se.npy"
        self.target_meta_cache = self.models_dir / "my_voice_se.meta"
        self.source_se_cache = self.models_dir / "base_speaker_se.npy"
        self._source_se_map: dict[str, np.ndarray] = {}

        self._lock = threading.Lock()
        self._ref_sess: Any = None
        self._conv_sess: Any = None
        self._target_se: np.ndarray | None = None
        self._source_se: np.ndarray | None = None

    @property
    def is_available(self) -> bool:
        return (
            ort is not None
            and self.reference_audio_path.exists()
            and self.ref_encoder_path.exists()
            and self.converter_path.exists()
        )

    def status(self) -> dict[str, Any]:
        return {
            "available": self.is_available,
            "reference_file": self.reference_audio_path.name if self.reference_audio_path.exists() else None,
            "onnxruntime_installed": ort is not None,
            "models_ready": self.ref_encoder_path.exists() and self.converter_path.exists(),
            "loaded": self._conv_sess is not None,
        }

    def _ensure_loaded(self) -> None:
        if self._conv_sess is not None and self._target_se is not None:
            return
        with self._lock:
            if self._conv_sess is not None and self._target_se is not None:
                return
            if not self.is_available:
                raise RuntimeError("OpenVoice v2 ONNX voice cloning assets are not available.")

            threads = min(6, max(2, (os.cpu_count() or 4) // 2))
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._ref_sess = ort.InferenceSession(
                str(self.ref_encoder_path), so, providers=["CPUExecutionProvider"]
            )
            self._conv_sess = ort.InferenceSession(
                str(self.converter_path), so, providers=["CPUExecutionProvider"]
            )
            self._target_se = self._load_or_extract_target_se()
            if self.source_se_cache.exists():
                try:
                    self._source_se = np.load(self.source_se_cache).astype(np.float32)
                except Exception:
                    self._source_se = None
            logger.info(
                "Loaded OpenVoice v2 ONNX cloner for '%s' (threads=%d).",
                self.reference_audio_path.name,
                threads,
            )

    def _extract_averaged_se(self, wav: np.ndarray, *, skip_horn_window: bool = False) -> np.ndarray:
        sr = SAMPLE_RATE
        dur = len(wav) / float(sr)
        chunks: list[np.ndarray] = []

        if skip_horn_window and dur >= 28.0:
            # Prefer clean speech segments, skipping any noisy middle traffic spike (~14.5s-19.5s)
            candidates = [
                wav[int(1.0 * sr) : int(13.5 * sr)],
                wav[int(19.5 * sr) : min(len(wav), int(36.0 * sr))],
            ]
            for cand in candidates:
                for st in range(0, max(1, len(cand) - sr * 3), sr * 3):
                    seg = cand[st : st + sr * 5]
                    if len(seg) >= sr * 2 and float(np.sqrt(np.mean(seg * seg))) >= 0.02:
                        chunks.append(seg)
        else:
            if len(wav) <= sr * 6:
                chunks.append(wav)
            else:
                for st in range(0, max(1, len(wav) - sr * 2), sr * 3):
                    seg = wav[st : st + sr * 5]
                    if len(seg) >= sr * 2 and float(np.sqrt(np.mean(seg * seg))) >= 0.02:
                        chunks.append(seg)

        if not chunks:
            chunks = [wav]

        in_name = self._ref_sess.get_inputs()[0].name
        embs = []
        for seg in chunks:
            spec = compute_openvoice_spectrogram(seg)[None, :, :]
            emb = self._ref_sess.run(None, {in_name: spec})[0]
            embs.append(emb)
        return np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32)

    def _load_or_extract_target_se(self) -> np.ndarray:
        st = self.reference_audio_path.stat()
        sig = f"{self.reference_audio_path.name}:{st.st_size}:{int(st.st_mtime)}"
        if self.target_se_cache.exists() and self.target_meta_cache.exists():
            try:
                if self.target_meta_cache.read_text(encoding="utf-8").strip() == sig:
                    return np.load(self.target_se_cache).astype(np.float32)
            except Exception:
                pass

        wav = decode_audio_to_22050_f32(self.reference_audio_path, denoise=True)
        se = self._extract_averaged_se(wav, skip_horn_window=True)
        try:
            np.save(self.target_se_cache, se)
            self.target_meta_cache.write_text(sig, encoding="utf-8")
        except Exception:
            pass
        return se

    def ensure_models_downloaded(self) -> bool:
        """Auto-download OpenVoice v2 FP32 ONNX weights (~125 MB total) on fresh clone if missing."""
        if self.ref_encoder_path.exists() and self.converter_path.exists():
            return True
        try:
            import httpx

            base_url = "https://huggingface.co/TigreGotico/voiceclonnx-openvoice-v2/resolve/main"
            self.models_dir.mkdir(parents=True, exist_ok=True)
            for fname, target in (
                ("tone_ref_encoder.onnx", self.ref_encoder_path),
                ("tone_converter.onnx", self.converter_path),
            ):
                if target.exists() and target.stat().st_size > 1_000_000:
                    continue
                logger.info("Downloading OpenVoice v2 ONNX model '%s'...", fname)
                with httpx.stream("GET", f"{base_url}/{fname}", follow_redirects=True, timeout=120.0) as resp:
                    resp.raise_for_status()
                    with target.open("wb") as fh:
                        for chunk in resp.iter_bytes(131072):
                            fh.write(chunk)
            return self.ref_encoder_path.exists() and self.converter_path.exists()
        except Exception as exc:
            logger.warning("OpenVoice v2 ONNX auto-download skipped: %s", exc)
            return False

    def warmup(self, neutral_base_mp3: bytes | None = None) -> bool:
        """Pre-load ONNX sessions and target/base speaker embeddings for low first-turn latency."""
        if ort is None or not self.reference_audio_path.exists():
            return False
        if not (self.ref_encoder_path.exists() and self.converter_path.exists()):
            self.ensure_models_downloaded()
        if not self.is_available:
            return False
        self._ensure_loaded()
        if neutral_base_mp3 and self._source_se is None:
            try:
                src_wav = decode_audio_to_22050_f32(neutral_base_mp3, denoise=False)
                se = self._extract_averaged_se(src_wav, skip_horn_window=False)
                self._source_se = se
                np.save(self.source_se_cache, se)
            except Exception as exc:
                logger.debug("Base speaker SE cache warmup skipped: %s", exc)
        return True

    def _source_cache_path(self, source_key: str) -> Path:
        if source_key == "en":
            return self.source_se_cache
        return self.models_dir / f"base_speaker_{source_key}_se.npy"

    def convert_mp3_bytes(
        self,
        source_mp3_bytes: bytes,
        *,
        expression: str = "neutral",
        source_key: str = "en",
    ) -> bytes:
        """Convert expressive base TTS MP3 bytes into the user's cloned voice MP3 bytes."""
        self._ensure_loaded()
        t0 = time.perf_counter()

        src_wav = decode_audio_to_22050_f32(source_mp3_bytes, denoise=False)
        if len(src_wav) < 1200:
            return source_mp3_bytes

        spec_t_f = compute_openvoice_spectrogram(src_wav)  # (T, 513)

        # Use cached neutral base speaker SE per base voice ('en' vs 'hi') so emotional prosody is preserved
        src_se = self._source_se_map.get(source_key)
        if src_se is None:
            cache_p = self._source_cache_path(source_key)
            if cache_p.exists():
                try:
                    src_se = np.load(cache_p).astype(np.float32)
                    self._source_se_map[source_key] = src_se
                except Exception:
                    src_se = None
        if src_se is None and source_key == "en" and self._source_se is not None:
            src_se = self._source_se
            self._source_se_map["en"] = src_se
        if src_se is None:
            ref_in = self._ref_sess.get_inputs()[0].name
            src_se = self._ref_sess.run(None, {ref_in: spec_t_f[None, :, :]})[0].astype(np.float32)
            if (expression or "neutral").lower() in ("neutral", "smiling") and len(src_wav) >= SAMPLE_RATE * 2:
                self._source_se_map[source_key] = src_se
                try:
                    np.save(self._source_cache_path(source_key), src_se)
                except Exception:
                    pass

        assert self._target_se is not None
        in_names = [inp.name for inp in self._conv_sess.get_inputs()]
        feed = {
            in_names[0]: spec_t_f.T[None, :, :],  # (1, 513, T)
            in_names[1]: np.array([spec_t_f.shape[0]], dtype=np.int64),
            in_names[2]: src_se[:, :, None],
            in_names[3]: self._target_se[:, :, None],
        }
        with self._lock:
            cloned_wav = self._conv_sess.run(None, feed)[0].squeeze()

        mp3_out = encode_22050_f32_to_mp3(cloned_wav, expression=expression)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "Cloned voice (%s) in %.0fms (%.2fs audio).",
            expression,
            elapsed_ms,
            len(cloned_wav) / float(SAMPLE_RATE),
        )
        return mp3_out
