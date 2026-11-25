# J.A.R.V.I.S. System Manual & Knowledge Base

## Architecture Overview
J.A.R.V.I.S. (Just A Rather Very Intelligent System) is an event-driven Python AI voice control assistant built on an asynchronous pub/sub core, FastAPI + WebSocket real-time HUD, multi-provider LLM tool calling, and a hybrid SQLite + NumPy semantic RAG vector store.

## Supported AI Brain Providers
- **Google Gemini**: Configured via `GEMINI_API_KEY` (default model `gemini-2.5-flash`) with native function declarations.
- **OpenAI / Groq / OpenRouter**: Configured via `OPENAI_API_KEY` or `GROQ_API_KEY` using OpenAI-compatible tool calling.
- **Local Ollama**: Private on-device LLM inference at `http://localhost:11434` (e.g., `llama3.2` or `qwen2.5`).
- **Offline Fast-Path & Intent Router**: Always available with zero API keys, supporting sub-50ms hardware commands and 30+ built-in natural language intents.

## Voice Control Modes
1. **Browser Hands-Free Wake Mic**: Uses Chrome/Edge Web Speech API in the Holographic HUD. Say "Jarvis" or "Hey Jarvis" followed by any command.
2. **Push-to-Talk**: Click the Arc Reactor orb or press `Space` in the HUD.
3. **Local Python Microphone Loop**: Uses `sounddevice` energy VAD with `faster-whisper` (offline neural STT), Groq/OpenAI Whisper, or Google Speech Recognition.
4. **Neural Text-to-Speech**: Uses Microsoft `edge-tts` (`en-GB-RyanNeural`) with automatic offline fallback to Windows SAPI5 (`pyttsx3`) and instant barge-in interruption (`Esc` or saying "Stop").

## Safety Protocol
High-impact commands such as `power_control` (shutdown, restart, sleep) are classified as `DANGEROUS` by the `SafetyGuard`. J.A.R.V.I.S. requires explicit verbal confirmation ("confirm" / "yes") or clicking Confirm in the HUD before executing them, and records every tool invocation in the SQLite audit log.
