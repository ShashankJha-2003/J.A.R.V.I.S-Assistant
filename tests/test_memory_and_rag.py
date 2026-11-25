"""Tests for SQLite Persistence and Hybrid NumPy Semantic Vector RAG."""

from __future__ import annotations

import time
from pathlib import Path

from jarvis.memory.database import JarvisDatabase
from jarvis.memory.vector_store import SemanticVectorStore, chunk_document


def test_database_notes_tasks_reminders_and_messages(tmp_path: Path) -> None:
    db = JarvisDatabase(tmp_path / "test.db")

    # Messages
    db.add_message("s1", "user", "hello")
    db.add_message("s1", "assistant", "hi there", metadata={"provider": "offline"})
    msgs = db.get_recent_messages("s1", limit=10)
    assert len(msgs) == 2
    assert msgs[1]["metadata"]["provider"] == "offline"

    # Notes
    n = db.add_note("Demo", "Build voice assistant with FastAPI", tags="portfolio")
    assert len(db.list_notes(query="FastAPI")) == 1
    assert db.delete_note(n["id"]) is True
    assert db.list_notes() == []

    # Tasks
    t = db.add_task("Finish README", priority="high")
    assert len(db.list_tasks(include_completed=False)) == 1
    assert db.complete_task(t["id"]) is True
    assert db.list_tasks(include_completed=False) == []

    # Reminders
    past = time.time() - 5.0
    future = time.time() + 300.0
    db.add_reminder("Drink water", trigger_at=past)
    db.add_reminder("Evening standup", trigger_at=future)
    due = db.pop_due_reminders()
    assert len(due) == 1
    assert due[0]["message"] == "Drink water"
    assert len(db.list_reminders(status="pending")) == 1

    db.close()


def test_semantic_vector_store_and_rag_ingestion(tmp_path: Path) -> None:
    db = JarvisDatabase(tmp_path / "vec.db")
    store = SemanticVectorStore(db)

    store.add_memory("My primary workstation runs Windows 11 with Python 3.12.", namespace="fact")
    store.add_memory("My favorite coffee order is an iced oat latte.", namespace="fact")

    hits = store.search("what operating system does my workstation run?")
    assert len(hits) >= 1
    assert "Windows 11" in hits[0]["content"]

    # Local document RAG ingestion
    kb_dir = tmp_path / "knowledge"
    kb_dir.mkdir()
    doc = kb_dir / "architecture.md"
    doc.write_text(
        "# Telemetry Protocol\n\n"
        "J.A.R.V.I.S. broadcasts live CPU, memory, and battery telemetry over WebSockets every 2.5 seconds.\n\n"
        "# Safety Gate\n\n"
        "Shutdown and restart commands require explicit confirmation tokens before execution.",
        encoding="utf-8",
    )

    summary = store.ingest_directory(kb_dir)
    assert summary["files_indexed"] == 1
    assert summary["total_chunks"] >= 1

    rag_hits = store.search("How does J.A.R.V.I.S. broadcast telemetry over WebSockets?", namespace="rag")
    assert len(rag_hits) >= 1
    assert "WebSockets" in rag_hits[0]["content"]

    assert len(chunk_document("word " * 300, chunk_size=200, overlap=40)) >= 4
    db.close()
