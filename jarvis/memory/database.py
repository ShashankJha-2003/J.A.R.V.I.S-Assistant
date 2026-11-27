"""SQLite persistence layer for conversations, notes, tasks, reminders, and audit logs."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class JarvisDatabase:
    """Thread-safe SQLite storage with WAL journaling for J.A.R.V.I.S."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls_json TEXT,
                    metadata_json TEXT,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    priority TEXT DEFAULT 'normal',
                    completed INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    completed_at REAL
                );

                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message TEXT NOT NULL,
                    trigger_at REAL NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    fired_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status, trigger_at);

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL DEFAULT 'fact',
                    source TEXT NOT NULL DEFAULT 'user',
                    content TEXT NOT NULL,
                    metadata_json TEXT,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding_model TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_ns ON memories(namespace);

                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_preview TEXT,
                    latency_ms REAL,
                    created_at REAL NOT NULL
                );
                """
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Messages / Conversation History
    # ------------------------------------------------------------------
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = time.time()
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO messages (session_id, role, content, tool_calls_json, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                    json.dumps(metadata, ensure_ascii=False) if metadata else None,
                    now,
                ),
            )
            return int(cur.lastrowid)

    def get_recent_messages(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, session_id, role, content, tool_calls_json, metadata_json, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in reversed(rows):
            result.append(
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "tool_calls": json.loads(row["tool_calls_json"]) if row["tool_calls_json"] else [],
                    "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
                    "created_at": row["created_at"],
                }
            )
        return result

    def clear_session(self, session_id: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            return cur.rowcount

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------
    def add_note(self, title: str, content: str, tags: str = "") -> dict[str, Any]:
        now = time.time()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO notes (title, content, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (title.strip(), content.strip(), tags.strip(), now, now),
            )
            note_id = int(cur.lastrowid)
        return {"id": note_id, "title": title.strip(), "content": content.strip(), "tags": tags.strip(), "created_at": now}

    def list_notes(self, query: str = "", limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            if query.strip():
                like = f"%{query.strip()}%"
                rows = self._conn.execute(
                    """
                    SELECT id, title, content, tags, created_at, updated_at
                    FROM notes
                    WHERE title LIKE ? OR content LIKE ? OR tags LIKE ?
                    ORDER BY updated_at DESC LIMIT ?
                    """,
                    (like, like, like, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, title, content, tags, created_at, updated_at FROM notes ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def delete_note(self, note_id: int) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Tasks / To-Do List
    # ------------------------------------------------------------------
    def add_task(self, title: str, priority: str = "normal") -> dict[str, Any]:
        now = time.time()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO tasks (title, priority, completed, created_at) VALUES (?, ?, 0, ?)",
                (title.strip(), priority.strip().lower(), now),
            )
            task_id = int(cur.lastrowid)
        return {"id": task_id, "title": title.strip(), "priority": priority.strip().lower(), "completed": False, "created_at": now}

    def list_tasks(self, include_completed: bool = True) -> list[dict[str, Any]]:
        with self._lock:
            if include_completed:
                rows = self._conn.execute(
                    "SELECT id, title, priority, completed, created_at, completed_at FROM tasks ORDER BY completed ASC, id DESC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, title, priority, completed, created_at, completed_at FROM tasks WHERE completed = 0 ORDER BY id DESC"
                ).fetchall()
        return [{**dict(r), "completed": bool(r["completed"])} for r in rows]

    def complete_task(self, task_id: int) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE tasks SET completed = 1, completed_at = ? WHERE id = ?",
                (time.time(), task_id),
            )
            return cur.rowcount > 0

    def delete_task(self, task_id: int) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Timers & Reminders
    # ------------------------------------------------------------------
    def add_reminder(self, message: str, trigger_at: float) -> dict[str, Any]:
        now = time.time()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO reminders (message, trigger_at, status, created_at) VALUES (?, ?, 'pending', ?)",
                (message.strip(), trigger_at, now),
            )
            rem_id = int(cur.lastrowid)
        return {"id": rem_id, "message": message.strip(), "trigger_at": trigger_at, "status": "pending", "created_at": now}

    def list_reminders(self, status: str | None = "pending") -> list[dict[str, Any]]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT id, message, trigger_at, status, created_at, fired_at FROM reminders WHERE status = ? ORDER BY trigger_at ASC",
                    (status,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, message, trigger_at, status, created_at, fired_at FROM reminders ORDER BY trigger_at DESC LIMIT 50"
                ).fetchall()
        return [dict(r) for r in rows]

    def pop_due_reminders(self, now: float | None = None) -> list[dict[str, Any]]:
        ts = now if now is not None else time.time()
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT id, message, trigger_at, status, created_at FROM reminders WHERE status = 'pending' AND trigger_at <= ? ORDER BY trigger_at ASC",
                (ts,),
            ).fetchall()
            due = [dict(r) for r in rows]
            for item in due:
                self._conn.execute(
                    "UPDATE reminders SET status = 'fired', fired_at = ? WHERE id = ?",
                    (ts, item["id"]),
                )
        return due

    def cancel_reminder(self, reminder_id: int) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE reminders SET status = 'cancelled' WHERE id = ? AND status = 'pending'",
                (reminder_id,),
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Audit Logs
    # ------------------------------------------------------------------
    def log_audit(
        self,
        skill_name: str,
        arguments: dict[str, Any],
        status: str,
        result_preview: str = "",
        latency_ms: float = 0.0,
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO audit_logs (skill_name, arguments_json, status, result_preview, latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    skill_name,
                    json.dumps(arguments, default=str),
                    status,
                    result_preview[:500],
                    round(latency_ms, 2),
                    time.time(),
                ),
            )
            return int(cur.lastrowid)

    def list_audit_logs(self, limit: int = 40) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, skill_name, arguments_json, status, result_preview, latency_ms, created_at FROM audit_logs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "skill_name": r["skill_name"],
                "arguments": json.loads(r["arguments_json"]),
                "status": r["status"],
                "result_preview": r["result_preview"],
                "latency_ms": r["latency_ms"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Raw Vector Memory Table Accessors (used by SemanticVectorStore)
    # ------------------------------------------------------------------
    def insert_memory_vector(
        self,
        namespace: str,
        source: str,
        content: str,
        embedding_bytes: bytes,
        embedding_dim: int,
        embedding_model: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO memories (namespace, source, content, metadata_json, embedding, embedding_dim, embedding_model, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    namespace,
                    source,
                    content,
                    json.dumps(metadata or {}),
                    embedding_bytes,
                    embedding_dim,
                    embedding_model,
                    time.time(),
                ),
            )
            return int(cur.lastrowid)

    def fetch_memory_vectors(
        self,
        namespace: str | None = None,
        embedding_model: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if namespace:
            clauses.append("namespace = ?")
            params.append(namespace)
        if embedding_model:
            clauses.append("embedding_model = ?")
            params.append(embedding_model)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, namespace, source, content, metadata_json, embedding, embedding_dim, embedding_model, created_at FROM memories {where} ORDER BY id DESC",
                params,
            ).fetchall()
        return [
            {
                "id": r["id"],
                "namespace": r["namespace"],
                "source": r["source"],
                "content": r["content"],
                "metadata": json.loads(r["metadata_json"]) if r["metadata_json"] else {},
                "embedding": r["embedding"],
                "embedding_dim": r["embedding_dim"],
                "embedding_model": r["embedding_model"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def delete_memories_by_source(self, source: str, namespace: str = "rag") -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE source = ? AND namespace = ?",
                (source, namespace),
            )
            return cur.rowcount

    def delete_memory(self, memory_id: int) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cur.rowcount > 0

    def memory_stats(self) -> dict[str, int]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            facts = self._conn.execute("SELECT COUNT(*) FROM memories WHERE namespace = 'fact'").fetchone()[0]
            rag = self._conn.execute("SELECT COUNT(*) FROM memories WHERE namespace = 'rag'").fetchone()[0]
            notes = self._conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            tasks = self._conn.execute("SELECT COUNT(*) FROM tasks WHERE completed = 0").fetchone()[0]
            reminders = self._conn.execute("SELECT COUNT(*) FROM reminders WHERE status = 'pending'").fetchone()[0]
        return {
            "total_vectors": int(total),
            "facts": int(facts),
            "rag_chunks": int(rag),
            "notes": int(notes),
            "open_tasks": int(tasks),
            "pending_reminders": int(reminders),
        }
