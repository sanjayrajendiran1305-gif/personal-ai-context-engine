import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.getenv("DATABASE_PATH", str(ROOT / "memory.db"))).expanduser()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def current_time():
    return datetime.now(timezone.utc).isoformat()


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _columns(conn, table):
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def init_db():
    with closing(get_connection()) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations(
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT 'New conversation',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS interactions(
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            message TEXT NOT NULL,
            requested_provider TEXT NOT NULL DEFAULT 'both',
            gemini_response TEXT,
            groq_response TEXT,
            selected_provider TEXT,
            selected_reasons TEXT NOT NULL DEFAULT '[]',
            custom_reason TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            gemini_error TEXT,
            groq_error TEXT,
            category TEXT NOT NULL DEFAULT 'general',
            subcategory TEXT NOT NULL DEFAULT 'general_question',
            output_format TEXT,
            attachments_json TEXT NOT NULL DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS memories(
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            memory_key TEXT NOT NULL,
            memory_value TEXT NOT NULL,
            source_type TEXT NOT NULL DEFAULT 'inferred',
            confidence REAL NOT NULL DEFAULT 0.5,
            evidence_count INTEGER NOT NULL DEFAULT 1,
            source_interaction_id TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS preference_examples(
            id TEXT PRIMARY KEY,
            interaction_id TEXT NOT NULL,
            category TEXT NOT NULL,
            provider TEXT NOT NULL,
            features_json TEXT NOT NULL,
            label INTEGER NOT NULL CHECK(label IN (0,1)),
            created_at TEXT NOT NULL,
            UNIQUE(interaction_id, provider)
        );

        CREATE TABLE IF NOT EXISTS learning_runs(
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            example_count INTEGER NOT NULL,
            accuracy REAL,
            model_path TEXT,
            status TEXT NOT NULL,
            details TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS attachments(
            id TEXT PRIMARY KEY,
            interaction_id TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS migration_log(
            name TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS artifacts(
            id TEXT PRIMARY KEY,
            interaction_id TEXT,
            filename TEXT NOT NULL,
            format TEXT NOT NULL,
            path TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_interactions_conversation
            ON interactions(conversation_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_interactions_created
            ON interactions(created_at);
        CREATE INDEX IF NOT EXISTS idx_memories_category
            ON memories(category, is_active, confidence);
        CREATE INDEX IF NOT EXISTS idx_preference_examples_category
            ON preference_examples(category, created_at);
        CREATE INDEX IF NOT EXISTS idx_attachments_interaction
            ON attachments(interaction_id);
        """)

        # Upgrade databases created by earlier versions without discarding data.
        migrations = {
            "interactions": {
                "conversation_id": "TEXT",
                "requested_provider": "TEXT NOT NULL DEFAULT 'both'",
                "gemini_response": "TEXT",
                "groq_response": "TEXT",
                "selected_provider": "TEXT",
                "selected_reasons": "TEXT NOT NULL DEFAULT '[]'",
                "custom_reason": "TEXT NOT NULL DEFAULT ''",
                "created_at": "TEXT NOT NULL DEFAULT ''",
                "gemini_error": "TEXT",
                "groq_error": "TEXT",
                "category": "TEXT NOT NULL DEFAULT 'general'",
                "subcategory": "TEXT NOT NULL DEFAULT 'general_question'",
                "output_format": "TEXT",
                "attachments_json": "TEXT NOT NULL DEFAULT '[]'",
            },
            "conversations": {
                "title": "TEXT NOT NULL DEFAULT 'New conversation'",
                "created_at": "TEXT NOT NULL DEFAULT ''",
                "updated_at": "TEXT NOT NULL DEFAULT ''",
            },
            "memories": {
                "category": "TEXT NOT NULL DEFAULT 'general'",
                "memory_key": "TEXT NOT NULL DEFAULT 'preference'",
                "memory_value": "TEXT NOT NULL DEFAULT ''",
                "source_type": "TEXT NOT NULL DEFAULT 'inferred'",
                "confidence": "REAL NOT NULL DEFAULT 0.5",
                "evidence_count": "INTEGER NOT NULL DEFAULT 1",
                "source_interaction_id": "TEXT",
                "is_active": "INTEGER NOT NULL DEFAULT 1",
                "created_at": "TEXT NOT NULL DEFAULT ''",
                "updated_at": "TEXT NOT NULL DEFAULT ''",
            },
        }
        for table, fields in migrations.items():
            existing = _columns(conn, table)
            for column, definition in fields.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

        now = current_time()
        rows = conn.execute(
            "SELECT id,message,conversation_id,created_at FROM interactions"
        ).fetchall()
        for row in rows:
            conversation_id = row["conversation_id"] or str(uuid.uuid4())
            created_at = row["created_at"] or now
            if not row["conversation_id"]:
                conn.execute(
                    "UPDATE interactions SET conversation_id=? WHERE id=?",
                    (conversation_id, row["id"]),
                )
            title = (row["message"] or "New conversation").strip().replace("\n", " ")[:55]
            conn.execute(
                "INSERT OR IGNORE INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                (conversation_id, title or "New conversation", created_at, created_at),
            )

        # Backfill pairwise preference examples from legacy selections exactly once.
        selected_rows = conn.execute("""
            SELECT id,category,selected_provider,gemini_response,groq_response,created_at
            FROM interactions
            WHERE selected_provider IN ('gemini','groq')
              AND gemini_response IS NOT NULL AND groq_response IS NOT NULL
        """).fetchall()
        for row in selected_rows:
            if conn.execute(
                "SELECT 1 FROM preference_examples WHERE interaction_id=? LIMIT 1",
                (row["id"],),
            ).fetchone():
                continue
            for provider, label in (("gemini", 1 if row["selected_provider"] == "gemini" else 0),
                                    ("groq", 1 if row["selected_provider"] == "groq" else 0)):
                answer = row[f"{provider}_response"] or ""
                features = _basic_response_features(answer)
                conn.execute("""
                    INSERT OR IGNORE INTO preference_examples(
                        id,interaction_id,category,provider,features_json,label,created_at
                    ) VALUES(?,?,?,?,?,?,?)
                """, (
                    str(uuid.uuid4()), row["id"], row["category"] or "general",
                    provider, json.dumps(features), label, row["created_at"] or now,
                ))
        conn.commit()


def _basic_response_features(text):
    import re
    text = text or ""
    words = re.findall(r"\b\w+\b", text)
    return {
        "log_words": round(__import__("math").log1p(len(words)), 5),
        "headings": len(re.findall(r"(?m)^\s{0,3}#{1,6}\s", text)),
        "bullets": len(re.findall(r"(?m)^\s*[-*+]\s", text)),
        "numbered_steps": len(re.findall(r"(?m)^\s*\d+[.)]\s", text)),
        "examples": len(re.findall(r"\b(example|for instance|such as)\b", text, re.I)),
        "code_blocks": len(re.findall(r"```", text)) // 2,
        "tables": len(re.findall(r"(?m)^\s*\|", text)),
        "paragraphs": len([p for p in re.split(r"\n\s*\n", text) if p.strip()]),
    }


def execute_query(query, params=()):
    with closing(get_connection()) as conn:
        cursor = conn.execute(query, params)
        conn.commit()
        return cursor.rowcount


def fetch_one(query, params=()):
    with closing(get_connection()) as conn:
        return conn.execute(query, params).fetchone()


def fetch_all(query, params=()):
    with closing(get_connection()) as conn:
        return conn.execute(query, params).fetchall()
