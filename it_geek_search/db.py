from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


class MessageDatabase:
    """SQLite-backed message storage with indexed querying and context retrieval."""

    def __init__(self, db_path: Union[str, Path] = "data/chat_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    text TEXT NOT NULL,
                    topic TEXT
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)"
            )
            conn.commit()

    def insert_messages(self, messages: Union[pd.DataFrame, List[Dict[str, Any]]]) -> int:
        """Insert messages from either a DataFrame or list of dicts."""
        if isinstance(messages, pd.DataFrame):
            records = messages.to_dict(orient="records")
        else:
            records = messages

        if not records:
            return 0

        rows = []
        for r in records:
            ts = r["timestamp"]
            if hasattr(ts, "isoformat"):
                ts_str = ts.isoformat()
            else:
                ts_str = str(ts)
            rows.append(
                (
                    int(r["message_id"]),
                    str(r.get("session_id", "default")),
                    str(r["sender"]),
                    ts_str,
                    str(r["text"]),
                    str(r.get("topic", "")),
                )
            )

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO messages (message_id, session_id, sender, timestamp, text, topic)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            return len(rows)

    def count(self) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM messages")
            return cursor.fetchone()[0]

    def get_message(self, message_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM messages WHERE message_id = ?", (message_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_context_around(
        self, message_id: int, window_before: int = 3, window_after: int = 3
    ) -> List[Dict[str, Any]]:
        """Fetch surrounding conversation context within the same session or chronological span."""
        target = self.get_message(message_id)
        if not target:
            return []

        session_id = target.get("session_id")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Fetch previous messages in session
            cursor.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ? AND timestamp <= ? AND message_id != ?
                ORDER BY timestamp DESC, message_id DESC
                LIMIT ?
                """,
                (session_id, target["timestamp"], message_id, window_before),
            )
            before_msgs = [dict(r) for r in cursor.fetchall()][::-1]

            # Fetch succeeding messages in session
            cursor.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ? AND timestamp >= ? AND message_id != ?
                ORDER BY timestamp ASC, message_id ASC
                LIMIT ?
                """,
                (session_id, target["timestamp"], message_id, window_after),
            )
            after_msgs = [dict(r) for r in cursor.fetchall()]

            return before_msgs + [target] + after_msgs

    def get_all_messages(self, as_df: bool = True) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM messages ORDER BY timestamp ASC, message_id ASC")
            rows = [dict(r) for r in cursor.fetchall()]
            if as_df:
                df = pd.DataFrame(rows)
                if not df.empty:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
            return rows

    def filter_messages(
        self,
        sender: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM messages WHERE 1=1"
        params: List[Any] = []

        if sender:
            query += " AND LOWER(sender) = LOWER(?)"
            params.append(sender)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    def clear(self) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages")
            conn.commit()

