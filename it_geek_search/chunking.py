from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd


def chunk_messages(
    messages: pd.DataFrame,
    window_size: int = 7,
    stride: int = 3,
    max_time_gap_minutes: int = 120
) -> List[Dict[str, Any]]:
    """
    Groups raw messages into sliding context windows while preserving
    conversational boundaries (session and time gaps).
    
    Each chunk contains:
    - chunk_id: unique identifier
    - text: formatted conversation string with speaker attribution
    - participants: list of all participants in the window
    - start_time / end_time: datetime range of the chunk
    - context: list of raw message records in the chunk
    """
    if messages.empty:
        return []

    ordered = messages.sort_values("timestamp").reset_index(drop=True)
    chunks: List[Dict[str, Any]] = []

    # If session_id exists, we can chunk per session, or chunk with time-gap breaking
    has_session = "session_id" in ordered.columns

    if has_session:
        grouped = ordered.groupby("session_id", sort=False)
        groups = [group for _, group in grouped]
    else:
        # Fallback: group by time gaps > max_time_gap_minutes
        groups = []
        current_group = []
        last_ts = None
        for _, row in ordered.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            if last_ts is not None and (ts - last_ts).total_seconds() > max_time_gap_minutes * 60:
                if current_group:
                    groups.append(pd.DataFrame(current_group))
                    current_group = []
            current_group.append(row.to_dict())
            last_ts = ts
        if current_group:
            groups.append(pd.DataFrame(current_group))

    chunk_counter = 0
    for group in groups:
        n = len(group)
        if n == 0:
            continue
        
        # If group is smaller than window_size, emit a single chunk for it
        if n <= window_size:
            starts = [0]
        else:
            starts = list(range(0, n - window_size + 1, stride))
            # Ensure the tail is included if not exact stride match
            if starts[-1] + window_size < n:
                starts.append(n - window_size)

        for start in starts:
            window = group.iloc[start : min(start + window_size, n)]
            
            # Formatted text preserving speaker attribution for better semantic context
            lines = [f"{row['sender']}: {row['text']}" for _, row in window.iterrows()]
            chunk_text = "\n".join(lines)
            
            participants = sorted(window["sender"].unique().tolist())
            start_ts = pd.Timestamp(window["timestamp"].iloc[0])
            end_ts = pd.Timestamp(window["timestamp"].iloc[-1])
            
            chunk = {
                "chunk_id": chunk_counter,
                "session_id": str(window["session_id"].iloc[0]) if "session_id" in window else f"sess_{chunk_counter}",
                "start_message_id": int(window["message_id"].iloc[0]),
                "end_message_id": int(window["message_id"].iloc[-1]),
                "sender": window["sender"].iloc[-1], # compatibility
                "participants": participants,
                "timestamp": end_ts, # compatibility
                "start_time": start_ts.isoformat(),
                "end_time": end_ts.isoformat(),
                "text": chunk_text,
                "topic": window["topic"].iloc[0] if "topic" in window else "general",
                "context": window[["message_id", "sender", "timestamp", "text"]].to_dict("records"),
            }
            chunks.append(chunk)
            chunk_counter += 1

    return chunks
