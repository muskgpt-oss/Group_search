import re
import pandas as pd

def load_chat_from_txt(filepath: str) -> pd.DataFrame:
    """
    Parses a WhatsApp-style text file into a Pandas DataFrame.
    Matches lines like: [2026-09-01 08:01] Riya: nice 😂
    Handles multi-line messages and automatically groups them into sessions
    based on 2-hour inactivity gaps.
    """
    data_rows = []
    msg_id = 0
    
    patterns = [
        re.compile(r"^\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}(?::\d{2})?)\]\s([^:]+):\s(.*)$"), # Custom
        re.compile(r"^(\d{2}/\d{2}/\d{2,4}),\s(\d{1,2}:\d{2}\s?(?:[ap]m|AM|PM)?)\s-\s([^:]+):\s(.*)$", re.IGNORECASE), # Android
        re.compile(r"^\[(\d{2}/\d{2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2}\s?(?:[ap]m|AM|PM)?)\]\s([^:]+):\s(.*)$", re.IGNORECASE) # iOS
    ]
    
    current_msg = None
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            match = None
            for p in patterns:
                m = p.match(line)
                if m:
                    match = m
                    break
                    
            if match:
                if current_msg:
                    data_rows.append(current_msg)
                    msg_id += 1
                
                groups = match.groups()
                if len(groups) == 3:
                    ts_str, sender, text = groups
                else:
                    date_str, time_str, sender, text = groups
                    ts_str = f"{date_str} {time_str}"
                
                try:
                    ts = pd.to_datetime(ts_str, dayfirst=True, format="mixed")
                except ValueError:
                    ts = pd.to_datetime(ts_str, errors="coerce")
                    if pd.isna(ts):
                        ts = pd.Timestamp("2026-01-01") # fallback
                
                current_msg = {
                    "message_id": msg_id,
                    "sender": sender.strip(),
                    "timestamp": ts,
                    "text": text,
                    "topic": "general"
                }
            else:
                # Ignore system messages like "X created group", "messages are encrypted"
                if " - " in line and ":" not in line.split(" - ")[1]:
                    continue
                # If it's not a new message line, append it to the current message (multi-line)
                if current_msg:
                    current_msg["text"] += "\n" + line
    
    if current_msg:
        data_rows.append(current_msg)
        
    df = pd.DataFrame(data_rows)
    
    # Generate session_ids based on time gaps (> 2 hours = new session)
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["message_id"] = range(1, len(df) + 1)
        
        session_id = 0
        session_ids = []
        last_ts = None
        
        for ts in df["timestamp"]:
            if last_ts is not None and (ts - last_ts).total_seconds() > 2 * 3600:
                session_id += 1
            session_ids.append(f"session_{session_id:04d}")
            last_ts = ts
            
        df["session_id"] = session_ids
        
    return df

