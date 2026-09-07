import os
import tempfile
import pandas as pd
import pytest
from it_geek_search.db import MessageDatabase


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_messages.db")
        db = MessageDatabase(db_path)
        yield db


def test_insert_and_count(temp_db):
    sample_data = [
        {"message_id": 1, "session_id": "s1", "sender": "Rohan", "timestamp": "2025-01-01T10:00:00", "text": "Hi all", "topic": "banter"},
        {"message_id": 2, "session_id": "s1", "sender": "Aisha", "timestamp": "2025-01-01T10:01:00", "text": "Hello Rohan", "topic": "banter"},
        {"message_id": 3, "session_id": "s2", "sender": "Karan", "timestamp": "2025-01-02T11:00:00", "text": "Server down", "topic": "server"},
    ]
    inserted = temp_db.insert_messages(sample_data)
    assert inserted == 3
    assert temp_db.count() == 3


def test_get_message(temp_db):
    sample_data = [
        {"message_id": 10, "session_id": "s1", "sender": "Priya", "timestamp": "2025-01-01T10:00:00", "text": "Meeting at 3", "topic": "meeting"},
    ]
    temp_db.insert_messages(sample_data)
    msg = temp_db.get_message(10)
    assert msg is not None
    assert msg["sender"] == "Priya"
    assert msg["text"] == "Meeting at 3"


def test_get_context_around(temp_db):
    messages = [
        {"message_id": i, "session_id": "s1", "sender": "User", "timestamp": f"2025-01-01T10:0{i}:00", "text": f"Msg {i}", "topic": "general"}
        for i in range(1, 8)
    ]
    temp_db.insert_messages(messages)
    
    # Context around message 4 with window 2 before, 2 after
    context = temp_db.get_context_around(4, window_before=2, window_after=2)
    assert len(context) == 5
    ids = [m["message_id"] for m in context]
    assert ids == [2, 3, 4, 5, 6]


def test_filter_messages(temp_db):
    sample_data = [
        {"message_id": 1, "session_id": "s1", "sender": "Rohan", "timestamp": "2025-01-01T10:00:00", "text": "Hi", "topic": "banter"},
        {"message_id": 2, "session_id": "s1", "sender": "Aisha", "timestamp": "2025-02-01T10:00:00", "text": "Hey", "topic": "banter"},
        {"message_id": 3, "session_id": "s1", "sender": "Rohan", "timestamp": "2025-03-01T10:00:00", "text": "Sup", "topic": "banter"},
    ]
    temp_db.insert_messages(sample_data)
    
    rohan_msgs = temp_db.filter_messages(sender="Rohan")
    assert len(rohan_msgs) == 2

    feb_msgs = temp_db.filter_messages(start_date="2025-01-15T00:00:00", end_date="2025-02-15T00:00:00")
    assert len(feb_msgs) == 1
    assert feb_msgs[0]["sender"] == "Aisha"

