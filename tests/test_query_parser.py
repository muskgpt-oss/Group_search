from it_geek_search.query_parser import parse_query


def test_parse_query_sender_attribution():
    q1 = parse_query("What did Rohan say about the server costs?")
    assert q1.sender == "Rohan"
    assert "server costs" in q1.clean_query.lower()
    assert q1.inferred_intent in ["attributed", "hybrid"]

    q2 = parse_query("Priya's message on hotel booking")
    assert q2.sender == "Priya"
    assert "hotel booking" in q2.clean_query.lower()

    q3 = parse_query("Who suggested spot instances from Vivek?")
    assert q3.sender == "Vivek"


def test_parse_query_temporal():
    q1 = parse_query("What did we discuss back in March?")
    assert q1.start_date is not None
    assert "2025-03-01" in q1.start_date
    assert "2025-03-31" in q1.end_date

    q2 = parse_query("Show me plans between 2025-02-01 and 2025-02-28")
    assert q2.start_date == "2025-02-01T00:00:00"
    assert q2.end_date == "2025-02-28T23:59:59"


def test_parse_query_hybrid():
    q = parse_query("What did Karan say in May about the 502 error?")
    assert q.sender == "Karan"
    assert q.start_date is not None
    assert "2025-05-01" in q.start_date
    assert "502 error" in q.clean_query.lower()

