from __future__ import annotations

import re
import calendar
from dataclasses import dataclass
from typing import Optional, Tuple
from .generator import PARTICIPANTS


@dataclass
class ParsedQuery:
    raw_query: str
    clean_query: str
    sender: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    inferred_intent: str = "semantic"  # "semantic", "attributed", "temporal", "hybrid"


MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def parse_query(query: str, default_year: int = 2025) -> ParsedQuery:
    """
    Parses a natural language query to extract:
    - Target sender (author attribution)
    - Date range (temporal filtering)
    - Cleaned query text (for semantic/keyword search)
    - Inferred intent type
    """
    cleaned = query.strip()
    extracted_sender: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    intent = "semantic"

    # 1. Extract sender if present
    for name in PARTICIPANTS:
        pattern = rf"\b(?:what did |from |by |posted by )?({re.escape(name)})(?:'s|\s+say|\s+mention|\s+ask|\s+post|\s+write)?\b"
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            extracted_sender = name
            intent = "attributed"
            # Remove sender references from query to purify semantic payload
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
            break

    # 2. Extract explicit date ranges (e.g. 2025-01-01 to 2025-02-15)
    iso_range_pattern = r"(?:between|from)?\s*(\d{4}-\d{2}-\d{2})\s*(?:to|and|-)\s*(\d{4}-\d{2}-\d{2})"
    iso_range_match = re.search(iso_range_pattern, cleaned, re.IGNORECASE)
    if iso_range_match:
        start_date = iso_range_match.group(1) + "T00:00:00"
        end_date = iso_range_match.group(2) + "T23:59:59"
        cleaned = re.sub(iso_range_pattern, "", cleaned, flags=re.IGNORECASE)
        intent = "temporal" if intent != "attributed" else "hybrid"
    else:
        # Check single ISO date
        iso_single = r"\b(\d{4}-\d{2}-\d{2})\b"
        single_match = re.search(iso_single, cleaned)
        if single_match:
            d = single_match.group(1)
            start_date = f"{d}T00:00:00"
            end_date = f"{d}T23:59:59"
            cleaned = re.sub(iso_single, "", cleaned)
            intent = "temporal" if intent != "attributed" else "hybrid"

    # 3. Extract named months (e.g., "in March", "around February")
    if not start_date:
        for month_name, month_num in MONTH_MAP.items():
            month_pattern = rf"\b(?:in|around|back in|during|for)?\s*({month_name})(?:\s+(\d{4}))?\b"
            m = re.search(month_pattern, cleaned, re.IGNORECASE)
            if m:
                year = int(m.group(2)) if m.group(2) else default_year
                _, last_day = calendar.monthrange(year, month_num)
                start_date = f"{year:04d}-{month_num:02d}-01T00:00:00"
                end_date = f"{year:04d}-{month_num:02d}-{last_day:02d}T23:59:59"
                cleaned = re.sub(month_pattern, "", cleaned, flags=re.IGNORECASE)
                intent = "temporal" if intent != "attributed" else "hybrid"
                break

    # 4. Clean up common conversational filler phrases
    filler_patterns = [
        r"\bwhat did we discuss (about|on|regarding)?\b",
        r"\bwhat was discussed (about|on|regarding)?\b",
        r"\bwhat was the final call on\b",
        r"\bfind (the|any)?\b",
        r"\bshow me (the|any)?\b",
        r"\bsearch for\b",
        r"\bwho (said|mentioned|suggested)\b",
        r"\bwhich message mentions\b",
        r"\btell me about\b",
        r"\bwhat did\b",
        r"\bany discussions? on\b",
        r"\bany messages? about\b",
    ]
    for fp in filler_patterns:
        cleaned = re.sub(fp, " ", cleaned, flags=re.IGNORECASE)

    # Normalize whitespace and punctuation
    cleaned = re.sub(r"[?!.,'\"]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # If query was emptied out (e.g., "What did we discuss in March?"), restore original
    if not cleaned:
        cleaned = query.strip()

    return ParsedQuery(
        raw_query=query,
        clean_query=cleaned,
        sender=extracted_sender,
        start_date=start_date,
        end_date=end_date,
        inferred_intent=intent,
    )

